"""Script to generate a mask for a single object."""

import sys
import os
import yaml
import h5py
import numpy as np
from typing import Tuple
from warnings import warn
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI


from pdb import set_trace

# ---------------------------------------
# Load utilities from `modules` directory
# ---------------------------------------

# Append tools directory to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "tools"
    )
)
import crossref as xr

# Load utilities, with safety checks to make sure that they are accessible
try:
    from peano import peano_hilbert_key_inverses
except ImportError:
    raise Exception(
        "Make sure you have added the `peano.py` module directory to your "
        "$PYTHONPATH.")
try:
    from read_swift import read_swift
except ImportError:
    raise Exception(
        "Make sure you have added the `read_swift.py` module directory to "
        "your $PYTHONPATH.")

# Set up MPI support. We do this at a global level, so that all functions
# can access the communicator easily
comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size

# Set up Matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Palatino'


class MakeMask:
    """
    Class to construct and save a mask.

    Upon instantiation, the parameter file is read, the mask is created
    and (by default) immediately saved to an HDF5 file.

    The mask data is stored in three internal attributes:
    - self.cell_coords : ndarray(float) [N_sel, 3]
        An array containing the relative 3D coordinates of the centres of
        N_sel cubic cells. The volume within these cells is to be
        re-simulated at high resolution. The origin of this coordinate system
        is in general not the same as for the parent simulation
        (see `self.mask_centre`).
    - self.cell_size : float
        The side length each mask cell.
    - self.mask_centre : ndarray(float) [3]
        The origin of the mask coordinate system in the parent simulation
        frame. In other words, these coordinates must be added to
        `self.cell_coords` to obtain the cell positions in the parent
        simulation. It is chosen as the geometric centre of the mask cells,
        i.e. `self.cell_coords` extends equally far in the positive and
        negative direction along each axis.

    The class also contains a `plot` method for generating an overview
    plot of the generated mask.

    Parameters
    ----------
    param_file : string, optional
        The name of the YAML parameter file defining the mask. If None
        (default), a parsed parameter structure must be provided as
        `params` instead.
    params : dict, optional
        The input parameters
    save : bool
        Switch to directly save the generated mask as an HDF5 file. This is
        True by default.

    Returns
    -------
    None
    """

    def __init__(self, param_file=None, params=None, save=True, plot=True):

        # Parse the parameter file, check for consistency, and determine
        # the centre and radius of high-res sphere around a VR halo if desired.
        self.read_param_file(param_file, params)
        
        # Create the actual mask...
        self.make_mask()

        # If desired, plot the mask and the amoeba. The actual plot is only
        # made by rank 0, but we also need the particle data from the other
        # ranks.
        if plot:
            self.plot()

        # Save the mask to hdf5
        if save and comm_rank == 0:
            self.save()

    def read_param_file(self, param_file, params):
        """
        Read parameters from a specified YAML file.

        See template file `param_files/template.yml` for a full listing and
        description of parameters. The values are stored in the internal
        `self.params` dict.

        If `params` is a dict, this is assumed to represent the parameter
        structure instead, and `param_file` is ignored. Parameters are still
        checked and processed in the same way. Otherwise, `params` is ignored.

        If the parameter file specifies the target centre in terms of a
        particular Velociraptor halo, the centre and radius of the high-res
        region are determined internally.

        In the MPI version, the file is only read on one rank and the dict
        then broadcast to all other ranks.

        """
        if comm_rank == 0:

            if not isinstance(params, dict):
                params = yaml.safe_load(open(param_file))

            # Set default values for optional parameters
            self.params = {}
            self.params['min_num_per_cell'] = 3
            self.params['cell_size_mpc'] = 3.
            self.params['topology_fill_holes'] = True
            self.params['topology_dilation_niter'] = 0
            self.params['topology_closing_niter'] = 0
            self.params['phid_name'] = 'PeanoHilbertIDs'
            self.params['padding_snaps'] = None
            self.params['highres_padding_width'] = 0
            
            # Define a list of parameters that must be provided. An error
            # is raised if they are not found in the YAML file.
            required_params = [
                'fname',
                'snap_file',
                'ids_are_ph',
                'bits',
                'data_type',
                'divide_ids_by_two',
                'select_from_vr',
                'output_dir'
            ]
            for att in required_params:
                if att not in params:
                    raise KeyError(
                        f"Need to provide a value for {att} in the parameter "
                        f"file '{param_file}'!")

            # Run checks for automatic group selection
            if params['select_from_vr']:
                params['shape'] = 'sphere'
                requirements = [
                    ('group_number', 'a group number to select'),
                    ('vr_file',
                     'a Velociraptor catalogue to select groups from'),
                    ('sort_type', 'the method for halo sorting')
                ]
                for req in requirements:
                    if not req[0] in params:
                        raise KeyError(f"Need to provide {req[1]}!")

                # Make sure that we have a positive high-res region size
                if 'highres_radius_r200' not in params:
                    params['highres_radius_r200'] = 0
                if 'highres_radius_r500' not in params:
                    params['highres_radius_r500'] = 0
                if max(params['highres_radius_r200'],
                       params['highres_radius_r500']) <= 0:
                    raise KeyError(
                        "At least one of 'highres_radius_r200' and "
                        "highres_radius_r500' must be positive!")

                # Set defaults for optional parameters
                self.params['highres_radius_min'] = 0
                self.params['highres_radius_buffer'] = 0
                self.params['target_mass'] = None
                
            else:
                # Consistency checks for manual target region selection
                if 'centre' not in params:
                    raise KeyError(
                        "Need to provide coordinates for the centre of the "
                        "high-resolution region.")
                if 'shape' not in params:
                    raise KeyError(
                        "Need to specify the shape of the target region!")
                if (params['shape'] in ['cuboid', 'slab']
                    and 'dim' not in params):
                    raise KeyError(
                        f"Need to provide dimensions of {params['shape']} "
                        f"high-resolution region.")
                if params['shape'] == 'sphere' and 'radius' not in params:
                    raise KeyError(
                        "Need to provide the radius of target high-resolution "
                        "sphere!")

            # Load all parameters into the class
            for att in params.keys():
                self.params[att] = params[att]

            # If desired, find the halo to center the high-resolution on
            # and the *unpadded* target high-resolution sphere radius.
            if self.params['select_from_vr']:
                self.params['centre'], self.params['hr_radius'], vr_index = (
                    self.find_highres_sphere())

            self.params['fname'] = (
                self.params['fname'].replace('$vr', f'{vr_index}'))
            self.params['output_dir'] = (
                self.params['output_dir'].replace('$vr', f'{vr_index}'))

            # Parse padding options, if provided
            if self.params['padding_snaps'] is not None:
                pad_snaps = self.params['padding_snaps'].split()
                if len(pad_snaps) == 1:
                    self.params['padding_snaps'] = np.array([pad_snaps[0]])
                else:
                    pad_start = pad_snaps[0]
                    pad_end = pad_snaps[1]
                    if len(pad_snaps) == 2:
                        pad_space = 1
                    self.params['padding_snaps'] = np.arange(
                        pad_start, pad_end+1, pad_space)
                    
            # Convert coordinates and cuboid/slab dimensions to ndarray
            self.params['centre'] = np.array(self.params['centre'], dtype='f8')
            if 'dim' in self.params:
                self.params['dim'] = np.array(self.params['dim'])

            # Create the output directory if it does not exist yet
            if not os.path.isdir(self.params['output_dir']):
                os.makedirs(self.params['output_dir'])
                
        else:
            # If this is not the root rank, don't read the file.
            self.params = None

        # Broadcast the read and processed dict to all ranks.
        self.params = comm.bcast(self.params)

    def find_highres_sphere(self) -> Tuple[np.ndarray, float, int]:
        """
        Determine the centre and radius of high-res sphere from Velociraptor.

        The selection is made based on the location of the halo in the
        catalogue, optionally after sorting them by M200c or M500c. This
        is determined by the value of `self.params['sort_type']`.

        This function is only executed by the root rank.

        Returns
        -------
        centre : ndarray(float)
            3-element array holding the centre of the high-res region.
        radius : float
            The target radius of the high-res region, without padding.
        vr_index : int
            The index of the VR halo.

        """
        # Make sure that we are on the root rank if over MPI
        if comm_rank != 0:
            raise ValueError(
                f"find_highres_sphere() called on MPI rank {comm_rank}!")

        # Look up the target halo index in Velociraptor catalogue
        vr_index = self.find_halo_index()

        with h5py.File(self.params['vr_file'], 'r') as vr_file:

            # First, determine the radius of the high-res region
            r_r200 = 0
            r_r500 = 0
            r200 = vr_file['R_200crit'][vr_index]
            self.params['r200'] = r200
            if self.params['highres_radius_r200'] > 0:
                r_r200 = r200 * self.params['highres_radius_r200']
            try:
                r500 = vr_file['SO_R_500_rhocrit'][vr_index]
                r_r500 = r500 * self.params['highres_radius_r500']
            except KeyError:
                r500 = None
                if self.params['highres_radius_r500'] > 0:
                    warn("Could not load R500c, ignoring request for "
                         f"minimum high-res radius of "
                         f"{self.params['highres_radius_r500']} r_500.",
                         RuntimeWarning)

            # Load halo centre
            names = ['X', 'Y', 'Z']
            centre = np.zeros(3)
            for icoord, prefix in enumerate(names):
                centre[icoord] = vr_file[f'{prefix}cminpot'][vr_index]

        r_highres = max(r_r200, r_r500)
        if r_highres <= 0:
            raise ValueError(
                f"Invalid radius of high-res region ({r_highres})")

        # If enabled, expand radius to requested minimum.
        if self.params['highres_radius_min'] > 0:
            r_highres = max(r_highres, self.params['highres_radius_min'])

        r500_str = '' if r500 is None else f'{r500:.4f}'
        m200_str = (
            '' if getattr(self, 'm200crit', None) is None else
            f'{self.m200crit:.4f}')
        m500_str = (
            '' if getattr(self, 'm500crit', None) is None else
            f'{self.m500crit:.4f}')
        print(
            "Velociraptor search results:\n"
            f"- Run name: {self.params['fname']}\t"
            f"GroupNumber: {self.params['group_number']}\t"
            f"VR index: {vr_index}\n"
            f"- Centre: {centre[0]:.3f} / {centre[1]:.3f} / {centre[2]:.3f} "
            f"- High-res radius: {r_highres:.4f}\n"
            f"- R_200crit: {r200:.4f}\n"
            f"- R_500crit: {r500_str}\n"
            f"- M_200crit: {m200_str}\n"
            f"- M_500crit: {m500_str}\n"
            )

        return centre, r_highres, vr_index

    def find_halo_index(self) -> int:
        """
        Find the index of the desired target halo.

        This function looks up the desired (field) halo if the selection
        is specified in terms of a position in the mass-ordered list.
        It should only ever be run on the root node, an error is raised if
        this is not the case.

        If the parameter file instructs to sort by M500c, but this is not
        recorded in the Velociraptor catalogue, an error is raised.

        Parameters
        ----------
        None

        Returns
        -------
        halo_index : int
            The catalogue index of the target halo.

        """
        if comm_rank != 0:
            raise ValueError("find_halo_index() called on rank {comm_rank}!")

        # If the parameter file already specified the VR index, we are done
        if self.params['sort_type'].lower() == "none":
            return self.params['group_number']            
        
        # ... otherwise, need to load the desired mass type of all (central)
        # VR haloes, sort them, and find the entry we want
        with h5py.File(self.params['vr_file'], 'r') as vr_file:
            structType = vr_file['/Structuretype'][:]
            field_halos = np.where(structType == 10)[0]

            sort_type = self.params['sort_type'].lower()
            if sort_type == 'm200crit':
                m_halo = vr_file['/Mass_200crit'][field_halos]
            elif sort_type == 'm500crit':
                # If M500 cannot be loaded, an error will be raised
                m_halo = vr_file['/SO_Mass_500_rhocrit'][field_halos]
            else:
                raise ValueError("Unknown sorting rule '{sort_type}'!")

        # If we want to get close to a given value:
        m_target = self.params['target_mass']
        if m_target is not None:
            m_target = float(m_target) / 1e10
            halo_index = np.argmin(np.abs(m_halo - m_target))
            
        # ... else sort groups by specified mass, in descending order
        else:
            sort_key = np.argsort(-m_halo)
            halo_index = sort_key[self.params['group_number']]

        # Store mass of target halo used for sorting, for later use
        setattr(self, sort_type, m_halo[halo_index])

        return field_halos[halo_index]

    def make_mask(self, padding_factor=2.0):
        """
        Main driver function to create a mask from a given snapshot file.

        This assumes that the centre and extent of the high-res region
        have already been determined, either from the parameter file or
        from the Velociraptor halo catalogue - in the latter case, this is
        done inside `read_param_file()`.

        Note that only MPI rank 0 contains the final mask, as an attribute
        `self.mask`.

        Parameters
        ----------
        padding_factor : float
            The mask is set up to extend beyond the region covered by the
            target particles by this factor. Default is 2.0, must be >= 1.0.

        """
        if padding_factor < 1:
            raise ValueError(
                f"Invalid value of padding_factor={padding_factor}!")

        # Find cuboidal frame enclosing the target high-resolution region
        # and any padding around it
        self.region = self.find_enclosing_frame()

        # Load IDs of all possibly relevant particles, and the (sub-)indices
        # of those lying in the target region and within the surrounding
        # padding zone. Only particles assigned to the current MPI rank are "
        # loaded, which may be none.
        ids, inds_target, inds_pad = self.load_primary_ids()

        # If desired, identify additional padding particles in other snapshots.
        # This is currently only possible in non-MPI runs.
        if (self.params['padding_snaps'] is not None and
            not self.params['pad_full_cells']):
            inds_pad = self.find_extra_pad_particles(
                ids, inds_target, inds_pad)
        
        # Find initial (Lagrangian) positions of particles from their IDs
        # (recall that these are really Peano-Hilbert indices).
        # Coordinates are in the same units as the box size, centred "
        # on the high-res target region and periodically wrapped.
        self.lagrangian_coords = self.compute_ic_positions(ids)

        # Find the corners of a box enclosing all particles in the ICs that
        # are so far identified as "to be treated as high-res"
        box, widths = self.compute_bounding_box(inds_target, inds_pad)

        if comm_rank == 0:
            print(
                f"Determined bounding box edges in ICs (re-centred):\n"
                f"\t{box[0, 0]:.3f} / {box[0, 1]:.3f} / {box[0, 2]:.3f} --> "
                f"{box[1, 0]:.3f} / {box[1, 1]:.3f} / {box[1, 2]:.3f}")

        # For simplicity, shift the coordinates relative to geometric box
        # center, so that particles extend equally far in each direction
        geo_centre = box[0, :] + widths / 2
        self.lagrangian_coords -= geo_centre
        self.lagrangian_coords = ic_coords

        # Also need to keep track of the mask centre in the original frame.
        self.mask_centre = self.params['centre'] + geo_centre

        # Build the basic mask. This is a cubic boolean array with an
        # adaptively computed cell size and extent that includes at least
        # twice the entire bounding box. It is True for any cells that contain
        # at least the specified threshold number of particles.
        #
        # `edges` holds the spatial coordinate of the lower cell edges. By
        # construction, this is the same along all three dimensions.
        #
        # We make the mask larger than the actual particle extent, as a safety
        # measure (**TODO**: check whether this is actually needed)
        # --> Some extra is needed to allow for re-shaping. Do later.
        self.target_mask, self.full_mask = self.build_basic_mask(
            lagrangian_coords, inds_target, inds_pad, box)

        self.cell_size = edges[0][1] - edges[0][0]

        # Now apply "stage-2 padding" (for entire target cells)
        if self.params['pad_full_cells']:
            full_target_inds = self.find_full_target_indices()
            inds_pad = self.find_extra_pad_particles(
                ids, full_target_inds, inds_pad, with_primary_snapshot=True)

        # Expand the mask with updated padding particles. If applicable, this
        # also adds entire high-res padding cells around those hosting target
        # particles. The mask is enlarged appropriately to allow refinement.
        self.full_mask.expand(self.lagrangian_coords[inds_pad, :])

        # We only need MPI rank 0 for the rest, since we are done working with
        # individual particles
        if comm_rank > 0:
            return

        # Fill holes and extrude the mask. This has to be done separately
        # for each of the three projections.
        for idim, name in enumerate(['x-y', 'y-z', 'x-z']):
            print(f"Topological extrision ({idim}/3, {name} plane)...")
            self.full_mask.refine(idim, self.params)

        self.full_mask.compute_box()
        mask_offset = self.full_mask.recenter()
        self.lagrangian_coords -= mask_offset

        self.full_mask.get_active_cell_centres()


    def find_enclosing_frame(self):
        """
        Compute the bounding box enclosing the target high-res region.

        This is only used to pre-select a region for particle reading from
        the snapshot, to make things more efficient.

        Returns
        -------
        frame : np.ndarray(float)
            A 2x3 element array containing the lower and upper coordinates
            of the bounding region in the x, y, and z coordinates.

        """
        centre = self.params['centre']
        pad_width = max(self.params['highres_padding_width'],
                        self.params['highres_diffusion_buffer'])
        frame = np.zeros((2, 3))

        # If the target region is a sphere, find the enclosing cube
        if self.params['shape'] == 'sphere':
            frame[0, :] = centre - (self.params['hr_radius'] + pad_width)
            frame[1, :] = centre + (self.params['hr_radius'] + pad_width)

        # If the target region is a cuboid, simply transform from centre and
        # side length to lower and upper bounds along each coordinate
        elif self.params['shape'] in ['cuboid', 'slab']:
            frame[0, :] = centre - (self.params['dim'] / 2. + pad_width)
            frame[1, :] = centre + (self.params['dim'] / 2. + pad_width)

        else:
            raise ValueError(f"Invalid shape {self.params['shape']}!")

        if comm_rank == 0:
            print(
                f"Boundary frame in selection snapshot:\n"
                f"{frame[0, 0]:.2f} / {frame[0, 1]:.2f} / {frame[0, 2]:.2f}"
                " -- "
                f"{frame[1, 0]:.2f} / {frame[1, 1]:.2f} / {frame[1, 2]:.2f}"
            )
        return frame

    def load_primary_ids(self):
        """
        Load IDs of particles in and near the specified high-res region.

        This includes both "target" particles that are within the specified
        region in the primary snapshot, and "padding" particles that surround
        the target high-res region.

        If run on multiple MPI ranks, this only load the particles belonging
        to the current rank, which may be none.

        In addition, relevant metadata are loaded and stored in the
        `self.params` dict.        

        Returns
        -------
        ids_target : ndarray(int)
            The particle IDs of the primary target particles (i.e. those in
            the target high-res region).
        ids_pad : ndarray(int)
            The IDs of additional padding particles that should also be
            treated as high-res, but are not within the target region. This
            may be empty, especially if self.params['highres_padding_width']
            is zero.

        """
        # To make life simpler, extract some frequently used parameters
        cen = self.params['centre']
        shape = self.params['shape']

        # First step: set up particle reader and load metadata.
        # Currently, only SWIFT input is supported, may add others later...
        if self.params['data_type'].lower() != 'swift':
            raise ValueError(
                f"Only SWIFT input supported, not {self.params['data_type']}.")

        elif self.params['data_type'].lower() == 'swift':
            snap = read_swift(self.params['snap_file'], comm=comm)
            self.params['box_size'] = float(snap.HEADER['BoxSize'])
            self.params['h_factor'] = float(snap.COSMOLOGY['h'])
            self.params['length_unit'] = 'Mpc'
            self.params['redshift'] = snap.HEADER['Redshift']
            snap.select_region(1, *self.region.T.flatten())
            snap.split_selection()

        if comm_rank == 0:
            self.zred_snap = self.params['redshift']
            print(f"Snapshot is at redshift z = {self.zred_snap:.2f}.")

        # Load DM particle IDs and coordinates (uniform across GADGET/SWIFT)
        if comm_rank == 0:
            print("\nLoading particle data...")
        coords = snap.read_dataset(1, 'Coordinates')

        # Shift coordinates relative to target centre, and wrap them to within
        # the periodic box
        cen = self.params['centre']
        coords -= cen
        periodic_wrapping(coords, self.params['box_size'])

        # Select particles within target region
        l_unit = self.params['length_unit']
        ind_primary = None

        if shape == 'sphere':
            if comm_rank == 0:
                print(f"Clipping to sphere around {cen}, with radius "
                      f"{self.params['hr_radius']:.4f} {l_unit}")

            dists = np.linalg.norm(coords, axis=1)
            ind_target = np.where(dists <= self.params['hr_radius'])[0]
            r_padded = (self.params['hr_radius'] +
                        self.params['highres_padding_width'])
            ind_pad = np.nonzero(
                (dists > self.params['hr_radius']) & (dists <= r_padded))[0]

        elif self.params['shape'] in ['cuboid', 'slab']:
            if comm_rank == 0:
                print(f"Clipping to {shape} with "
                      f"dx={self.params['dim'][0]:.2f} {l_unit}, "
                      f"dy={self.params['dim'][1]:.2f} {l_unit}, "
                      f"dz={self.params['dim'][2]:.2f} {l_unit}\n"
                      f"around {cen} {l_unit}.")

            # To find particles within target cuboid, normalize each coordinate
            # offset by the maximum allowed extent in the corresponding
            # dimension, and find those where the result is between -1 and 1
            # for all three dimensions
            l_target = self.params['dim'] / 2
            l_padded = l_target + self.params['highres_padding_width']
            ind_target = np.where(
                np.max(np.abs(coords / l_target), axis=1) <= 1)[0]
            ind_pad = np.where(
                (np.max(np.abs(coords / l_target), axis=1) > 1) &
                (np.max(np.abs(coords / l_padded), axis=1) <= 1)
            )[0]
        else:
            raise ValueError(f"Invalid shape {self.params['shape']}")

        # We need the IDs of particles lying in the mask region
        ids = snap.read_dataset(1, 'ParticleIDs')

        # If IDs are Peano-Hilbert indices multiplied by two (as in e.g.
        # simulations with baryons), need to undo this multiplication here
        if self.params['divide_ids_by_two']:
            ids //= 2

        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles')

        # Make a plot of the selection environment
        self.plot_halo(coords)

        return ids, ind_target, ind_pad
    
    def load_mask_ids(self, primary_ids) -> np.ndarray:
        """
        Find the IDs of all particles that must be included in the mask.

        This includes the primary particles, and in addition all those that
        are close enough to them that they would affect the target region.
        Padding particles can be identified in multiple snapshots.

        Parameters
        ----------
        primary_ids : ndarray(int)
            The IDs of primary particles.

        Returns
        -------
        mask_ids : ndarray(int)
            The (unique) IDs of all particles to be included in the mask.

        """
        pass
        
    def compute_ic_positions(self, ids) -> np.ndarray:
        """
        Compute the particle positions in the ICs.

        This exploits the fact that the particle IDs are set to Peano-Hilbert
        keys that encode the positions in the ICs.

        Parameters
        ----------
        ids : ndarray(int)
            The Particle IDs for which to calculate IC coordinates. If they
            are not Peano-Hilbert indices themselves, these are retrieved
            fro the ICs file.

        Returns
        -------
        coords : ndarray(float)
            The coordinates of particles in the ICs [Mpc]. They are shifted
            (and wrapped) such that the centre of the high-res region is at
            the origin.
        """
        print(f"[Rank {comm_rank}] Computing initial positions of dark matter "
              "particles...")

        if not self.params['ids_are_ph']:
            print("Translating particle IDs to PH IDs...", end='', flush=True)
            with h5py.File(self.params['ics_file'], 'r') as f:
                ics_ph_ids = f[f"PartType1/{self.params['phid_name']}"][...]
                ids = ics_ph_ids[ids-1]

            print(" done.")

        # First, convert the (scalar) PH key for each particle back to a triple
        # of indices giving its (quantised, normalized) offset from the origin
        # in x, y, z. This must use the same grid (bits value) as was used
        # when generating the ICs for the base simulation. An external utility
        # function is used to handle the PH algorithm.
        x, y, z = peano_hilbert_key_inverses(ids, self.params['bits'])
        ic_coords = np.vstack((x, y, z)).T
        
        # Make sure that we get consistent values for the coordinates
        ic_min, ic_max = np.min(ic_coords), np.max(ic_coords)
        if ic_min < 0 or ic_max > 2**self.params['bits']:
            raise ValueError(
                f"Inconsistent range of quantized IC coordinates: {ic_min} - "
                f"{ic_max} (allowed: 0 - {2**self.params['bits']})"
                )

        # Re-scale quantized coordinates to floating-point distances between
        # origin and centre of corresponding grid cell
        cell_size = self.params['box_size'] / 2**self.params['bits']
        ic_coords = (ic_coords.astype('float') + 0.5) * cell_size

        # Shift coordinates to the centre of target high-resolution region and
        # apply periodic wrapping
        ic_coords -= self.params['centre']
        periodic_wrapping(ic_coords, self.params['box_size'])

        return ic_coords.astype('f8')

    def compute_bounding_box(self, pad=0.0, serial_only=False):
        """
        Find the corners of a box enclosing a set of points across MPI ranks.

        Parameters
        ----------
        r : ndarray(float) [N_part, 3] or list thereof
            The coordinates of the `N_part` particles held on this MPI rank.
            The second array dimension holds the x/y/z components per point.
            If a list, this is assumed to hold multiple such arrays and the
            box enclosing all of them is determined.
        pad : float, optional
            Padding between outermost point and box edge (default: 0).
        serial_only : bool, optional
            Switch to disable cross-MPI comparison of box extent (default:
            False). If True, the return values will generally differ between
            different MPI ranks.

        Returns
        -------
        box : ndarray(float) [2, 3]
            The coordinates of the lower and upper vertices of the bounding
            box. These are stored in index 0 and 1 along the first dimension,
            respectively.

        widths : ndarray(float) [3]
            The width of the box along each dimension.

        """
        box = np.zeros((2, 3))
        r = self.lagrangian_coords

        # Find vertices of local particles (on this MPI rank). If there are
        # none, set lower (upper) vertices to very large (very negative)
        # numbers so that they will not influence the cross-MPI min/max.
        if not isinstance(r, list):
            rlist = [r]
        for r_curr in rlist:
            n_part = r_curr.shape[0]
            box[0, :] = (np.min(r_curr, axis=0) - pad
                         if n_part > 0 else sys.float_info.max)
            box[1, :] = (np.max(r_curr, axis=0) + pad
                         if n_part > 0 else -sys.float_info.max)

        # Now compare min/max values across all MPI ranks
        if not serial_only:
            for idim in range(3):
                box[0, idim] = comm.allreduce(box[0, idim], op=MPI.MIN)
                box[1, idim] = comm.allreduce(box[1, idim], op=MPI.MAX)

        return box, box[1, :] - box[0, :]

    def build_basic_mask(self, r, inds_target, inds_pad, mask_box):
        """
        Build the basic mask for an input particle distribution.

        This is a cubic boolean array with an adaptively computed cell size and
        extent that stretches by at least `min_width` in each dimension.
        The mask value is True for any cells that contain at least the
        specified threshold number of particles.

        The mask is based on particles on all MPI ranks.

        Parameters
        ----------
        r_target : ndarray(float) [N_p, 3]
            The coordinates of (local) particles for which to create the mask.
            They must be shifted such that they lie within +/- `min_width` from
            the origin in each dimension.
        r_pad : ndarray(float) [N_p, 3]
            As r_target, but holding the particles that are only included for
            padding.
        mask_box : ndarray(float) [3]
            The full side length of the mask along each dimension (must be
            identical across MPI ranks). It is clipped to the box size.

        Returns
        -------
        target_mask : Mask object
            The mask for the target region (without padding).
        full_mask : Mask object
            The mask for the entire high-resolution region (with padding).
        
        """
        # Find out how far from the origin we need to extend the mask
        widths = np.clip(mask_box, 0, self.params['box_size'])

        # Work out how many cells we need along each dimension so that the
        # cells remain below the specified threshold size
        num_cells = int(np.ceil(widths / self.params['cell_size_mpc']))

        # Compute number of particles in each cell, across MPI ranks
        n_p_target, edges = np.histogramdd(
            r[inds_target, :], bins=num_cells,
            range=[(-w/2, w/2) for w in widths]
        )
        n_p_pad, edges = np.histogramdd(
            r[inds_pad, :], bins=num_cells,
            range=[(-w/2, w/2) for w in widths]
        )
        n_p_target = comm.allreduce(n_p_target, op=MPI.SUM)
        n_p_full = comm.allreduce(n_p_target + n_p_pad, op=MPI.SUM)

        # Convert particle counts to True/False mask
        mask_target = n_p_target >= self.params['min_num_per_cell']
        mask_full = n_p_full >= self.params['min_num_per_cell']

        return Mask(mask_target, edges), Mask(mask_full, edges)


    def find_full_target_indices(self):
        """Find all particles that lie within target mask cells.

        Parameters
        ----------
        lagrangian_coords : ndarray(float) [N_p, 3]
            The Lagrangian coordinates of all possibly relevant particles.

        Returns
        -------
        indices : ndarray(int)
            The indices (within `lagrangian_coords`) of those particles that
            occupy target cells.

        """
        cell_indices = coordinate_to_cell(
            self.lagrangian_coords, self.cell_size, self.target_mask.shape)

        # First cut: particles that are in valid mask region
        ind_in_mask = np.nonzero(
            (np.min(cell_indices, axis=1) >= 0) &
            (cell_indices[:, 0] < self.target_mask.shape[0]) &
            (cell_indices[:, 1] < self.target_mask.shape[1]) &
            (cell_indices[:, 2] < self.target_mask.shape[2])
        )[0]

        # Second cut: particles that are in activated mask cells
        subind_target = np.nonzero(
            self.target_mask[
                cell_indices[ind_in_mask, 0],
                cell_indices[ind_in_mask, 1],
                cell_indices[ind_in_mask, 2]
            ]
        )[0]

        print(f"Out of {self.lagrangian_coords.shape[0]} particles, "
              f"{len(subind_target)} lie in a target mask cell.")

        return ind_in_mask[subind_target]

    def find_extra_pad_particles(
        self, ids, inds_target, inds_pad, with_primary_snapshot=False):
        """
        Identify particles close to target particles in a snapshot.

        Parameters
        ----------
        ids : ndarray(int)
            The IDs of all potentially relevant particles.
        inds_target : ndarray(int)
            The indices (within `ids`) of target particles, i.e. those whose
            neighbours are to be found.
        inds_pad : ndarray(int)
            The indices (within `ids`) of already identified neighbours.
        with_primary_snapshot : bool, optional
            Should the primary snapshot be included in the search? Default: no.

        Returns
        -------
        inds_pad : ndarray(int)
            Updated list of neighbour indices. This is guaranteed to include
            at least thosee particles that were in the list on input.

        Notes
        -----
        This function is not (yet?) MPI parallelized. An error will be thrown
        if it is called with more than one MPI rank.

        Since the location of the particles cannot be known in advance, the
        entire coordinate array has to be read in (except for the primary
        snapshot, if included).

        """
        if comm_size > 1:
            raise ValueError("Finding extra padding particles does not work "
                             "over MPI. Sorry.")

        snaps = self.params['padding_snaps']
        if with_primary_snapshot:
            snaps = np.concatenate(([self.params['primary_snapshot']], snaps))

        for snap in snaps:
            snap_file = self.params['snapshot_base'] + f'{snap:04d}.hdf5'
            with h5.File(snap_file, 'r') as f:
                snap_ids = f['PartType1/ParticleIDs'][...]
                snap_pos = f['PartType1/Coordinates'][...]

            inds, in_snap = xr.find_id_indices(ids, snap_ids)
            if len(in_snap) < len(inds):
                raise ValueError(
                    f"Could only locate {len(in_snap)} out of "
                    f"{len(inds)} particles in snapshot {snap}!"
                )
            r = snap_pos[inds, :]
            r_target = r[inds_target, :]

            target_tree = cKDTree(r_target, boxsize=self.params['box_size'])
            ngb_tree = cKDTree(r, boxsize=self.params['box_size'])

            ngbs_lol = ngb_tree.query_ball_tree(
                target_tree, r=self.params['highres_padding_width'])
            is_tagged = np.zeros(len(inds), dtype=bool)
            for ii in len(inds):
                is_tagged[ngbs_lol[ii]] = True

            inds_tagged = np.nonzero(is_tagged)[0]
            inds_pad = np.unique(np.concatenate((inds_pad, inds_tagged)))

        return inds_pad

    def expand_full_mask(self, mask, r, inds_pad):
        """Expand the given mask by the particles specified in r."""



    def plot(self, max_npart_per_rank=int(1e5)):
        """
        Make an overview plot of the zoom-in region.

        Note that this function must be called on all MPI ranks, even though
        only rank 0 generates the actual plot. The others are still required
        to access (a subset of) the particles stored on them.

        """
        axis_labels = ['x', 'y', 'z']

        # Select a random sub-sample of particle coordinates on each rank and
        # combine them all on rank 0
        np_ic = self.ic_coords.shape[0]
        n_sample = int(min(np_ic, max_npart_per_rank))
        indices = np.random.choice(np_ic, n_sample, replace=False)
        plot_coords = self.ic_coords[indices, :]
        plot_coords = comm.gather(plot_coords)

        # Only need rank 0 from here on, combine all particles there.
        if comm_rank != 0:
            return
        plot_coords = np.vstack(plot_coords)

        # Extract frequently needed attributes for easier structure
        bound = self.mask_extent
        cell_size = self.cell_size

        fig, axarr = plt.subplots(1, 3, figsize=(13, 4))

        # Plot each projection (xy, xz, yz) in a separate panel. `xx` and `yy`
        # denote the coordinate plotted on the x and y axis, respectively.
        for ii, (xx, yy) in enumerate(zip([0, 0, 1], [1, 2, 2])):
            ax = axarr[ii]
            ax.set_aspect('equal')

            # Draw the outline of the cubic bounding region
            rect = patches.Rectangle(
                [-bound / 2., -bound/2.], bound, bound,
                linewidth=1, edgecolor='maroon', facecolor='none')
            ax.add_patch(rect)

            # Draw on outline of cuboidal bounding region
            box_corners = [self.mask_box[:, xx], self.mask_box[:, yy]]
            ax.plot(
                box_corners[0][[0, 1, 1, 0, 0]],
                box_corners[1][[0, 0, 1, 1, 0]],
                color='maroon', linestyle='--', linewidth=0.7
            )

            # Plot particles.
            ax.scatter(
                plot_coords[:, xx], plot_coords[:, yy],
                s=0.5, c='blue', zorder=-100, alpha=0.3)

            ax.set_xlim(-bound/2. * 1.05, bound/2. * 1.05)
            ax.set_ylim(-bound/2. * 1.05, bound/2. * 1.05)

            # Plot (the centres of) selected mask cells.
            ax.scatter(
                self.cell_coords[:, xx], self.cell_coords[:, yy],
                marker='x', color='red', s=5, alpha=0.2)

            # Plot cell outlines if there are not too many of them.
            if self.cell_coords.shape[0] < 10000:
                for e_x, e_y in zip(
                    self.cell_coords[:, xx], self.cell_coords[:, yy]):
                    rect = patches.Rectangle(
                        (e_x - cell_size/2, e_y - cell_size/2),
                        cell_size, cell_size,
                        linewidth=0.5, edgecolor='r', facecolor='none',
                        alpha=0.2
                    )
                    ax.add_patch(rect)

            ax.set_xlabel(f"${axis_labels[xx]}$ [Mpc]")
            ax.set_ylabel(f"${axis_labels[yy]}$ [Mpc]")

            # Plot target high-resolution sphere (if that is our shape).
            if self.params['shape'] == 'sphere':
                phi = np.arange(0, 2.0001*np.pi, 0.001)
                radius = self.params['hr_radius']
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='white', linestyle='-', linewidth=2)
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='grey', linestyle='--', linewidth=1)
                if ii == 0:
                    ax.text(
                        0, self.params['hr_radius'],
                        f'z = ${self.zred_snap:.2f}$',
                        color='grey', fontsize=6, va='bottom', ha='center',
                        bbox={'facecolor': 'white', 'edgecolor': 'grey',
                              'pad': 0.25, 'boxstyle': 'round',
                              'linewidth': 0.3}
                    )
        # Save the plot
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
        plotloc = os.path.join(
            self.params['output_dir'], self.params['fname']) + ".png"
        plt.savefig(plotloc, dpi=200)
        plt.close()

    def plot_halo(self, pos):
        """
        Make an overview plot of the selected region.

        Note that this function must be called on all MPI ranks, even though
        only rank 0 generates the actual plot. The others are still required
        to access (a subset of) the particles stored on them.

        """
        axis_labels = ['x', 'y', 'z']

        # Extract frequently needed attributes for easier structure
        frame = self.region - self.params['centre']
        bound = np.max(np.abs(frame))
        try:
            r200 = self.params['r200']
        except AttributeError:
            r200 = None
            
        ind_sel = np.nonzero(np.max(np.abs(pos), axis=1) <= bound)[0]
        hists = np.zeros((3, 200, 200))
        for ii, (xx, yy) in enumerate(zip([0, 0, 1], [1, 2, 2])):
            hists[ii, ...], xedges, yedges = np.histogram2d(
                pos[ind_sel, yy], pos[ind_sel, xx], bins=200,
                range=[[-bound, bound], [-bound, bound]]
            )

        hist_full = np.zeros((3, 200, 200)) if comm_rank == 0 else None
        comm.Reduce([hists, MPI.DOUBLE],
                    [hist_full, MPI.DOUBLE],
                    op=MPI.SUM, root=0)

        # Only need rank 0 from here on, combine all particles there.
        if comm_rank != 0:
            return

        fig, axarr = plt.subplots(1, 3, figsize=(13, 4))

        ind_filled = np.nonzero(hist_full > 0)
        vmin, vmax = np.percentile(hist_full[ind_filled], [1, 99.99])

        # Plot each projection (xy, xz, yz) in a separate panel. `xx` and `yy`
        # denote the coordinate plotted on the x and y axis, respectively.
        for ii, (xx, yy) in enumerate(zip([0, 0, 1], [1, 2, 2])):
            ax = axarr[ii]
            ax.set_aspect('equal')

            ax.imshow(
                np.log10(hist_full[ii, ...] + 1e-2),
                origin='lower', interpolation='none',
                extent=[-bound, bound, -bound, bound], aspect='equal',
                vmin=np.log10(vmin), vmax=np.log10(vmax),
            )
            
            # Draw the outline of the cubic bounding region
            ax.plot((frame[0, xx], frame[1, xx], frame[1, xx],
                     frame[0, xx], frame[0, xx]),
                    (frame[0, yy], frame[0, yy], frame[1, yy],
                     frame[1, yy], frame[0, yy]),
                    linewidth=1, color='maroon'
            )

            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)

            ax.set_xlabel(f"${axis_labels[xx]}$ [Mpc]")
            ax.set_ylabel(f"${axis_labels[yy]}$ [Mpc]")
            
            # Plot target high-resolution sphere (if that is our shape).
            if self.params['shape'] == 'sphere':
                phi = np.arange(0, 2.0001*np.pi, 0.001)
                radius = self.params['hr_radius']
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='white', linestyle='-', linewidth=2,
                        zorder=100)
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='grey', linestyle='--', linewidth=1,
                        zorder=100)
                if r200 is not None:
                    ax.plot(np.cos(phi) * r200, np.sin(phi) * r200,
                            color='white', linestyle=':', linewidth=1,
                            zorder=100)
                
                if ii == 0:
                    ax.text(
                        0, self.params['hr_radius']*0.9,
                        f'z = ${self.zred_snap:.2f}$',
                        color='grey', fontsize=6, va='bottom', ha='center',
                        bbox={'facecolor': 'white', 'edgecolor': 'grey',
                              'pad': 0.25, 'boxstyle': 'round',
                              'linewidth': 0.3}
                    )
        # Save the plot
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.99)
        plotloc = os.path.join(
            self.params['output_dir'], self.params['fname']) + "_selection.png"
        plt.savefig(plotloc, dpi=200)
        plt.close()
        
    def save(self):
        """
        Save the generated mask for further use.

        Only rank 0 should do this, but we'll check just to be sure...

        Returns
        -------
        None

        """
        if comm_rank != 0:
            return

        outloc = os.path.join(
            self.params['output_dir'], self.params['fname']) + ".hdf5"

        with h5py.File(outloc, 'w') as f:

            # Push parameter file data as file attributes
            g = f.create_group('Params')
            for param_attr in self.params:
                g.attrs.create(param_attr, self.params[param_attr])

            if self.params['shape'] in ['cuboid', 'slab']:
                high_res_volume = np.prod(self.params['dim'])
            else:
                high_res_volume = 4 / 3. * np.pi * self.params['hr_radius']**3
            g.attrs.create('high_res_volume', high_res_volume)

            # Main output is the centres of selected mask cells
            self.full_mask.write(f)
            self.target_mask.write(f, subgroup=TargetMask)

        print(f"Saved mask data to file `{outloc}`.")


class Mask:
    """Low-level class to represent one individual mask."""

    def __init__(self, mask_array, mask_edges):
        self.mask = mask_array
        self.edges = mask_edges

        self.shape = np.array(self.mask.shape)
        self.cell_size = self.edges[0][1] - self.edges[0][0]

    def expand(self, r, target_mask=None, cell_padding_size=0,
               refinement_allowance=None):
        """
        Expand the mask to accommodate additional particles and cells.

        In addition to the specified particles, two optional expansions can
        be performed: (i) cells within a given distance from a target

        """

        cell_indices = coordinate_to_cell(r, self.cell_size, self.shape)

        if cell_padding_size > 0:
            if target_mask is None:
                raise ValueError("Must provide target mask!")
            cell_pad_extra = cell_padding_size*2

        if refinement_allowance is not None:
            ref_extra = np.ceil(target_mask.shape * refinement_allowance[0])
            ref_extra_min = refinement_allowance[1]
            ref_extra = np.maximum(ref_extra, ref_extra_min).astype(int)

        uniform_extra = np.maximum(cell_pad_extra, ref_extra)

        extra_low = np.abs(np.min(cell_indices, axis=0))
        extra_low = np.maximum(extra_low, uniform_extra)
        extra_high = np.max(cell_indices, axis=0) - self.shape
        extra_high = np.maximum(extra_high, uniform_extra)

        new_shape = self.shape + extra_low + extra_high

        new_edges = []
        for idim in range(3):
            new_min = self.edges[idim][0] - self.cell_size * extra_low
            new_max = self.edges[idim][-1] + self.cell_size * extra_high
            new_edges.append(np.arange(new_min, new_max, self.cell_size))

        # Compute number of particles in each cell, across MPI ranks
        n_p, edges = np.histogramdd(r, bins=new_edges)
        n_p = comm.allreduce(n_p, op=MPI.SUM)

        # Convert particle counts to True/False mask
        new_mask = n_p >= self.params['min_num_per_cell']

        # Add in old mask
        new_mask[extra_low[0]:-extra_high[0],
                 extra_low[1]:-extra_high[1],
                 extra_low[2]:-extra_high[2]] += self.mask

        # If desired, activate padding cells
        if cell_padding_size > 0:
            all_cell_centres = self.get_cell_centres()
            target_cell_centres = target_mask.get_cell_centres()
            max_dist = cell_padding_size + np.sqrt(3) * self.cell_size

            tree = cKDTree(all_cell_centres)    # No wrapping necessary here!

            for ix in range(target_mask.shape[0]):
                for iy in range(target_mask.shape[1]):
                    for iz in range(target_mask.shape[2]):

                        # Only expand around active target mask cells!
                        if not target_mask.mask[ix, iy, iz]:
                            continue

                        r_target = target_cell_centres[ix, iy, iz]
                        ngbs = tree.query_ball_point(r_target, max_dist)
                        ngbs_3d = np.unravel_index(ngbs, new_shape)
                        new_mask[ngbs_3d] = True 

    def get_cell_centres(self, active_only=False)
        """Find the centres of all cells (optionally: active ones only)."""
        all_cell_centres_grid = np.meshgrid(
            (self.edges[0][1:] + self.edges[0][:-1]) / 2,
            (self.edges[1][1:] + self.edges[1][:-1]) / 2,
            (self.edges[2][1:] + self.edges[2][:-1]) / 2
        )

        ncells = np.prod(self.new_shape)
        all_cell_centres = np.zeros((ncells, 3))
        for idim in range(3):
            all_cell_centres[:, idim] = all_cell_centres_grid[idim].ravel()

        return all_cell_centres

    def refine(self, idim, params):
        """
        Refine the mask by checking for holes and processing the morphology.

        The refinement is performed iteratively along all slices along the
        specified axis. It consists of the ndimage operations ...

        The mask array (`mask`) is modified in place.

        Parameters
        ----------
        idim : int
            The perpendicular to which slices of the mask are to be processed
            (0: x, 1: y, 2: z)

        Returns
        -------
        None

        """
        mask = self.mask

        # Process each layer (slice) of the mask in turn
        for layer_id in range(mask.shape[idim]):

            # Since each dimension loops over a different axis, set up an
            # index for the current layer
            if idim == 0:
                index = np.s_[layer_id, :, :]
            elif idim == 1:
                index = np.s_[:, layer_id, :]
            elif idim == 2:
                index = np.s_[:, :, layer_id]
            else:
                raise ValueError(f"Invalid value idim={idim}!")

            # Step 1: fill holes in the mask
            if params['topology_fill_holes']:
                mask[index] = (
                    ndimage.binary_fill_holes(mask[index]).astype(bool)
                )
            # Step 2: regularize the morphology
            if params['topology_dilation_niter'] > 0:
                mask[index] = (
                    ndimage.binary_dilation(
                        mask[index],
                        iterations=params['topology_dilation_niter']
                    ).astype(bool)
                )
            if params['topology_closing_niter'] > 0:
                mask[index] = (
                    ndimage.binary_closing(
                        mask[index],
                        iterations=params['topology_closing_niter']
                    ).astype(bool)
                )

    def compute_box(self):
        """Compute the boundary of active cells."""
        # Find the box that (fully) encloses all selected cells, and the
        # side length of its surrounding cube
        ind_sel = np.where(self.mask)   # 3-tuple of ndarrays!
        self.mask_box = np.zeros((2, 3))
        for idim in range(3):
            min_index = np.min(ind_sel[idim])
            max_index = np.max(ind_sel[idim])
            self.mask_box[0, idim] = self.edges[min_index]
            self.mask_box[1, idim] = self.edges[max_index + 1]

        self.mask_widths = self.mask_box[1, :] - self.mask_box[0, :]
        self.mask_extent = np.max(self.mask_widths)

        self.box_volume = np.prod(self.mask_widths)

        print(
            f"Encompassing dimensions:\n"
            f"\tx = {self.mask_widths[0]:.4f} Mpc\n"
            f"\ty = {self.mask_widths[1]:.4f} Mpc\n"
            f"\tz = {self.mask_widths[2]:.4f} Mpc\n"
            f"Bounding length: {self.mask_extent:.4f} Mpc")

    def recenter(self):
        """Re-center the mask such that the selection is centred on origin."""
        mask_offset = (self.mask_box[1, :] + self.mask_box[0, :]) / 2
        self.mask_box[0, :] -= mask_offset
        self.mask_box[1, :] -= mask_offset
        self.mask_centre += mask_offset
        for idim in range(3):
            self.edges[idim] -= mask_offset

        print(f"Re-centred mask by {mask_offset} Mpc.") 
        return mask_offset


    def get_active_cell_centres(self):
        """Find the centre of all selected mask cells"""
        ind_sel = np.where(self.mask)   # Note: 3-tuple of ndarrays!
        self.cell_coords = np.vstack(
            (self.edges[ind_sel[0]],
             self.edges[ind_sel[1]],
             self.edges[ind_sel[2]]
            )
        ).T
        self.cell_coords += self.cell_size * 0.5

        n_sel = len(ind_sel[0])
        cell_fraction = n_sel * self.cell_size**3 / self.box_volume
        cell_fraction_cube = n_sel * self.cell_size**3 / self.mask_extent**3
        print(f'There are {n_sel:d} selected mask cells.')
        print(f'They fill {cell_fraction * 100:.3f} per cent of the bounding '
              f'box ({cell_fraction_cube * 100:.3f} per cent of bounding '
              f'cube).')

    def write(self, f, subgroup=None):
        """Write the mask data to an HDF5 file (handle f)."""

        if subgroup is not None:
            g = f.create_group(subgroup)
        else:
            g = f

        ds = g.create_dataset(
            'Coordinates', data=np.array(self.cell_coords, dtype='f8'))
        ds.attrs.create('Description',
                        "Coordinates of the centres of selected mask "
                        "cells [Mpc]. The (uniform) cell width is stored "
                        "as the attribute `grid_cell_width`.")

        # Store attributes directly related to the mask as HDF5 attributes.
        ds.attrs.create('mask_corners', self.mask_box)
        ds.attrs.create('bounding_length', self.mask_extent)
        ds.attrs.create('geo_centre', self.mask_centre)
        ds.attrs.create('grid_cell_width', self.cell_size)

        # Also store full regular mask structure
        ds = g.create_dataset(
            'FullMask', data=np.array(self.mask))
        ds.attrs.create(
            'Description',
            'Full boolean mask as 3D array. Cells that are True correspond '
            'to selected volume.')
        ds.attrs.create('x_edges', self.edges[0])
        ds.attrs.create('y_edges', self.edges[1])
        ds.attrs.create('z_edges', self.edges[2])
        ds.attrs.create('mask_corners', self.mask_box)
        ds.attrs.create('bounding_length', self.mask_extent)
        ds.attrs.create('geo_centre', self.mask_centre)
        ds.attrs.create('grid_cell_width', self.cell_size)


def periodic_wrapping(r, boxsize, return_copy=False):
    """
    Apply periodic wrapping to an input set of coordinates.

    Parameters
    ----------
    r : ndarray(float) [N, 3]
        The coordinates to wrap.
    boxsize : float
        The box size to wrap the coordinates to. The units must correspond to
        those used for `r`.
    return_copy : bool, optional
        Switch to return a (modified) copy of the input array, rather than
        modifying the input in place (which is the default).

    Returns
    -------
    r_wrapped : ndarray(float) [N, 3]
        The wrapped coordinates. Only returned if `return_copy` is True,
        otherwise the input array `r` is modified in-place.

    """
    if return_copy:
        r_wrapped = ((r + 0.5 * boxsize) % boxsize - 0.5 * boxsize)
        return r_wrapped

    # To perform the wrapping in-place, break it down into three steps
    r += 0.5 * boxsize
    r %= boxsize
    r -= 0.5 * boxsize


# Allow using the file as stand-alone script
if __name__ == '__main__':
    MakeMask(sys.argv[1])
