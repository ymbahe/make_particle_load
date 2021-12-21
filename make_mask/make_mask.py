"""Script to generate a mask for a single object."""

import sys
import os
import yaml
import h5py
import numpy as np
from typing import Tuple
from warnings import warn
from scipy import ndimage
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
            self.params['padding_snaps'] = None
            
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
            if self.params['select_from_vr']:
                self.params['centre'], self.params['radius'], vr_index = (
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
            The target radius of the high-res region, including any requested
            padding.
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

            r_highres = max(r_r200, r_r500)
            if r_highres <= 0:
                raise ValueError(
                    f"Invalid radius of high-res region ({r_highres})")

            # Store unpadded high-res readius
            self.params['primary_highres_radius'] = r_highres
            
            # If enabled, add a fixed "padding" radius to the high-res sphere
            if self.params['highres_radius_padding'] > 0:
                r_highres += self.params['highres_radius_padding']

            # If enabled, expand radius to requested minimum.
            if self.params['highres_radius_min'] > 0:
                r_highres = max(r_highres, self.params['highres_radius_min'])

            # Load halo centre
            names = ['X', 'Y', 'Z']
            centre = np.zeros(3)
            for icoord, prefix in enumerate(names):
                centre[icoord] = vr_file[f'{prefix}cminpot'][vr_index]

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
        from the Velociraptor halo catalogue.

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
        self.region = self.find_enclosing_frame()

        # Load IDs of particles within target high-res region from snapshot.
        # Note that only particles assigned to current MPI rank are loaded,
        # which may be none.
        primary_ids, mask_ids = self.load_primary_ids()

        # Identify mask particles. This includes all primary particles, but
        # also those that need to be included in the high-resolution region
        # to keep the primary particles well padded.
        if self.params['padding_snaps'] is not None:
            mask_ids = self.load_mask_ids(primary_ids)
        
        # Find initial positions from particle IDs (recall that these are
        # really Peano-Hilbert indices). Coordinates are in the same units
        # as the box size, centred (and wrapped) on the high-res target region.
        ic_coords = self.compute_ic_positions(ids)

        # Find the corners of a box enclosing all particles in the ICs.
        box, widths = self.compute_bounding_box(ic_coords)
        if comm_rank == 0:
            print(
                f"Determined bounding box edges in ICs (re-centred):\n"
                f"\t{box[0, 0]:.3f} / {box[0, 1]:.3f} / {box[0, 2]:.3f} --> "
                f"{box[1, 0]:.3f} / {box[1, 1]:.3f} / {box[1, 2]:.3f}")

        # For simplicity, shift the coordinates relative to geometric box
        # center, so that particles extend equally far in each direction
        geo_centre = box[0, :] + widths / 2
        ic_coords -= geo_centre
        self.ic_coords = ic_coords

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
        self.mask, edges = self.build_basic_mask(
            ic_coords, np.max(widths) * padding_factor)
        self.cell_size = edges[1] - edges[0]
        self.basic_mask = np.copy(self.mask)

        # We only need MPI rank 0 for the rest, since we are done working with
        # individual particles
        if comm_rank > 0:
            return

        # Fill holes and extrude the mask. This has to be done separately
        # for each of the three projections.
        for idim, name in enumerate(['x-y', 'y-z', 'x-z']):
            print(f"Topological extrision ({idim}/3, {name} plane)...")
            self.refine_mask(idim)

        # Finally, we need to find the centre of all selected mask cells, and
        # the box enclosing all those cells
        ind_sel = np.where(self.mask)   # Note: 3-tuple of ndarrays!
        self.cell_coords = np.vstack(
            (edges[ind_sel[0]], edges[ind_sel[1]], edges[ind_sel[2]])
            ).T
        self.cell_coords += self.cell_size * 0.5

        # Find the box that (fully) encloses all selected cells, and the
        # side length of its surrounding cube
        self.mask_box, self.mask_widths = self.compute_bounding_box(
            self.cell_coords, serial_only=True)
        self.mask_box[0, :] -= self.cell_size * 0.5
        self.mask_box[1, :] += self.cell_size * 0.5
        self.mask_widths += self.cell_size
        self.mask_extent = np.max(self.mask_widths)
        
        # Need to re-center such that the *mask* is centred on origin
        mask_offset = (self.mask_box[1, :] + self.mask_box[0, :]) / 2
        self.mask_box[0, :] -= mask_offset
        self.mask_box[1, :] -= mask_offset
        self.mask_centre += mask_offset
        self.cell_coords -= mask_offset
        self.ic_coords -= mask_offset
        print(f"Re-centred mask by {mask_offset} Mpc.") 
        
        print(
            f"Encompassing dimensions:\n"
            f"\tx = {self.mask_widths[0]:.4f} Mpc\n"
            f"\ty = {self.mask_widths[1]:.4f} Mpc\n"
            f"\tz = {self.mask_widths[2]:.4f} Mpc\n"
            f"Bounding length: {self.mask_extent:.4f} Mpc")

        box_volume = np.prod(self.mask_widths)
        n_sel = len(ind_sel[0])
        cell_fraction = n_sel * self.cell_size**3 / box_volume
        cell_fraction_cube = n_sel * self.cell_size**3 / self.mask_extent**3
        print(f'There are {n_sel:d} selected mask cells.')
        print(f'They fill {cell_fraction * 100:.3f} per cent of the bounding '
              f'box ({cell_fraction_cube * 100:.3f} per cent of bounding '
              f'cube).')

        # Final sanity check: make sure that all particles are within cubic
        # bounding box
        if (not np.all(box[0, :] - geo_centre >= -self.mask_extent / 2) or
            not np.all(box[1, :] - geo_centre <= self.mask_extent / 2)):
            raise ValueError(
                f"Cubic bounding box around final mask does not enclose all "
                f"input particles!\n"
                f"({box}\n   vs. {self.mask_extent:.4f})"
            )

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
        frame = np.zeros((2, 3))

        # If the target region is a sphere, find the enclosing cube
        if self.params['shape'] == 'sphere':
            frame[0, :] = centre - self.params['radius']
            frame[1, :] = centre + self.params['radius']

        # If the target region is a cuboid, simply transform from centre and
        # side length to lower and upper bounds along each coordinate
        elif self.params['shape'] in ['cuboid', 'slab']:
            frame[0, :] = centre - self.params['dim'] / 2.
            frame[1, :] = centre + self.params['dim'] / 2.

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
        Load IDs of particles in specified high-res region.

        In addition, relevant metadata are loaded and stored in the
        `self.params` dict.

        Returns
        -------
        ids : ndarray(int)
            The particle IDs of the primary particles.
        ids_all : ndarray(int)
            The IDs of all particles to be treated as high-res, including
            ones that are only included for padding.

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
                      f"{self.params['radius']:.4f} {l_unit}")

            dists = np.linalg.norm(coords, axis=1)
            ind_sel = np.where(dists <= self.params['radius'])[0]
            ind_primary = np.nonzero(
                dists <= self.params['primary_highres_radius'])[0]

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
            ind_sel = np.where(
                np.max(np.abs(coords / (self.params['dim'] / 2)), axis=1)
                <= 1)[0]
            ind_primary = np.copy(ind_sel)

        # We need the IDs of particles lying in the mask region
        ids = snap.read_dataset(1, 'ParticleIDs')

        # If IDs are Peano-Hilbert indices multiplied by two (as in e.g.
        # simulations with baryons), need to undo this multiplication here
        if self.params['divide_ids_by_two']:
            ids //= 2

        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles')

        return ids[ind_primary], ids[ind_sel]
    
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
                ph_ids = f['PartType1/ParticleIDs'][...]
                ids = ph_ids[ids]
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

    def compute_bounding_box(self, r, serial_only=False):
        """
        Find the corners of a box enclosing a set of points across MPI ranks.

        Parameters
        ----------
        r : ndarray(float) [N_part, 3]
            The coordinates of the `N_part` particles held on this MPI rank.
            The second array dimension holds the x/y/z components per point.
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

        # Find vertices of local particles (on this MPI rank). If there are
        # none, set lower (upper) vertices to very large (very negative)
        # numbers so that they will not influence the cross-MPI min/max.
        n_part = r.shape[0]
        box[0, :] = np.min(r, axis=0) - 0.01 if n_part > 0 else sys.float_info.max
        box[1, :] = np.max(r, axis=0) + 0.01 if n_part > 0 else -sys.float_info.max

        # Now compare min/max values across all MPI ranks
        if not serial_only:
            for idim in range(3):
                box[0, idim] = comm.allreduce(box[0, idim], op=MPI.MIN)
                box[1, idim] = comm.allreduce(box[1, idim], op=MPI.MAX)

        return box, box[1, :] - box[0, :]

    def build_basic_mask(self, r, max_width):
        """
        Build the basic mask for an input particle distribution.

        This is a cubic boolean array with an adaptively computed cell size and
        extent that stretches by at least `min_width` in each dimension.
        The mask value is True for any cells that contain at least the
        specified threshold number of particles.

        The mask is based on particles on all MPI ranks.

        Parameters
        ----------
        r : ndarray(float) [N_p, 3]
            The coordinates of (local) particles for which to create the mask.
            They must be shifted such that they lie within +/- `min_width` from
            the origin in each dimension.
        max_width : float
            The maximum extent of the mask along all six directions from the
            origin. It may get shrunk if `max_width` is larger than the whole
            box, but the mask will always remain centred on the origin. Note
            that this value must be identical across MPI ranks.

        Returns
        -------
        mask : ndarray(Bool) [N_cell, 3]
            The constructed boolean mask.
        edges : ndarray(N_cell + 1)
            The cell edges (same along each dimension)

        """
        # Find out how far from the origin we need to extend the mask
        width = min(max_width, self.params['box_size'])

        # Work out how many cells we need along each dimension so that the
        # cells remain below the specified threshold size
        num_bins = int(np.ceil(width / self.params['cell_size_mpc']))

        # Compute number of particles in each cell, across MPI ranks
        n_p, edges = np.histogramdd(
            r, bins=num_bins, range=[(-width, width)] * 3)
        n_p = comm.allreduce(n_p, op=MPI.SUM)

        # Convert particle counts to True/False mask
        mask = n_p >= self.params['min_num_per_cell']

        return mask, edges[0]   # edges is a 3-tuple

    def refine_mask(self, idim):
        """
        Refine the mask by checking for holes and processing the morphology.

        The refinement is performed iteratively along all slices along the
        specified axis. It consists of the ndimage operations ...

        The mask array (`self.mask`) is modified in place.

        Parameters
        ----------
        idim : int
            The perpendicular to which slices of the mask are to be processed
            (0: x, 1: y, 2: z)

        Returns
        -------
        None

        """
        # Process each layer (slice) of the mask in turn
        for layer_id in range(self.mask.shape[idim]):

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
            if self.params['topology_fill_holes']:
                self.mask[index] = (
                    ndimage.binary_fill_holes(self.mask[index]).astype(bool)
                )
            # Step 2: regularize the morphology
            if self.params['topology_dilation_niter'] > 0:
                self.mask[index] = (
                    ndimage.binary_dilation(
                        self.mask[index],
                        iterations=self.params['topology_dilation_niter']
                    ).astype(bool)
                )
            if self.params['topology_closing_niter'] > 0:
                self.mask[index] = (
                    ndimage.binary_closing(
                        self.mask[index],
                        iterations=self.params['topology_closing_niter']
                    ).astype(bool)
                )

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
                radius = self.params['radius']
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='white', linestyle='-', linewidth=2)
                ax.plot(np.cos(phi) * radius, np.sin(phi) * radius,
                        color='grey', linestyle='--', linewidth=1)
                if ii == 0:
                    ax.text(
                        0, self.params['radius'],
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

    def save(self):
        """
        Save the generated mask for further use.

        Note that, for consistency with IC GEN, all length dimensions must
        be in units of h^-1. This is already taken care of.

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

            # Main output is the centres of selected mask cells
            ds = f.create_dataset(
                'Coordinates', data=np.array(self.cell_coords, dtype='f8')
            )
            ds.attrs.create('Description',
                            "Coordinates of the centres of selected mask "
                            "cells [Mpc]. The (uniform) cell width is stored "
                            "as the attribute `grid_cell_width`.")

            db = f.create_dataset(
                'BasicMask', data=np.array(self.basic_mask))
            db.attrs.create('Description', "Unrefined boolean mask.")

            # Store attributes directly related to the mask as HDF5 attributes.
            ds.attrs.create('mask_corners', self.mask_box)
            ds.attrs.create('bounding_length', self.mask_extent)
            ds.attrs.create('box_size', self.params['box_size'])
            ds.attrs.create('geo_centre', self.mask_centre)
            ds.attrs.create('grid_cell_width', self.cell_size)
            if self.params['shape'] in ['cuboid', 'slab']:
                high_res_volume = np.prod(self.params['dim'])
            else:
                high_res_volume = 4 / 3. * np.pi * self.params['radius']**3.
            ds.attrs.create('high_res_volume', high_res_volume)

        print(f"Saved mask data to file `{outloc}`.")


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
    x = MakeMask(sys.argv[1])
