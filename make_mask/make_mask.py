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
import argparse

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set up Matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Palatino'

except ModuleNotFoundError:
    print('matplotlib not found...')
    mpl = None
    
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from pdb import set_trace

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print('mpi4py not found... This may not work yet.')
    MPI = None
    
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
import cosmology as co
import utils
from timestamp import TimeStamp

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
if MPI is not None:
    comm = MPI.COMM_WORLD
    comm_rank = comm.rank
    comm_size = comm.size
else:
    comm = None
    comm_rank = 0
    comm_size = 1
    

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

    def __init__(self, args, param_file=None, params=None, save=True,
                 plot=True):

        # Parse the parameter file, check for consistency, and determine
        # the centre and radius of high-res sphere around a VR halo if desired.
        if param_file is None:
            param_file = args.param_file
        self.read_param_file(param_file, params, override_params=args.params)
        
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

    def read_param_file(self, param_file, params, override_params=None):
        """
        Read parameters from a specified YAML file.

        See template file `param_files/template.yml` for a full listing and
        description of parameters. The values are stored in the internal
        `self.params` dict.

        If `params` is a dict, this is assumed to represent the parameter
        structure instead, and `param_file` is ignored. Parameters are still
        checked and processed in the same way. Otherwise, `params` is ignored.

        The optional `override_params` can provide a dict of parameter name/
        value pairs that will override entries in `params` or the parameter
        file (or provide them, if they don't exist yet).

        If the parameter file specifies the target centre in terms of a
        particular Velociraptor halo, the centre and radius of the high-res
        region are determined internally.

        Note:
        -----
        In the MPI version, the file is only read on one rank and the dict
        then broadcast to all other ranks.

        """
        if comm_rank == 0:

            if not isinstance(params, dict):
                params = yaml.safe_load(open(param_file))

            if isinstance(override_params, dict):
                for key in override_params:
                    params[key] = override_params[key]
                
            # Set default values for optional parameters
            self.params = {}
            self.params['min_num_per_cell'] = 3
            self.params['cell_size_mpc'] = 3.
            self.params['topology_fill_holes'] = True
            self.params['topology_dilation_niter'] = 0
            self.params['topology_closing_niter'] = 0
            self.params['orig_id_name'] = 'PeanoHilbertIDs'
            self.params['padding_snaps'] = None
            self.params['highres_padding_width'] = 0
            self.params['highres_diffusion_buffer'] = 50.0
            self.params['cell_padding_width'] = 0.0
            self.params['mask_pad_in_mips'] = 3.0
            self.params['dither_gap'] = None
            self.params['pad_lcs_as_particles'] = False
            self.params['direct_primary_load'] = 0
            
            # Convert "None"/"True"/"False" strings where applicable:
            utils.set_none(params, 'padding_snaps')
            utils.set_none(params, 'dither_gap')
            utils.set_none(params, 'pad_lcs_as_particles')

            # Define a list of parameters that must be provided. An error
            # is raised if they are not found in the YAML file.
            required_params = [
                'fname',
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

            if 'snapshot_file' not in params:
                params['snapshot_file'] = params['snapshot_base'].replace(
                    '$isnap', f"{params['primary_snapshot']:04d}")
            if 'vr_file' in params and 'primary_snapshot' in params:
                params['vr_file'] = params['vr_file'].replace(
                    '$isnap', f"{params['primary_snapshot']:04d}")

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
            elif self.params['shape'] == 'sphere':
                self.params['hr_radius'] = self.params['radius']

            # Replace VR halo placeholders (if-checks are there to avoid
            # an error when manual target region selection is used)
            if '$vr' in self.params['fname']:
                self.params['fname'] = (
                    self.params['fname'].replace('$vr', f'{vr_index}'))
            if '$vr' in self.params['output_dir']:
                self.params['output_dir'] = (
                    self.params['output_dir'].replace('$vr', f'{vr_index}'))

            # Parse padding options, if provided
            if self.params['padding_snaps'] is not None:
                if isinstance(self.params['padding_snaps'], int):
                    self.params['padding_snaps'] = (
                        f"{self.params['padding_snaps']}")
                pad_snaps = np.array(
                    self.params['padding_snaps'].split(), dtype=int)
                if len(pad_snaps) == 1:
                    self.params['padding_snaps'] = np.array([pad_snaps[0],])
                else:
                    pad_start = pad_snaps[0]
                    pad_end = pad_snaps[1]
                    pad_space = 1 if len(pad_snaps) == 2 else pad_snaps[2]
                    self.params['padding_snaps'] = np.arange(
                        pad_start, pad_end+1, pad_space)
            else:
                self.params['padding_snaps'] = np.zeros(0, dtype=int)
                    
            # Check that we can compute a cell size
            if self.params['cell_size_mpc'] == 'auto':
                if 'cell_size_mips' not in self.params:
                    raise AttributeError(
                        "Must specify cell size to MIPS ratio for AUTO!")
                    
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
        if comm is not None:
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

        print(f"Reading data from [{self.params['vr_file']}]")
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

        r_highres = max(r_r200, r_r500) + self.params['highres_radius_buffer']
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

        m_part = self.base_particle_mass()
        self.mips = compute_mips(m_part, self.params['base_cosmology'])
        if self.params['cell_size_mpc'] == 'auto':
            self.params['cell_size_mpc'] = (
                self.mips * self.params['cell_size_mips'])
        if self.params['dither_gap'] is not None:
            self.params['dither_gap'] *= self.mips

        # Load IDs of all possibly relevant particles, and the (sub-)indices
        # of those lying in the target region and within the surrounding
        # padding zone. Only particles assigned to the current MPI rank are "
        # loaded, which may be none.
        if self.params['direct_primary_load'] == 1:
            ids, inds_target, inds_pad = self.load_primary_ids_direct()
        else:
            ids, inds_target, inds_pad = self.load_primary_ids()

        self.inds_target = inds_target
        
        # Find initial (Lagrangian) positions of particles from their IDs
        # (recall that these are really Peano-Hilbert indices).
        # Coordinates are in the same units as the box size.
        self.lagrangian_coords = self.compute_lagrangian_coordinates(ids)
        self.ids = ids
        
        # If desired, identify additional padding particles in other snapshots.
        # This is currently only possible in non-MPI runs.
        if (len(self.params['padding_snaps']) > 0
            and not self.params['pad_full_cells']):
            inds_pad = self.find_extra_pad_particles(
                ids, inds_target, inds_pad)

        if len(inds_pad) > 0:
            r_max = self.primary_radii[inds_pad].max()
            print(f"Maximum distance of pad particle from selection centre "
                  f"in primary snapshot is {r_max:.3f} Mpc.")
        else:
            print("No padding particles identified!")
            

        # Find the corners of a box enclosing all particles in the ICs that
        # are so far identified as "to be treated as high-res". The coordinates
        # are relative to the centre of the box, which is found internally
        # (accounting for periodic wrapping); particle coordinates are also
        # shifted to this frame.
        inds_all = np.concatenate((inds_target, inds_pad))
        box, origin = self.compute_bounding_box(inds_all)
        box[0, :] -= self.mips * self.params['mask_pad_in_mips']
        box[1, :] += self.mips * self.params['mask_pad_in_mips']
        if (np.min(box) < -self.params['box_size'] / 2 or
            np.max(box) > self.params['box_size'] / 2):
            raise ValueError(
                "Target and (preliminary) padding particles extend beyond "
                "half a simulation box size from the centre: "
                f"{np.min(box):.3f} / {np.max(box):.3f} vs. "
                f"{(self.params['box_size'] / 2):.3f} Mpc."
            )
        self.lagrangian_coords_origin = origin
        
        # Build the basic masks. These are 3D Boolean arrays covering the
        # bounding box computed above.
        # Cells containing at least the specified threshold number of
        # (target / target + pad) particles are True, others are False.
        self.cell_size = self.params['cell_size_mpc']
        print(f"Mask cell size is {self.cell_size:.3f} Mpc.")
        self.target_mask = Mask(
            self.lagrangian_coords[inds_target], box, self.params)
        self.full_mask = Mask(
            self.lagrangian_coords[inds_all], box, self.params)

        # Record the origin (centre) of the mask in the full simulation box
        self.target_mask.set_origin(origin)
        self.full_mask.set_origin(origin)
        self.target_mask.set_simulation_box_size(self.params['box_size'])
        self.full_mask.set_simulation_box_size(self.params['box_size'])

        # Now apply "stage-2 padding" (for entire target cells)
        if self.params['pad_full_cells']:
            full_target_inds = self.target_mask.find_particles_in_active_cells(
                self.lagrangian_coords)
            inds_pad = self.find_extra_pad_particles(
                ids, full_target_inds, inds_pad, with_primary_snapshot=True)
            self.inds_target_xt = full_target_inds
        else:
            self.inds_target_xt = self.inds_target
        self.inds_pad = inds_pad

        print(f"There are {len(self.inds_target_xt)} target particles, and "
              f"{len(self.inds_pad)} padding particles.")
        
        # Expand the full mask with updated padding particles. If applicable,
        # also add entire high-res padding cells around those hosting target
        # particles. The mask is enlarged appropriately to allow refinement.
        self.full_mask.expand(
            self.lagrangian_coords[inds_pad, :],
            target_mask=self.target_mask,
            cell_padding_width=self.params['cell_padding_width'],
            refinement_allowance=(0.2, 2))

        # We only need MPI rank 0 for the rest, since we are done working with
        # individual particles
        if comm_rank > 0:
            return

        # Fill holes and extrude the mask. This has to be done separately
        # for each of the three projections.
        for idim, name in enumerate(['x-y', 'y-z', 'x-z']):
            print(f"Topological extrision ({idim}/3, {name} plane)...")
            self.full_mask.refine(idim, self.params)

        # Re-center the full mask to account for possible shifts
        # (this does not involve any particles)
        self.target_mask.compute_active_box()
        self.full_mask.compute_active_box()
        self.full_mask.recenter()

        # Compute the centres of active cells (key output)
        self.target_mask.get_active_cell_centres()
        self.full_mask.get_active_cell_centres()

    # ------------------------------------------------------------------------
    # ------------ Sub-functions of make_mask() ------------------------------
    # ------------------------------------------------------------------------

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

    def base_particle_mass(self):
        """Load the particle mass of the base simulation."""
        if comm_rank == 0:
            with h5py.File(self.params['snapshot_file'], 'r') as f:
                m_part = f['PartType1/Masses'][0] * 1e10
            print(f'Base simulation particle mass: {m_part:.3e} M_Sun.')
        else:
            m_part = None

        if comm is None:
            self.m_part_base = m_part
        else:
            self.m_part_base = comm.bcast(m_part)
        return self.m_part_base

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
            snap = read_swift(self.params['snapshot_file'], comm=comm)
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

        # Shift coordinates relative to target centre, and apply periodic
        # wrapping if required
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
        self.primary_radii = np.linalg.norm(coords, axis=1)
        
        return ids, ind_target, ind_pad
    
    def load_primary_ids_direct(self):
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

        with h5py.File(self.params['snapshot_file'], 'r') as f:
            coords = f['PartType1/Coordinates'][...] - cen
            ids = f['PartType1/ParticleIDs'][...]
            self.params['box_size'] = f['Header'].attrs['BoxSize'][0]
            self.params['length_unit'] = 'Mpc'
            self.zred_snap = f['Header'].attrs['Redshift'][0]
            
        # Periodic wrapping if required
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

        # If IDs are Peano-Hilbert indices multiplied by two (as in e.g.
        # simulations with baryons), need to undo this multiplication here
        if self.params['divide_ids_by_two']:
            ids //= 2

        print(f'[Rank {comm_rank}] Loaded {len(ids)} dark matter particles')

        # Make a plot of the selection environment
        self.plot_halo(coords)
        self.primary_radii = np.linalg.norm(coords, axis=1)
        
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

    def compute_lagrangian_coordinates(self, ids) -> np.ndarray:
        """Compute the Lagrangian particle coordinates from their IDs.

        The coordinates can be encoded either as Peano-Hilbert index,
        or in simple sequential form as used in monofonIC. The behaviour
        depends on the value of `self.params['ids_type']`. It is also possible
        to use IDs that are indices into a position-encoding number.

        Parameters
        ----------
        ids : ndarray(int)
            The Particle IDs for which to calculate IC coordinates. If they
            are not coordinate-encoding themselves, they are translated
            via the specified ICs file `self.params['ics_file']`.

        Returns
        -------
        coords : ndarray(float)
            The coordinates of particles in the ICs [Mpc].

        """
        print(f"[Rank {comm_rank}] Computing initial positions of dark matter "
              "particles...")

        # First resolve possible non-encoding IDs
        if not self.params['ids_encode_position']:
            print("Translating particle IDs to position-encoding ones...",
                  end='', flush=True)
            with h5py.File(self.params['ics_file'], 'r') as f:
                ics_ph_ids = f[f"PartType1/{self.params['orig_id_name']}"][...]
                ids = ics_ph_ids[ids-1]

            print(" done.")

        # Now translate the IDs to positions. This differs between
        # Peano-Hilbert and lattice encoding forms.
        if self.params['ids_type'].lower() == 'peano':
            return self._compute_lagrangian_coordinates_peano(ids)
        elif self.params['ids_type'].lower() == 'lattice':
            return self._compute_lagrangian_coordinates_lattice(ids)
        else:
            raise ValueError(f'Invalid ID type "{self.params[ids_type]}"!')

    def _compute_lagrangian_coordinates_peano(self, ids) -> np.ndarray:
        """Translate Peano-Hilbert IDs into Lagrangian coordinates."""

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

        return ic_coords.astype('f8')

    def _compute_lagrangian_coordinates_lattice(self, ids) -> np.ndarray:
        """Translate Lattice IDs into Lagrangian coordinates."""

        ngrid = self.params['parent_npart']
        lbox = self.params['box_size']
        z = ids % ngrid
        y = ((ids - z) / ngrid) % ngrid
        x = ((ids - z) / ngrid - y) / ngrid

        # Re-scale coordinates from [0, ngrid] --> [0, lbox]
        coords = np.vstack((x, y, z)).T
        coords *= lbox / ngrid

        return coords.astype('f8')

    def compute_bounding_box(
        self, inds, serial_only=False, with_wrapping=True):
        """
        Find the corners of a box enclosing a set of points across MPI ranks.

        Parameters
        ----------
        inds : ndarray(int) or list thereof
            The particles (indices into self.lagrangian_coords) for which
            the bounding box should be found.
        serial_only : bool, optional
            Switch to disable cross-MPI comparison of box extent (default:
            False). If True, the return values will generally differ between
            different MPI ranks.

        Returns
        -------
        box : ndarray(float) [2, 3]
            The coordinates of the lower and upper vertices of the bounding
            box. These are stored in index 0 and 1 along the first dimension,
            respectively. The coordinate frame is shifted w.r.t. that of
            the input coordinates such that the origin lies in the box centre.
            Periodic wrapping is applied where necessary, i.e. the values are
            guaranteed to be <= L/2 (L = parent simulation box size).
        origin : ndarray(float) [3]
            The geometric centre of the box, wrapped to 0 --> L.

        Note
        ----
        Cases in which the points are split across one or more periodic box
        edges of the parent simulation are handled, but this does not always
        work if the points extend by more than L/2 in at least one dimension.
        An error is thrown if this is the case.

        """
        origin = np.zeros(3)

        if isinstance(inds, list):
            inds = np.concatenate(inds)

        # Find vertices of particles (cross-MPI)
        box = find_vertices(
            self.lagrangian_coords[inds, :], serial_only=serial_only)

        # If we run with wrapping check, see whether we are close to the edge
        # in one or more dimensions:
        if with_wrapping:
            redo = False
            l_box = self.params['box_size']
            for idim in range(3):
                if (box[0, idim] < l_box*0.05 and box[1, idim] > l_box*0.95):
                    redo = True
                    origin[idim] += l_box * 0.5
                    self.lagrangian_coords[:, idim] -= (l_box * 0.5)

            if redo:
                # Wrap (potentially shifted) coordinates back to range 0 --> L
                periodic_wrapping(self.lagrangian_coords, l_box, mode='corner')

                # Find box again (in shifted frame) and see whether it's better
                box = find_vertices(
                    self.lagrangian_coords[inds, :], serial_only=serial_only)

                for idim in range(3):
                    if (box[0, idim] < l_box * 0.05 and
                        box[1, idim] > l_box * 0.95):
                        raise ValueError(
                            f"Lagrangian coordinates are close to both box "
                            f"edges in dimension {idim}, both before and "
                            f"after shifting. This probably means that they "
                            f"cover >50% of the box size -- I am giving up."
                        )

        # If we get here, we either don't care about wrapping issues or the
        # particles are all within the same image of the box. For consistency
        # and simplicity, shift all coordinates to the box centre:
        shift = (box[0, :] + box[1, :]) / 2
        origin += shift
        box -= shift
        self.lagrangian_coords -= shift

        # Although all *selected* particles are guaranteed to be in the same
        # image of the simulation box, there may be others that are still
        # wrapped around an edge
        if with_wrapping:
            periodic_wrapping(self.lagrangian_coords, l_box)

        if comm_rank == 0:
            print(
                f"Lagrangian bounding box centred on {origin[0]:.3f} / "
                f"{origin[1]:.3f} / {origin[2]:.3f}, with edges\n"
                f"\t{box[0, 0]:.3f} / {box[0, 1]:.3f} / {box[0, 2]:.3f} --> "
                f"{box[1, 0]:.3f} / {box[1, 1]:.3f} / {box[1, 2]:.3f}")

        return box, origin

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
        n_pad_initial = len(inds_pad)
        
        snaps = self.params['padding_snaps']
        if with_primary_snapshot:
            snaps = np.concatenate(([self.params['primary_snapshot']], snaps))
        print("About to search for extra padding particles in snapshots ",
              snaps)
        if self.params['pad_lcs_as_particles']:
            snaps = np.concatenate(([-1], snaps))

        # Set up a mask to record which particles are tagged as padding
        is_tagged = np.zeros(len(ids), dtype=bool)
        is_tagged[inds_pad] = True
        
        # As an optimization, we search for neighbours in bunches of target
        # particles. This avoids potentially severe overheads in constructing
        # the neighbour list-of-lists below.
        bounds = np.arange(0, len(inds_target)+1, 10000)
        if bounds[-1] != len(inds_target):
            bounds = np.concatenate((bounds, [len(inds_target)]))
        nbatch = len(bounds) - 1
        
        for snap in snaps:
            tx = TimeStamp()
            print(f"   ... snapshot {snap}...")

            if snap >= 0:
                snapshot_base = self.params['snapshot_base']
                snapshot_file = snapshot_base.replace('$isnap', f'{snap:04d}')
                print(snapshot_file)
                with h5py.File(snapshot_file, 'r') as f:
                    snap_ids = f['PartType1/ParticleIDs'][...]
                    snap_pos = f['PartType1/Coordinates'][...]
                tx.set_time('Load snapshot data')
                print(f"    ... loaded data ({tx.get_time():.2f} sec.) ...")
            else:
                snap_pos = self.lagrangian_coords
                snap_ids = self.ids
            
            # Need to locate all particles we are considering in this snap
            inds, in_snap = xr.find_id_indices(ids, snap_ids)
            if len(in_snap) < len(inds):
                raise ValueError(
                    f"Could only locate {len(in_snap)} out of "
                    f"{len(inds)} particles in snapshot {snap}!"
                )
            r = snap_pos[inds, :]
            r_target = r[inds_target, :]
            tx.set_time('Match particles')
            print(f"    ... matched IDs ({tx.get_time():.2f} sec.) ...")

            # At this point, `r` holds the coordinates of considered particles
            # in the current snapshot, sorted in the same way as in the
            # primary snapshot.
            print(f"Finding current neighbours for {len(inds_target)} "
                  "target particles.")

            n_free = np.count_nonzero(is_tagged == False)
            for ibatch in range(nbatch):
                tss = TimeStamp()
                print(f"Batch {ibatch+1} / {nbatch}...")

                # For efficiency reasons, we periodically rebuild the
                # neighbour tree, keeping only not-yet-tagged particles.
                # There is definite room for more optimization here...
                if ibatch % 10 == 0:

                    # Check whether we should re-build
                    n_free_now = np.count_nonzero(is_tagged == False)
                    if (ibatch == 0 or
                        n_free - n_free_now > 10000 or
                        n_free_now / n_free < 0.9):
                        print(f"   ... build new tree ...")
                        ind_ngb = np.nonzero(is_tagged == False)[0]    
                        ngb_tree = cKDTree(r[ind_ngb, :],
                                       boxsize=self.params['box_size'])
                        tss.set_time('ngb tree')
                        n_free = n_free_now

                # Build the target tree for (target) particles in batch
                target_tree = cKDTree(
                    r_target[bounds[ibatch] : bounds[ibatch+1], :],
                    boxsize=self.params['box_size'])
                tss.set_time('target tree')

                # Query the tree to find neighbours of targets in the full
                # neighbour tree
                ngbs_lol = target_tree.query_ball_tree(
                    ngb_tree, r=self.params['highres_padding_width'])
                tss.set_time('query')
                for ngbs in ngbs_lol:
                    is_tagged[ind_ngb[ngbs]] = True

                tss.set_time('transcribe')
                tx.import_times(tss)

            tx.set_time('Batches')
            n_pad_old = len(inds_pad)
            inds_pad = np.nonzero(is_tagged)[0]
            tx.set_time('Update tagged particles')
            tx.print_time_usage('Finished ngb search', mode='sub')
            tx.print_time_usage(f'Finished snap {snap}')
            print(f"Found {len(inds_pad)} neighbours so far "
                  f"[snap {snap}, up from {n_pad_old}]...")
            
        n_pad_final = len(inds_pad)
        print(f"Increased padding from {n_pad_initial} to {n_pad_final} "
              "particles.")
        return inds_pad

    def plot(self, max_npart_per_rank=int(1e5)):
        """
        Make an overview plot of the zoom-in region.

        Note that this function must be called on all MPI ranks, even though
        only rank 0 generates the actual plot. The others are still required
        to access (a subset of) the particles stored on them.

        """
        if mpl is None:
            return
        if comm_rank == 0:
            print("Plotting Lagrangian region...")
        axis_labels = ['x', 'y', 'z']
        mask = self.full_mask
        target_mask = self.target_mask
        
        plot_inds = self.inds_target_xt
        pad_inds = self.inds_pad
        
        # Select a random sub-sample of particle coordinates on each rank and
        # combine them all on rank 0
        np_ic = len(plot_inds)
        n_sample = int(min(np_ic, max_npart_per_rank))
        indices = np.random.choice(np_ic, n_sample, replace=False)
        plot_coords = self.lagrangian_coords[plot_inds[indices], :]

        np_pad = len(pad_inds)
        n_sample_pad = int(min(np_pad, max_npart_per_rank))
        indices_pad = np.random.choice(np_pad, n_sample_pad, replace=False)
        pad_coords = self.lagrangian_coords[pad_inds[indices_pad], :]

        if comm is not None:
            plot_coords = comm.gather(plot_coords)
            pad_coords = comm.gather(pad_coords)

        # Only need rank 0 from here on, combine all particles there.
        if comm_rank != 0:
            return
        plot_coords = np.vstack(plot_coords)
        pad_coords = np.vstack(pad_coords)

        origin_shift = self.lagrangian_coords_origin - mask.origin
        plot_coords += origin_shift
        pad_coords += origin_shift

        origin_shift_target = target_mask.origin - mask.origin
        
        # Extract frequently needed attributes for easier structure
        bound = mask.mask_extent
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
            box_corners = [mask.mask_box[:, xx], mask.mask_box[:, yy]]
            ax.plot(
                box_corners[0][[0, 1, 1, 0, 0]],
                box_corners[1][[0, 0, 1, 1, 0]],
                color='maroon', linestyle='--', linewidth=0.7
            )

            # Plot particles.
            ax.scatter(
                plot_coords[:, xx], plot_coords[:, yy],
                s=0.5, c='blue', zorder=-100, alpha=0.3)
            ax.scatter(
                pad_coords[:, xx], pad_coords[:, yy],
                s=0.25, c='grey', zorder=-200, alpha=0.3)

            ax.set_xlim(-bound/2. * 1.05, bound/2. * 1.05)
            ax.set_ylim(-bound/2. * 1.05, bound/2. * 1.05)

            # Plot (the centres of) selected mask cells.
            ax.scatter(
                mask.cell_coords[:, xx], mask.cell_coords[:, yy],
                marker='x', color='red', s=5, alpha=0.2)
            ax.scatter(
                target_mask.cell_coords[:, xx] + origin_shift_target[xx],
                target_mask.cell_coords[:, yy] + origin_shift_target[yy],
                marker='.', color='green', s=2, alpha=0.2)

            # Plot cell outlines if there are not too many of them.
            if mask.cell_coords.shape[0] < 10000:
                for e_x, e_y in zip(
                    mask.cell_coords[:, xx], mask.cell_coords[:, yy]):
                    rect = patches.Rectangle(
                        (e_x - cell_size/2, e_y - cell_size/2),
                        cell_size, cell_size,
                        linewidth=0.5, edgecolor='r', facecolor='none',
                        alpha=0.2
                    )
                    ax.add_patch(rect)

            if target_mask.cell_coords.shape[0] < 10000:
                for e_x, e_y in zip(
                    target_mask.cell_coords[:, xx],
                    target_mask.cell_coords[:, yy]):
                    rect = patches.Rectangle(
                        (e_x - cell_size/2 + origin_shift_target[xx],
                         e_y - cell_size/2 + origin_shift_target[yy]),
                        cell_size, cell_size,
                        linewidth=1.0, edgecolor='green', facecolor='none',
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
        print("...done!")

    def plot_halo(self, pos):
        """
        Make an overview plot of the selected region.

        Note that this function must be called on all MPI ranks, even though
        only rank 0 generates the actual plot. The others are still required
        to access (a subset of) the particles stored on them.

        """
        if mpl is None:
            return
        axis_labels = ['x', 'y', 'z']

        # Extract frequently needed attributes for easier structure
        frame = self.region - self.params['centre']
        frame[0, :] += self.params['highres_diffusion_buffer']
        frame[1, :] -= self.params['highres_diffusion_buffer']
        bound = np.max(np.abs(frame))
        try:
            r200 = self.params['r200']
        except KeyError:
            r200 = None
            
        ind_sel = np.nonzero(np.max(np.abs(pos), axis=1) <= bound)[0]
        hists = np.zeros((3, 200, 200))
        for ii, (xx, yy) in enumerate(zip([0, 0, 1], [1, 2, 2])):
            hists[ii, ...], xedges, yedges = np.histogram2d(
                pos[ind_sel, yy], pos[ind_sel, xx], bins=200,
                range=[[-bound, bound], [-bound, bound]]
            )

        if comm is None:
            hist_full = np.copy(hists)
        else:
            hist_full = np.zeros((3, 200, 200)) if comm_rank == 0 else None
            comm.Reduce([hists, MPI.DOUBLE],
                        [hist_full, MPI.DOUBLE],
                        op=MPI.SUM, root=0)

        # Only need rank 0 from here on, combine all particles there.
        if comm_rank != 0:
            return

        fig, axarr = plt.subplots(1, 3, figsize=(13, 4))

        if (hist_fill > 0).any():
            ind_filled = np.nonzero(hist_full > 0)
            vmin, vmax = np.percentile(hist_full[ind_filled], [0.1, 99.99])
        else:
            vmin, vmax = 0, 1

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
                try:
                    g.attrs.create(param_attr, self.params[param_attr])
                except TypeError:
                    att_value = self.params[param_attr]
                    if att_value is None:
                        g.attrs.create(param_attr, "None")
                    else:
                        g.attrs.create(param_attr, "[Object]")

            if self.params['shape'] in ['cuboid', 'slab']:
                high_res_volume = np.prod(self.params['dim'])
            else:
                high_res_volume = 4 / 3. * np.pi * self.params['hr_radius']**3
            g.attrs.create('high_res_volume', high_res_volume)

            # Main output is the centres of selected mask cells
            self.full_mask.write(f)
            self.target_mask.write(f, subgroup="TargetMask")

        print(f"Saved mask data to file `{outloc}`.")


class Mask:
    """Low-level class to represent one individual mask."""

    def __init__(self, r, box, params):

        self.params = params
        
        # Work out how many cells we need along each dimension so that the
        # cells remain below the specified threshold size
        box_widths = box[1, :] - box[0, :]
        self.cell_size = params['cell_size_mpc']
        num_cells = np.ceil(box_widths / self.cell_size).astype(int)

        # To keep the cells cubic at the specified side length, we need to
        # extend the mask region slightly beyond the box
        bin_edges = []
        for idim in range(3):
            extent_dim = num_cells[idim] / 2 * self.cell_size
            bin_edges.append(
                np.linspace(-extent_dim, extent_dim, num=num_cells[idim]+1))

        self.build_mask_from_coordinates(
            r, bin_edges, n_threshold=params['min_num_per_cell'], store=True,
            dither=params['dither_gap'])
        self.edges = bin_edges
        
        self.box = np.zeros((2, 3))
        for idim in range(3):
            self.box[0, idim] = self.edges[idim][0]
            self.box[1, idim] = self.edges[idim][-1]

    def build_mask_from_coordinates(
        self, r, edges, n_threshold, store=True, dither=False):
        """
        Build a boolean mask from a set of input coordinates.

        Parameters
        ----------
        r : ndarray(float)
            The coordinates from which to build the mask.
        edges : list of ndarray(float)
            A list of n arrays specifying the bin edges in the i-th dimension.
        n_threshold : int
            The minimum number of particles for a cell to be classed "active".
        store : bool, optional
            Store the computed mask as attribute `self.mask`? Default: True.
        dither : float or None, optional
            Offset for dithering to apply before counting particles per cell.
            Each particle is offset by [-1, 0, +1] * dither in each dimension,
            for a total of 27 dither positions. If None (default), no
            dithering is applied and only a single count is taken.

        Returns
        -------
        mask : ndarray(bool)
            An array with one element per cell that is True for cells with
            particles and False for others. Its dimensions depend on the
            input array `edges`.

        """
        if dither is None:
            dr = np.array([0.])
        else:
            dr = np.array([-dither, 0., dither])

        mask = None
        for dx in dr:
            for dy in dr:
                for dz in dr:
                    r_curr = np.copy(r)
                    r_curr[:, 0] += dx
                    r_curr[:, 1] += dy
                    r_curr[:, 2] += dz
                    
                    # Compute the number of particles in each cell,
                    # across MPI ranks
                    n_p, hist_edges = np.histogramdd(r_curr, bins=edges)
                    for idim in range(len(edges)):
                        if len(hist_edges[idim]) != len(edges[idim]):
                            raise ValueError(
                                "Inconsistent histogram edge length.")
                        if np.count_nonzero(hist_edges[idim] != edges[idim]):
                            raise ValueError(
                                "Inconsistent histogram edge values.")
                    if comm is not None:
                        n_p = comm.allreduce(n_p, op=MPI.SUM)

                    # Convert particle counts to True/False mask
                    mask_curr = (n_p >= n_threshold)
                    if mask is None:
                        mask = mask_curr
                    else:
                        mask += mask_curr   # [False/True] + True = True

        shape = np.array(mask.shape)

        if store:
            self.mask = mask
            self.shape = shape

        if comm_rank == 0:
            n_med = np.median(n_p[mask])
            print(f"Median number of particles per active mask cell: {n_med}")

        return mask

    def set_origin(self, origin):
        self.origin = np.copy(origin)

    def set_simulation_box_size(self, l_box):
        self.l_box = l_box

    def find_particles_in_active_cells(self, r):
        """Find all particles that lie within target mask cells.

        Parameters
        ----------
        r : ndarray(float)
            The particles to test against the mask.

        Returns
        -------
        indices : ndarray(int)
            The indices of those particles that occupy active cells.

        """
        cell_indices = self.coordinates_to_cell(r)

        # First cut: particles that are in valid mask region
        ind_in_mask = np.nonzero(
            (np.min(cell_indices, axis=1) >= 0) &
            (cell_indices[:, 0] < self.shape[0]) &
            (cell_indices[:, 1] < self.shape[1]) &
            (cell_indices[:, 2] < self.shape[2])
        )[0]

        # Second cut: particles that are in activated mask cells
        subind_target = np.nonzero(
            self.mask[
                cell_indices[ind_in_mask, 0],
                cell_indices[ind_in_mask, 1],
                cell_indices[ind_in_mask, 2]
            ]
        )[0]

        print(f"Out of {r.shape[0]} particles, "
              f"{len(subind_target)} lie in an active mask cell.")

        return ind_in_mask[subind_target]

    def coordinates_to_cell(self, r):
        """Find the cells for a given set of coordinates."""
        cells = ((r - self.box[0, :]) // self.cell_size).astype(int)
        return cells

    def expand(self, r, target_mask=None, cell_padding_width=0,
               refinement_allowance=None):
        """
        Expand the mask to accommodate additional particles and cells.

        In addition to the specified particles, two optional expansions can
        be performed: (i) cells within a given distance from a target

        """
        # Determine (integer) number of cells below and above current mask
        # to accommodate extra padding particles
        if len(r) > 0:
            cell_indices = self.coordinates_to_cell(r)
            extra_low = np.abs(np.min(cell_indices, axis=0))
            extra_high = np.max(cell_indices, axis=0) - self.shape + 1
        else:
            extra_low = np.zeros(3, dtype=int)
            extra_high = np.zeros(3, dtype=int)

        # (Symmetric) extra number of cells for cell-level padding
        if cell_padding_width > 0:
            if target_mask is None:
                raise ValueError("Must provide target mask!")
            cell_pad_extra = int(np.ceil(
                cell_padding_width / self.cell_size + np.sqrt(3) - 1))
        else:
            cell_pad_extra = np.zeros(3, dtype=int)

        # (Symmetric) extra number of cells for later refinement
        if refinement_allowance is not None:
            ref_extra = np.ceil(target_mask.shape * refinement_allowance[0])
            ref_extra_min = refinement_allowance[1]
            ref_extra = np.maximum(ref_extra, ref_extra_min).astype(int)
        else:
            ref_extra = np.zeros(3, dtype=int)

        extra_low = np.maximum(extra_low, cell_pad_extra) + ref_extra
        extra_high = np.maximum(extra_high, cell_pad_extra) + ref_extra
        new_shape = self.shape + extra_low + extra_high

        new_edges = []
        for idim in range(3):
            new_min = self.edges[idim][0] - self.cell_size * extra_low[idim]
            new_max = self.edges[idim][-1] + self.cell_size * extra_high[idim]
            new_edges.append(
                np.linspace(new_min, new_max, num=new_shape[idim] + 1))
            new_cell_size = new_edges[idim][1] - new_edges[idim][0]
            if np.abs(new_cell_size - self.cell_size) / self.cell_size > 1e-6:
                raise ValueError(
                    f"Cell size changed during expansion "
                    f"({new_cell_size:.3f} vs. {self.cell_size:.3f} Mpc.)"
                )
            
        # Compute number of particles in each cell, across MPI ranks
        new_mask = self.build_mask_from_coordinates(
            r, new_edges, n_threshold=self.params['min_num_per_cell'],
            store=False, dither=self.params['dither_gap']
        )

        # Add in old mask
        new_mask[extra_low[0]:new_shape[0]-extra_high[0],
                 extra_low[1]:new_shape[1]-extra_high[1],
                 extra_low[2]:new_shape[2]-extra_high[2]
        ] += self.mask

        # Don't need the old mask/edges anymore, replace with updated ones
        self.mask = new_mask
        self.edges = new_edges
        self.shape = new_shape

        # If desired, activate padding cells
        if cell_padding_width > 0:
            all_cell_centres = self.get_cell_centres()
            target_cell_centres = target_mask.get_cell_centres(shape='3D')
            max_dist = cell_padding_width + np.sqrt(3) * self.cell_size

            tree = cKDTree(all_cell_centres)    # No wrapping necessary here!

            for ix in range(target_mask.shape[0]):
                for iy in range(target_mask.shape[1]):
                    for iz in range(target_mask.shape[2]):

                        # Only expand around active target mask cells!
                        if not target_mask.mask[ix, iy, iz]:
                            continue

                        r_target = target_cell_centres[ix, iy, iz, :]
                        ngbs = tree.query_ball_point(r_target, max_dist)
                        ngbs_3d = np.unravel_index(ngbs, new_shape)
                        new_mask[ngbs_3d] = True 

    def get_cell_centres(self, active_only=False, shape='1D'):
        """Find the centres of all cells (optionally: active ones only)."""
        all_cell_centres_grid = np.meshgrid(
            (self.edges[0][1:] + self.edges[0][:-1]) / 2,
            (self.edges[1][1:] + self.edges[1][:-1]) / 2,
            (self.edges[2][1:] + self.edges[2][:-1]) / 2,
            indexing='ij',
        )

        ncells = np.prod(self.shape)

        if shape == '1D':
            all_cell_centres = np.zeros((ncells, 3))
            for idim in range(3):
                all_cell_centres[:, idim] = all_cell_centres_grid[idim].ravel()
        elif shape == '3D':
            arr_shape = np.array([*self.shape, 3], dtype=int)
            all_cell_centres = np.zeros(arr_shape)
            for idim in range(3):
                all_cell_centres[..., idim] = all_cell_centres_grid[idim]
        else:
            raise ValueError(f"Invalid shape '{shape}'!")
            
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

    def compute_active_box(self):
        """Compute the boundary of active cells."""
        ind_sel = np.where(self.mask)   # 3-tuple of ndarrays!
        self.mask_box = np.zeros((2, 3))

        if len(ind_sel[0]) > 0:
            for idim in range(3):
                edges = self.edges[idim]
                ind_dim = ind_sel[idim]
                self.mask_box[0, idim] = np.min(edges[ind_dim])
                self.mask_box[1, idim] = np.max(edges[ind_dim + 1])
        else:
            # If there are no active mask cells at all, we artificially
            # set the min/max to 0
            self.mask_box[...] = 0

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
        """
        Re-center the mask such that the selection is centred on origin.

        This affects the internal coordinates of cells, but not their 'actual'
        position in the simulation frame. The shift is based on the current
        value of the "self.mask_box" array.

        """
        mask_offset = (self.mask_box[1, :] + self.mask_box[0, :]) / 2

        self.mask_box[0, :] -= mask_offset
        self.mask_box[1, :] -= mask_offset
        self.origin += mask_offset
        for idim in range(3):
            self.edges[idim] -= mask_offset[idim]

        # Check that mask_box does not extend by more than half the
        # simulation box size from the centre in any direction
        low = np.min(self.mask_box[0, :])
        high = np.max(self.mask_box[1, :])
        if (low < -0.5 * self.l_box or high > 0.5 * self.l_box):
            raise ValueError(
                f"Active mask cells extend from {low:.3f} --> {high:.3f} Mpc, "
                f"which exceeds the simulation half-box size "
                f"({self.l_box / 2:.2f} Mpc). This is not supported."
            )

        print(f"Re-centred mask by {mask_offset[0]:.3f} / "
              f"{mask_offset[1]:.3f} / {mask_offset[2]:.3f} Mpc.")

    def get_active_cell_centres(self):
        """Find the centre of all selected mask cells"""
        ind_sel = np.where(self.mask)   # Note: 3-tuple of ndarrays!
        self.cell_coords = np.vstack(
            (self.edges[0][ind_sel[0]],
             self.edges[1][ind_sel[1]],
             self.edges[2][ind_sel[2]]
            )
        ).T
        self.cell_coords += self.cell_size * 0.5

        n_sel = len(ind_sel[0])
        cell_fraction = n_sel * self.cell_size**3 / self.box_volume
        cell_fraction_cube = n_sel * self.cell_size**3 / self.mask_extent**3
        cell_fraction_sim = n_sel * self.cell_size**3 / self.l_box**3
        print(f'There are {n_sel:d} selected mask cells.')
        print(f'They fill {cell_fraction * 100:.3f} per cent of the bounding '
              f'box ({cell_fraction_cube * 100:.3f} per cent of bounding '
              f'cube, {cell_fraction_sim * 100:.3f} per cent of the '
              f'parent simulation).')

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
        ds.attrs.create('geo_centre', self.origin)
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
        ds.attrs.create('geo_centre', self.origin)
        ds.attrs.create('grid_cell_width', self.cell_size)


def compute_mips(m_part, cosmo_name):
    """Compute the total masses in the simulation volume."""
    cosmo = co.get_cosmology_params(cosmo_name)
    h = cosmo['hubbleParam']
    omega0 = cosmo['Omega0']
    omega_baryon = cosmo['OmegaBaryon']
    cosmo = FlatLambdaCDM(
        H0=h*100., Om0=omega0, Ob0=omega_baryon)

    rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc ** 3).value
    rho_mean = omega0 * rho_crit

    mips = np.cbrt(m_part / rho_mean)
    if comm_rank == 0:
        print(f"Mean inter-particle separation: {mips*1e3:.3f} kpc")
    return mips


def find_vertices(r, serial_only=False):
    """Find the vertices (lower and upper edges) of points."""
    box = np.zeros((2, 3))
    box[0, :] = sys.float_info.max
    box[1, :] = sys.float_info.min

    n_part = r.shape[0]

    if n_part > 0:
        box[0, :] = np.min(r, axis=0)
        box[1, :] = np.max(r, axis=0)

    # Now compare min/max values across all MPI ranks
    if not serial_only and comm is not None:
        for idim in range(3):
            box[0, idim] = comm.allreduce(box[0, idim], op=MPI.MIN)
            box[1, idim] = comm.allreduce(box[1, idim], op=MPI.MAX)

    return box


def periodic_wrapping(r, boxsize, return_copy=False, mode='centre'):
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
    mode : string, optional
        Specify whether coordinates should be wrapped to within -0.5 --> 0.5
        times the boxsize ('centre', default) or 0 --> 1 boxsize ('corner').

    Returns
    -------
    r_wrapped : ndarray(float) [N, 3]
        The wrapped coordinates. Only returned if `return_copy` is True,
        otherwise the input array `r` is modified in-place.

    """
    if mode in ['centre', 'center']:
        shift = 0.5 * boxsize
    elif mode in ['corner']:
        shift = 0.0
    else:
        raise ValueError(f'Invalid wrapping mode "{mode}"!')

    if return_copy:
        r_wrapped = ((r + shift) % boxsize - shift)
        return r_wrapped

    # To perform the wrapping in-place, break it down into three steps
    r += shift
    r %= boxsize
    r -= shift


def parse_arguments():
    """Parse the input arguments into a structure."""

    parser = argparse.ArgumentParser(
	description="Generate a mask for a zoom simulation.")
    parser.add_argument(
        'param_file', help='Parameter file with settings for the mask.')
    parser.add_argument(
        '-p', '--params',
        help='[Optional] Override one or more entries in the parameter file.'
             'The format is name1: value1[, name2: value2, ...]'
    )

    args = parser.parse_args()

    # Process parameter override values here...
    args.params = utils.process_param_string(args.params)
                
    # Some sanity checks
    if not os.path.isfile(args.param_file):
        raise OSError(f"Could not find parameter file {args.param_file}!")

    return args


# Allow using the file as stand-alone script
if __name__ == '__main__':
    args = parse_arguments()                                   
    MakeMask(args)              

