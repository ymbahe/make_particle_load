import os
import sys
import re
import yaml
import h5py
import numpy as np
import subprocess
from scipy.spatial import distance
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from mpi4py import MPI
from ParallelFunctions import repartition
#import MakeGrid as cy
from MakeParamFile import *
from scipy.io import FortranFile

import numpy as np
import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)
import MakeGrid as cy

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


class ParticleLoad:
    """
    Class to generate and save a particle load from an existing mask.

    Parameters
    ----------
    param_file : str
        The name of the YAML parameter file containing the settings for the
        particle load creation.
    randomize : bool, optional
        Shuffle particles on each MPI rank before load balancing them and
        writing to file (default: False).
    only_calc_ntot : bool, optional
        Only calculate the total number of particles, but do not generate
        any of them (default: False).
    verbose : bool, optional
        Switch to enable more detailed log messages (default: False).
    save_data : bool, optional
        Directly save the generated particle load (default: True).
    create_parameter_files : bool, optional
        Create parameter files for IC-Gen and SWIFT alongside particle load
        (default: False, not fully implemented).
    create_submit_files : bool, optional
        Create submission files for IC-Gen and SWIFT alongside particle load
        (default: False, not fully implemented).

    """
    def __init__(self, param_file: str, randomize: bool = False,
                 only_calc_ntot: bool = False, verbose: bool = False,
                 save_data: bool = True, create_parameter_files: bool = False,
                 create_submit_files: bool = False
    ) -> None:

        self.verbose = verbose

        # For now, hard-code SWIFT as only supported simulation type.
        # Maybe add support for Gadget(/4) later.
        # --> MOVE TO PARAM FILE
        self.sim_types = ['SWIFT']

        # Read and process the parameter file.
        self.read_param_file(param_file)

        # Generate particle load.
        self.nparts = {}
        self.scube = {}
        self.make_particle_load(only_calc_ntot=only_calc_ntot)

        # Generate param and submit files, if desired (NOT YET IMPLEMENTED).
        if comm_rank == 0:

            if create_parameter_files:
                self.save_param_files()

            if create_submit_files:
                self.save_submit_files()

        comm.barrier()

        # Save particle load
        if save_data:
            self.save_particle_load(randomize=randomize)

    def read_param_file(self, param_file: str) -> None:
        """Read in parameters for run."""

        # Read params from YAML file on rank 0, then broadcast to other ranks.
        if comm_rank == 0:
            params = yaml.safe_load(open(param_file))
        else:
            params = None
        params = comm.bcast(params, root=0)

        # Define and enforce required parameters.
        required_params = [
            'box_size',
            'n_particles',
            'glass_num',
            'f_name',
            'panphasian_descriptor',
            'ndim_fft_start',
            'is_zoom',
            'cosmology'
        ]
        for att in required_params:
            if att not in params:
                raise KeyError(f"Need to have {att} as required parameter.")

        # *** Re-structure this: use dicts instead of flooding self space ***

        # Some dummy placers.
        self.mask = None

        # Things of unclear nature
        self.radius = 0.
        self.ic_params = {
            'all_grid': False,
            'n_species': 1
        }

        self.mask_file = None  # Use a precomputed mask for glass particles.
        self.zone1_type = 'glass'
        self.zone2_type = 'glass'
        self.n_species = 1  # Number of DM species.

        self.constraint_phase_descriptor = '%dummy'
        self.constraint_phase_descriptor_path = '%dummy'
        self.constraint_phase_descriptor_levels = '%dummy'

        self.constraint_phase_descriptor2 = '%dummy'
        self.constraint_phase_descriptor_path2 = '%dummy'
        self.constraint_phase_descriptor_levels2 = '%dummy'

        # Save the particle load data.
        self.save_data = True
        self.save_as_hdf5 = False
        self.save_as_fortran = True

        # Make swift param files?
        self.make_swift_param_files = False
        self.swift_dir = './SWIFT_runs/'
        self.swift_exec_location = 'swiftexechere'
        self.swift_ic_dir_loc = '.'

        # Make ic gen param files?
        self.make_ic_param_files = False
        self.ic_dir = './ic_gen_output/'

        # Params for hi res grid.
        self.nq_mass_reduce_factor  = 0.5       # Mass of first nq level relative to grid
        self.zone3_ncell_factor     = 0.5       # Number of cells in Zone III shells compared to lowest-res in gcube
        self.skin_reduce_factor     = 8.        # What successive factors do high res skins reduce by
        self.min_num_per_cell       = 8         # Min number of particles in high res cell (must be cube).
        self.radius_factor          = 1.
        self.num_buffer_gcells      = 2         # Number of buffer cell skins.
        self.ic_region_buffer_frac  = 1.        # Buffer for FFT grid during ICs.

        # Default starting and finishing redshifts.
        self.starting_z = 127.0
        self.finishing_z = 0.0

        # Is DM only? (Only important for softenings).
        self.dm_only = False

        # Memory setup for IC gen code.
        # These need to be set to what you have compiled the IC gen code with.
        self.nmaxpart = 36045928
        self.nmaxdisp = 791048437
        self.mem_per_core = 18.2e9
        self.max_particles_per_ic_file = 400**3

        # What type of IDs to use.
        self.use_ph_ids = True
        self.nbit = 21  # 14 for EAGLE

        # How many times the n-frequency should the IC FFT at least be?
        self.fft_times_fac = 2.

        # If a zoom, use multigrid IC grid?
        self.multigrid_ics = True

        # Info for submit files.
        self.n_cores_ic_gen = 28
        self.n_cores_gadget = 32
        self.n_nodes_swift = 1
        self.num_hours_ic_gen = 24
        self.num_hours_swift = 72
        self.ncores_node = 28

        # Params for outer low res particles
        self.zone3_min_n_cells = 20
        self.zone3_max_n_cells = 1000

        # For special set of params.
        self.template_set = 'dmo'

        # Is this a slab simulation? Should always be False, to be removed.
        self.is_slab = False

        # Use glass files to surround the high res region rather than grids?
        self.glass_files_dir = './glass_files/'

        # Softening length for zooms, in units of mean inter-particle distance
        self.softening_ratio_background = 0.02  # 1/50 M-P-S.

        # Default GADGET executable.
        self.gadget_exec = 'P-Gadget3-DMO-NoSF'

        # Relace with param file values.
        for att in params.keys():
            setattr(self, att, params[att])

        # Assign cosmology.
        if self.which_cosmology == 'Planck2013':
            self.Omega0 = 0.307
            self.OmegaLambda = 0.693
            self.OmegaBaryon = 0.04825
            self.HubbleParam = 0.6777
            self.Sigma8 = 0.8288
            self.linear_ps = 'extended_planck_linear_powspec'
        elif self.which_cosmology == 'Planck2018':
            self.Omega0 = 0.3111
            self.OmegaLambda = 0.6889
            self.OmegaBaryon = 0.04897
            self.HubbleParam = 0.6766
            self.Sigma8 = 0.8102
            self.linear_ps = 'EAGLE_XL_powspec_18-07-2019.txt'
        else:
            raise ValueError("Invalid cosmology")

        # Make sure coords is a numpy array.
        self.centre = np.array(self.centre)

        # Sanity check: number of particles must be an integer multiple of the
        # glass file particle number.
        if not (self.n_particles / self.glass_num) % 1 < 1e-6:
            raise ValueError(
                f"The number of particles ({self.n_particles}) must be an "
                f"integer multiple of the number per glass file "
                f"({self.glass_num})!")

    def compute_masses(self):
        """
        Compute the total and per-particle masses for the simulation volume.

        The calculation takes the specified cosmology into account. All
        particles are assigned equal masses, based on the box volume and
        specified number of particles N_part.

        For each particle, three mass types are computed:
        - m_dmo: the mass appropriate for a DM-only simulation (i.e.
            distributing the entire box mass over N_part particles).
        - m_dm: the mass for dark matter particles (i.e. distributing the
            fraction of total mass in dark matter over N_part particles).
        - m_gas: the mass for gas particles (i.e. distributing the fraction
            of total mass in baryons over N_part particles).

        ** TODO ** Update this, since we only need total box mass...

        """
        if comm_rank == 0:
            h = self.HubbleParam
            cosmo = FlatLambdaCDM(
                H0=h*100., Om0=self.Omega0, Ob0=self.OmegaBaryon)
            rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc ** 3).value
            omega_dm = self.Omega0 - self.OmegaBaryon

            box_volume = (self.box_size / h)**3
            m_tot = self.Omega0 * rho_crit * box_volume
            m_tot_dm = omega_dm * rho_crit * box_volume
            m_tot_gas = self.OmegaBaryon * rho_crit * box_volume

            m_dm = m_tot_dm / self.n_particles
            m_dmo = m_tot / self.n_particles
            m_gas = m_tot_gas / self.n_particles

            if self.verbose:
                print(f"Dark matter only particle mass: {m_dmo:.3f} M_sun "
                      f"(={m_dmo * h / 1e10:.3f} x 10^10 h^-1 M_sun)")
                print(f"Dark matter particle mass: {m_dm:.3f} M_sun "
                      f"(={m_dm * h / 1e10:.3f} x 10^10 h^-1 M_sun)")
                print(f"Gas particle mass: {m_gas:.3f} M_Sun" 
                      f"(={m_gas * h / 1e10:.3f} x 10^10 h^-1 M_Sun)")
        else:
            m_tot = None
            m_dm = None
            m_gas = None
            m_dmo = None

        # Send masses to all MPI ranks and store as class attributes
        self.m_tot = comm.bcast(m_tot)
        self.m_dmo = comm.bcast(m_dmo)
        self.m_dm = comm.bcast(m_dm)
        self.m_gas = comm.bcast(m_gas)

    def generate_gcells(self):
        """
        Generate data for all local gcells, but don't populate them yet.

        For uniform resolution simulations, this simply declares all gcells
        as type 0 (target resolution). For zooms, it assigns each gcell a type
        depending on its location relative to the input mask. Then, we work
        out how many particles each cell should contain.

        Parameters
        ----------
        None

        Returns
        -------
        gcells : dictionary
            A dictionary with five keys specifying the local gcell properties:
            - index : ndarray(int) [N_cells]
                The scalar index of each gcell.
            - pos : ndarray(float) [N_cells, 3]
                The central coordinates of all local gcells, in units of the
                gcell side length and with the origin at the centre of the
                full gcell cube (gcube).
            - types : ndarray(int) [N_cells]
                The type (i.e. resolution level) of each local gcell.
            - num : int
                The total number of local gcells.
            - memsize : int
                The memory footprint in byte of the gcell arrays.

        """
        gcube = self.gcube

        # Assign indices of local gcells, such that all cells with
        # index % comm_size == comm_rank end up on local MPI rank
        gcell_idx = np.arange(comm_rank, gcube['num_cells'], comm_size)
        n_gcells = len(gcell_idx)

        # Calculate the central coordinates of local gcells
        gcell_pos = cell_centre_from_index(gcell_idx, [gcube['n_cells']] * 3)

        if self.is_zoom:

            # Rescale coordinates of mask cells to gcell coordinates. Recall
            # that the mask cell coordinates already have the same origin.
            f_scale = 1. / gcube['cell_size']
            mask_pos = self.mask_data['cell_coordinates'] * f_scale
            mask_cell_size = self.mask_data['cell_size'] * f_scale

            # Check that the values make sense
            cmin = np.min(mask_pos)
            cmax = np.max(mask_pos + mask_cell_width)
            if cmin < -gcube['n_cells'] / 2:
                raise ValueError(
                    f"Minimum mask coordinate in cell units is {cmin}, but "
                    f"it must be above {-gcube['n_cells']/2}!"
                )
            if cmax > gcube['n_cells'] / 2:
                raise ValueError(
                    f"Maximum mask coordinate in cell units is {cmax}, but "
                    f"it must be below {gcube['n_cells']/2}!"
                )
            if comm_rank == 0:
                for idim, name in enumerate(['x', 'y', 'z']):
                    print(
                        f"Mask coords: {mask_pos[:, idim].min():.5f} <= "
                        f"{name} <= {mask_pos[:, idim].max():.5f} gcells"
                    )
                print(f"Mask cell size = {mask_cell_size:.2f} gcells.")

            # Assign a type (resolution level) to each local gcell.
            # Those that overlap with the mask are type 0 (i.e. target res.),
            # others have >= 1 depending on distance from the mask.
            gcell_types = np.ones(len(offsets), dtype='i4') * -1
            cy.assign_mask_cells(
                mask_pos, mask_cell_size, gcell_pos, gcell_types)
            self._assign_zone2_types(gcell_pos, gcell_idx, gcell_types)

        else:
            # If this is not a zoom simulation, all gcells are type 0.
            gcell_types = np.zeros(n_gcells, dtype='i4')

        # Total memory size gcell structure
        memory_size_in_byte = (
            sys.getsizeof(gcell_types) + sys.getsizeof(gcell_idx) +
            sys.getsizeof(gcell_pos)
        )

        # Final step: work out particle load info for all cells
        self.prepare_gcube_particles(gcell_types)
        return {'index': gcell_idx, 'pos': gcell_pos, 'types': gcell_types,
                'num': n_gcells, 'memsize': memory_size_in_byte}

    def _assign_zone2_types(self, pos, index, types):
        """
        Assign types (resolution level) to the gcells outside the target mask.

        Starting from the target-resolution gcells (type 0), 
        neighbouring gcells of those of type i are iteratively assigned type
        i + 1.

        This is an MPI-collective function; ranks need to exchange information
        about gcell neighbours hosted on other ranks.

        Call path: generate_gcells()

        Parameters
        ----------
        pos : ndarray(float) [N_gcells, 3]
            The coordinates (in cell size units) of the local gcell centres.
        index : ndarray(int) [N_gcells]
            Scalar indices of all local gcells
        types : ndarray(int) [N_gcells]
            Types of all local gcells. Initially -1 for all gcells that are
            not type 0; those are updated to >=1 on return.

        Returns
        -------
        None (updates input array `types`).

        """
        num_tot_zone2 = comm.allreduce(np.count_nonzero(types == -1))
        if comm_rank == 0:
            print(f"Assigning resolution level to {num_tot_zone2} gcells "
                  f"outside target high-res region.")

        # Initialize loop over (undetermined) number of gcell types.
        num_to_assign = num_tot_zone2
        source_type = 0
        num_assigned = 0

        while num_to_assign > 0:

            # Find direct neighbours of all (local) gcells of current type
            ind_source = np.nonzero(types == source_type)[0]
            ngb_indices = self._find_neighbour_gcells(pos[ind_source, :])

            # Combine skin cells across MPI ranks, removing duplicates
            ngb_indices = mpi_combine_arrays(ngb_indices, unicate=True)

            # Find local, unassigned gcells amongst the neighbours
            idx = np.where((np.isin(index, ngb_indices)) & (types == -1))[0]
            types[idx] = source_type + 1

            # Update number of assigned and still-to-be-assigned gcells
            num_assigned += comm.allreduce(
                np.count_nonzero(cell_types == this_type))
            num_to_assign = comm.allreduce(np.count_nonzero(cell_types == -1))
            if num_to_assign == 0:
                break

            source_type += 1

        if num_assigned != num_tot_zone2:
            raise ValueError(
                f"Assigned {num_assigned} zone-II gcells, not {num_tot_zone2}")
        if comm_rank == 0:
            print(f"   ... assigned them to {this_type} skin levels.")

    def _find_neighbour_gcells(self, pos_source, max_dist=1.):
        """
        Compute the indices of gcells that are near source cells.

        Neighbour cells are defined by their distance from a source cell,
        they can be local (on same rank) or remote.

        This is a subfunction of _assign_zone2_types(). Because that is an
        MPI-collective function, we may also get here with an empty source
        list; in this case, None is returned immediately.

        Parameters
        ----------
        pos_source : ndarray(float) [N_source, 3]
            The central coordinates of source cells, in gcell size units.
        max_dist : float, optional
            The maximum distance between cell centres for a gcell to be tagged
            as a neighbour. Default: 1.0, i.e. only the six immediate
            neighbour cells are tagged.

        Returns
        -------
        ngb_inds : ndarray(int) [N_ngb]
            The global scalar indices of all identified neighbours.

        """
        n_source = pos_source.shape[0]
        if n_source == 0:
            return

        # Build a kernel that contains the positions of all neighbour cells
        # relative to the target centre.
        max_kernel_length = 2*int(np.floor(max_dist)) + 1
        kernel = self.generate_uniform_grid(n=max_kernel_length, centre=True)
        kernel_r = np.linalg.norm(kernel, axis=1)
        ind_true_kernel = np.nonzero(kernel_r <= max_dist)[0]
        kernel = kernel[ind_true_kernel, :]
        n_kernel = len(ind_true_kernel)

        # Apply the kernel coordinate offsets to every source position
        n_source = pos_source.shape[0]
        pos_stretched = pos_source[np.arange(n_kernel*n_source) // n_kernel, :]
        ngb_pos = pos_stretched + np.repeat(kernel, n_source, axis=0)

        # Find the indices of neighbours that are actually within the gcube
        ind_in_gcube = np.nonzero(
            np.max(np.abs(ngb_pos, axis=1)) <= self.gcube['n_cells'] // 2)[0]
        ngb_inds = cell_index_from_centre(
            ngb_pos[ind_in_gcube, :], self.gcube['n_cells'])

        return ngb_inds

    def find_nq_slab(self, suggested_nq, slab_width, eps=1.e-4):

        # What is half a slab width? (our starting point)/
        half_slab = slab_width / 2.

        # Dict to save info.
        self.nq_info = {'diff': 1.e20, 'n_tot_lo': 0}

        # What are we starting from?
        if comm_rank == 0:
            print('Computing slab nq: half_slab=%.2f suggested_nq=%i' % (half_slab, suggested_nq))

        # You start with a given suggested nq, then you remove nq_reduce from it at each level.
        # This tries multiple nq_reduce values.
        for nq_reduce in np.arange(1, 10):

            # Loop over an arbitrary number of starting nqs to test them (shouldn't ever need).
            for i in range(200):

                # If this takes me >50% away from the suggested nq, don't bother.
                if np.true_divide(suggested_nq - i, suggested_nq) < 0.5:
                    break

                # Reset all values.
                offset = half_slab  # Starting from the edge of the slab.
                this_nq = suggested_nq - i  # Trying this starting nq.
                starting_nq = suggested_nq - i
                nlev_slab = 0  # Counting the number of levels.
                n_tot_lo = 0  # Counting the total number of low res particles.

                # Iterate levels for this starting nq until you reach the edge of the box.
                while True:
                    # Cell length at this level.
                    m_int_sep = self.box_size / float(this_nq)

                    # Add this level.
                    offset += m_int_sep
                    nlev_slab += 1

                    # How close are we to the edge.
                    diff = (offset - self.box_size / 2.) / self.box_size

                    # If adding the level has gone outside the box, then stop.
                    # Or if we are really close to the edge.
                    if (offset > self.box_size / 2.) or np.abs(diff) <= 1.e-3:

                        # Try to make it a bit better.

                        # First go back to the previous level.
                        offset -= m_int_sep

                        # Loop over a range of new nq options for the last level/
                        for extra in np.arange(-nq_reduce, nq_reduce, 1):
                            # Try a new nq for the last level.
                            this_nq += extra
                            m_int_sep = self.box_size / float(this_nq)
                            diff = (offset + m_int_sep - self.box_size / 2.) / self.box_size

                            # Have we found a new best level?
                            if np.abs(diff) < np.abs(self.nq_info['diff']):
                                self.nq_info['diff'] = diff
                                self.nq_info['starting_nq'] = starting_nq
                                self.nq_info['finishing_nq'] = this_nq - extra
                                self.nq_info['nq_reduce'] = nq_reduce
                                self.nq_info['extra'] = extra
                                self.nq_info['nlev_slab'] = nlev_slab
                                if (nlev_slab - 1) % comm_size == comm_rank:
                                    n_tot_lo += 2 * this_nq ** 2
                                self.nq_info['n_tot_lo'] = n_tot_lo
                                self.nq_info['dv_slab'] = -1.0 * (
                                        (offset + m_int_sep - self.box_size / 2.) * self.box_size ** 2.
                                ) / self.nq_info['finishing_nq'] ** 2.

                                # Reset for next try.
                                if (nlev_slab - 1) % comm_size == comm_rank:
                                    n_tot_lo -= 2 * this_nq ** 2

                            # Reset for next try.
                            this_nq -= extra
                        break
                    else:
                        if (nlev_slab - 1) % comm_size == comm_rank:
                            n_tot_lo += 2 * this_nq ** 2

                    # Compute nq for next level.
                    this_nq -= nq_reduce

                    # We've gotten too small. 
                    if this_nq < 10:
                        break

        if comm_rank == 0:
            print((
                      '[Rank %i] Best nq: starting_nq=%i, finishing_nq=%i, extra=%i, diff=%.10f, nlev_slab=%i, '
                      'nq_reduce=%i, dv_slab=%.10f n_tot_lo=%i'
                  ) % (
                comm_rank,
                self.nq_info['starting_nq'],
                self.nq_info['finishing_nq'],
                self.nq_info['extra'],
                self.nq_info['diff'],
                self.nq_info['nlev_slab'],
                self.nq_info['nq_reduce'],
                self.nq_info['dv_slab'],
                self.nq_info['n_tot_lo']
            ))
        assert np.abs(self.nq_info['diff']) <= eps, 'Could not find good nq'

        return self.nq_info['n_tot_lo']

    def find_scube_structure(self, target_n_cells, side, eps=0.01):
        """
        Work out the optimal structure of the nested shell cube in Zone III.
    
        The space between the gcube and simulation box edge is filled with a
        series of self-similar cubic shells (i.e., the outermost cells of a
        uniform cube). It should satisfy three boundary conditions:

        (i)   At its inner edge (on the boundary to the gcube), the cell size
              should be similar to (slightly larger) than the mean inter-
              particle separation in the lowest-resolution gcube cells. The
              (pre-computed) target value is target_n_scells.
        (ii)  All shells (except possibly the outermost, see below) have the
              same number of cells. This implies that successive shells differ
              in side length by a factor of n_cells / (n_cells - 2). 
        (iii) The outer edge of the outermost shell should lie on the edge
              of the simulation box.

        In general, all three conditions cannot be met simultaneously. We
        therefore try a number of n_cells values close to target_n_cells, and
        also allow the outermost shell to have slightly fewer (more) cells
        than the others, to make it slightly thicker (thinner) and get closer
        to the simulation box edge.

        Typically, at least one combination of n_cells and n_extra (the
        difference in cell number of the outermost shell) leads to a structure
        whose volume is correct to within a fraction of a per cent. We deal
        with this remaining difference later when generating the particles.

        Parameters
        ----------
        target_n_cells : int
            The target number of cells per dimension in each shell.
        tolerance_n_cells : int, optional
            The maximum deviation from the target cell number (default: 5).
        max_extra : int, optional
            The maximum difference between the number of cells in the outermost
            and other shells (default: 10). 
        eps : float, optional
            The maximum tolerance in the total volume of the scube, as a
            fraction of the exact Zone III volume (default: 0.01). 

        Returns
        -------
        num_part : int
            Total number of Zone III particles generated on this rank.

        Class attributes
        ----------------
        scube : dict
            Properties of the cube (with a central hole...) formed by the
            set of cubic shells in Zone III.

        Notes
        -----
        All lengths in this function are expressed in units of the simulation
        box size (self.box_size).
        
        """ 
        if target_n_cells < 5:       # "Dat kan niet."
            raise ValueError(
                f"Cannot build scube with target n_cells = {target_n_cells}.")
        if target_n_cells < 25 and comm_rank == 0:
            print(f"WARNING: target_n_cells = {target_n_cells}, limited "
                  f"range available for variations.")

        gcube_length = self.gcube['sidelength'] / self.box_size

        # Set up arrays of all to-be-evaluated combinations of n_cells and
        # n_extra. For extra, we first set up a slighly offset float array
        # and reduce it later to test n_extra = 0 twice (see below).
        n_1d = np.arange(np.max(target_n_cells - tolerance_n_cells, 10),
                         target_n_cells + tolerance_n_cells + 1)
        nx_1d = np.arange(-max_extra-0.5, max_extra+1.5)
        n, nx = np.meshgrid(n_1d, nx_1d, indexing='ij')
        n = n.flatten()
        nx = nx.flatten()

        # Negative (positive) n_extra only makes sense if the volume is
        # otherwise slightly too small (large), i.e. if we pick a number of
        # shells just below (above) the ideal fractional value. For
        # n_extra = 0, we test both.
        delta_ns = np.clip(np.sign(nx), 0, 1).astype('int')
        n_extra = nx.astype('int')

        # Reject combinations that have too few cells in outermost shell
        ind_useful = np.nonzero(n + n_extra >= 10)[0]
        n = n[ind_useful]
        n_extra = n_extra[ind_useful]
        delta_ns = delta_ns[ind_useful]

        # Find increase factor f between successive shells and total number
        # [ideal number of shells is log_f(box_size / gcube_size)]
        factors = n / (n - 2)
        ns_ideal = np.log10(1 / gcube_length) / np.log10(factors)
        ns = (np.floor(ns)).astype(int) + delta_ns
    
        # Volume enclosed by next-to-outermost shell
        v_inner = factors**(3 * (ns - 1))

        # Volume of outermost shell, accounting for different cell number
        n_outer = n + n_extra
        num_cells_outer = 6 * n_outer * n_outer - 12 * n_outer + 8
        cellsizes_outer = factors**(ns - 1) / (n_outer - 2)
        v_outer = num_cells_outer * cellsizes_outer**3

        # Fractional difference in volume from ideal value
        # (recall that the simulation box has length = volume = 1 here)
        v_tot = v_inner + v_outer
        v_diff = (v_tot - 1) / (1 - gcube_length**3)

        # Find the combination with the smallest v_diff
        ind_best = np.argmin(np.abs(v_diff))
        if v_diff[ind_best] > eps:
            raise Exception(
                f"Could not find acceptable scube parameters "
                f"(best combination differs by {v_diff[ind_best] * 100} "
                f"per cent, max allowed is {eps * 100})."
            )

        self.scube['n_cells'] = n[ind_best]
        self.scube['num_shells_all'] = ns[ind_best]
        self.scube['n_extra'] = n_extra[ind_best]
        self.scube['volume'] = vtot[ind_best] - gcube_length**3
        self.scube['delta_volume_fraction'] = vdiff[ind_best]

        # Compute number of shells and Zone III particles for this rank.
        # (we will assign successive shells to different MPI ranks).
        tot_nshells = self.scube['num_shells_all']
        nc = self.scube['n_cells']
        num_shells_local = tot_nshells // comm_size
        if comm_rank < tot_nshells % comm_size:
            num_shells_local += 1
        self.scube['num_shells_local'] = num_shells_local

        # Do we get the outermost shell?
        have_outer_shell = (tot_nshells - 1) % comm_size == comm_rank
        if have_outer_shell:
            num_part_inner = (6*nc*nc - 12*nc + 8) * (num_shells_local - 1)
            nc_outer = nc + self.scube['n_extra']
            num_part_outer = (6*nc_outer*nc_outer - 12*nc_outer + 8)
            self.scube['num_part_local'] = num_part_inner + num_part_outer
        else:
            num_part_local = (6*nc*nc - 12*nc + 8) * num_shells_local
            self.scube['num_part_local'] = num_part_local

        self.scube['num_part_all'] = comm.allreduce(
            self.scube['num_part_local'])

        # Report what we found.
        if comm_rank == 0:
            print(
                f"Found optimal scube structure:\n"
                f"  N_cells = {n[ind_best]} (target: {target_n_cells})\n"
                f"  N_extra = {n_extra[ind_best]}\n"
                f"  N_shells = {ns[ind_best]} (ideal: {ns_ideal[ind_best]})\n"
                f"  Volume = {self.scube['volume']}\n"
                f"  Volume deviation = {vdiff[ind_best]}\n"
                f"  Total particle number = {self.scube['num_part_all']}\n"
            )

        return self.scube['num_part']

    def prepare_gcube_particles(self, gcell_types):
        """ 
        Work out how to populate gcells with particles.

        This function calculates how many particles of which mass should be
        generated within each gcell.

        Parameters
        ----------
        gcell_types : ndarray(int) [N_cells]
            Types of local gcells.

        Returns
        -------
        gcell_info : dict
            Information about particle load per assigned gcell type:
            - type : ndarray(int)
                The gcell types, in the same order as other entries.
            - num_part_per_gcell : ndarray(int)
                The number of particles generated within each gcell of a given
                type.
            - particle_pmass : ndarray(float)
                Pseudo masses (really, volume fractions) of particles in each
                gcell type.
            - num_gcells : ndarray(int)
                Number of (local) gcells of each type.
        npart_gcube : int
            The (combined) number of zone 1 and zone 2 particles that will be
            generated on the local rank.

        Class attributes stored
        ----------------------- 
        nparts :
            - 'zone1_local', 'zone2_local', 'tot_local' : int
                The local (this rank) number of particles in zone1, zone2, and
                total (last one will be updated later for zone3).
            - 'zone1_all', 'zone2_all', 'tot_all' : int
                The total (cross-MPI) number of particles in zone1, zone2, and
                total (last one will be updated later for zone3).

        """
        # Number of high resolution particles this rank will have.
        self.gcell_info = {
            'type': [],
            'num_part_per_gcell': [],
            'particle_pmass': [],
            'num_gcells': [],
        }

        npart_zone1 = 0
        npart_zone2 = 0
        max_gcell_load = 0
        min_gcell_load = int(1e30)   # Initialize to impossibly large value
        num_part_equiv_high = None

        # Loop over all gcell types (i.e. particle masses) and find how
        # many particles (and total mass) will be assigned to it
        for itype in np.unique(gcell_types):

            ind_type = np.nonzero(cell_types == itype)[0]
            num_gcells_type = len(ind_type)
            self.gcell_info['type'].append(itype)
            self.gcell_info['num_gcells'].append(num_gcells_type)

            # Find ideal number of particles to put into each gcell
            if itype == 0:
                zone_type = self.zone1_type
                target_gcell_load = self.glass_num
            else:
                zone_type = self.zone2_type
                reduction_factor = self.skin_reduce_factor * itype
                target_gcell_load = max(self.min_num_per_cell,
                    int(np.ceil(self.glass_num / reduction_factor)))

            # Apply quantization constraints to find actual particle number
            if zone_type == 'glass':
                gcell_load = (self.find_nearest_glass_file(target_gcell_load))
            else:
                gcell_load = self.find_next_cube(target_gcell_load)
            self.gcell_info['num_part_per_gcell'].append(gcell_load)
            min_gcell_load = min(min_gcell_load, gcell_load)
            max_gcell_load = max(max_gcell_load, gcell_load)

            # Pseudo-mass of each particle in current gcell type
            gcell_volume_frac = (gcube['cell_size'] / self.box_size)**3
            pmass_type = gcell_volume_frac / gcell_load
            self.cell_info['particle_pmass'].append(pmass_type)

            # Total number of particles across cells
            num_part_type = num_gcells_type * num_part_per_gcell
            if itype == 0:
                npart_zone1 = num_part_type
            else:
                npart_zone2 += num_part_type

        # Convert cell_info lists to ndarrays
        for key in self.gcell_info:
            self.gcell_info[key] = np.array(self.gcell_info[key])

        # Total number of zone 1 and 2 particles locally and across MPI ranks.
        self.nparts['zone1_local'] = npart_zone1
        self.nparts['zone2_local'] = npart_zone2
        self.nparts['tot_local'] = npart_zone1 + npart_zone2
        self.nparts['zone1_all'] = comm.allreduce(npart_zone1)
        self.nparts['zone2_all'] = comm.allreduce(npart_zone2)
        self.nparts['tot_all'] = (
            self.nparts['zone1_all'] + self.nparts['zone2_all'])

        gcell_load_range = [min_gcell_load, max_gcell_load]
        self._find_global_resolution_range(gcell_load_range)
        self._find_global_pmass_distribution()

    def _find_global_resolution_range(self, gcell_load_range):
        """
        Find and print the global min/max particle load per gcell.
        Call path: prepare_gcube_particles() <- generate_gcells().

        Parameters
        ----------
        gcell_load_range : [int, int]
            The local min and max number of particles per gcell.

        Attributes updated
        ------------------
        scube['n_equiv_lowest_gcube'] : int
            Number of particles per dimension if all gcells were at the
            lowest assigned resolution.

        """
        gcube = self.gcube

        # Find the particle number per dimension that would fill the gcube
        # at the highest and lowest resolutions
        lowest_gcell_load = comm.allreduce(gcell_load_range[0], op=MPI.MIN)
        highest_gcell_load = comm.allreduce(gcell_load_range[1], op=MPI.MAX)
        num_part_equiv_low = lowest_gcell_load * gcube['num_cells']
        num_part_equiv_high = highest_gcell_load * gcube['num_cells']

        # Print some statistics about the load factors
        if comm_rank == 0:
            for ii, name in enumerate(['Target', 'Lowest']):
                if ii == 0:
                    neq = num_part_equiv_high
                    load = num_highest_res
                else:
                    neq = num_part_equiv_low
                    load = num_lowest_res
                nbox = neq * self.box_size**3 / gcube['volume']
                print(f"{name} gcell load is {load} ({load**(1/3):.2f}^3) "
                      f"particles.")
                print(
                    f"This corresponds to {neq} ({neq**(1/3):.2f}^3) "
                    f"particles in the gcube, {nbox} ({nbox**(1/3):.2f}^3) "
                    f"particles in the entire simulation box, and "
                    f"{neq / gcube['volume']:.3f} particles per (cMpc/h)^3."
                )       

        self.scube['n_equiv_lowest_gcube'] = (
            lowest_gcell_load * gcube['n_cells'])

    def prepare_zone3_particles(self, slab_width=None):
        """
        Set up the information required to populate Zone III with particles.

        This is a thin wrapper around self.find_scube_structure(), which
        fills the dict `scube` with information about the cubic shell
        structure holding the particles.

        Called by: make_particle_load()

        Parameters
        ----------
        slab_width : float, optional
            Width of the slab, only in (not fully supported) slab mode...

        Returns
        -------
        None

        Class attributes updated
        ------------------------
        nparts :
            - 'zone3_local', 'tot_local' : int
                The local (this rank) number of particles in zone3 and total.
            - 'zone3_all', 'tot_all' : int
                The total (cross-MPI) number of particles in zone3 and total.

        """
        if not self.is_zoom:
            return 0

        # Ideal number of scells per dimension: a bit lower than the equivalent
        # of the largest inter-particle spacing in the gcube.
        target_n_scells = int(np.clip(
            self.scube['n_equiv_lowest_gcube'] * self.zone3_ncell_factor,
            self.zone3_n_cell_min, self.zone3_n_cell_max
        ))
        if comm_rank == 0:
            print(f"Zone III cubic shells should have a target number of "
                  f"{target_n_scells} cells per dimension.")

        if self.is_slab:
            npart_zone3 = self.find_nq_slab(suggested_nq, slab_width)
        else:
            npart_zone3 = self.find_scube_structure(target_n_scells)

        self.nparts['zone3_local'] = npart_zone3
        self.nparts['tot_local'] += npart_zone3
        self.nparts['zone3_all'] = comm.allreduce(npart_zone3)
        self.nparts['tot_all'] += self.nparts['zone3_all']

    def find_nearest_glass_file(self, num):
        """Find the glass file with the number of particles closest to num."""
        files = os.listdir(self.glass_files_dir)
        num_in_files = [int(_f.split('_')[2]) for _f in files if 'ascii' in x]
        num_in_files = np.array(num_in_files, dtype='i8')
        index_best = np.abs(num_in_files - num).argmin()
        return files[index_best]

    def _find_global_mass_distribution(self):
        """
        Find and store the global distribution of (pseudo-)masses. This is a
        subfunction of find_npart_in_gcube()." 

        """
        # Gather all globally occurring Zone I+II particle masses
        local_masses = self.gcell_info['particle_mass']
        all_masses = np.unique(np.concatenate(comm.allgather(local_masses)))
        self.zone1_mass = np.min(all_masses)
        zone1_mass_msun = self.zone1_mass * self.total_box_mass
        if self.is_zoom:
            ind_zone2 = np.nonzero(all_masses > self.zone1_mass)[0]
            self.min_zone2_mass = np.min(all_masses[ind_zone2])
            self.max_zone2_mass = np.max(all_masses[ind_zone2])
            zone2_mass_msun = (
                self.min_zone2_mass * self.total_box_mass,
                self.max_zone2_mass * self.total_box_mass
            )

        if comm_rank == 0:
            print(
                f"Zone I particles have a log pseudo-mass "
                f"{np.log10(self.zone1_pmass):.3f} ({zone1_mass:.3f} M_Sun)."
            )
            if self.is_zoom:
                print(
                    f"Zone II particle log pseudo-mass range is "
                    f"{np.log10(self.min_zone2_pmass):.3f} - "
                    f"{np.log10(self.max_zone2_pmass):.3f} "
                    f"({zone2_mass[0]:.3f} - {zone2_mass[1]:.3f} M_Sun)."
                )

    def generate_gcube_particles(self, gcells, parts):
        """
        Generate the particles that fill the mask cube.

        This includes Zone I (the volume filled by the mask cells, i.e. the
        target high-resolution region), as well as Zone II (the rest of the
        cube enclosing the mask). Both can be realised as glass or grid.

        Parameters
        ----------
        gcells : dict
            Properties of local gcells that are to be populated.
        parts : dict
            Dict containing the (to be filled) coordinate and mass arrays
            for the local particles.

        Returns
        -------
        None

        """
        gcube = self.gcube
        npart_gcube = self.nparts['zone1_local'] + self.nparts['zone2_local']

        # Load all the glass files we need into a list of coordinate arrays
        glass = self.load_glass_files()

        # Loop over each gcell type and fill them with particles
        parts_created = 0
        for iitype, itype in enumerate(self.cell_info['type']):
            ind_type = np.where(gcells['types'] == itype)[0]
            ncell_type = len(ind_type)
            if ncell_type == 0:
                raise Exception(f"Did not find any gcells of type {itype}!")
            
            glass_num_type = self.gcell_info['glass_num'][iitype]
            npart_type = self.gcell_info['num_particles_per_cell'][iitype]
            particle_mass_type = self.cell_info['particle_mass'][iitype]

            if glass_num_type is None:
                # This indicates that we should use a grid
                kernel = self.generate_uniform_grid(num=npart_type)    
            else:
                kernel = glass[glass_num_type]

            cy.fill_gcells_with_particles(
                gcells['pos'][ind_type], kernel, parts['pos'],
                parts['masses'], particle_mass_type, parts_created
            )
            np = ncell_type * npart_type
            parts_created += np

            # Brag about what we have done.
            if self.verbose or comm_rank == 0:
                log_mass = np.log10(particle_mass_type)
                true_mass = particle_mass_type * self.total_box_mass
                print(
                    f"[{comm_rank}: Type {itype}] num_part = {np} "
                    f"({np**(1/3):.3f}^3) in {ncell_type} cells "
                    f"({npart_type}/cell), log pseudo-mass {log_mass:.4f} "
                    f"(DMO mass {true_mass:.3f} M_sun)."
                )

        num_part_total = comm.allreduce(parts_created)
        if comm_rank == 0:
            print(
                f"Generated {num_part_total} ({num_part_total**(1/3):.3f}^3) "
                f"particles in high-res region."
            )

        # Scale coordinates to units of the simulation box size (i.e. such
        # that the edges of the simulation box (not gcube!) are at coordinates 
        # -0.5 and 0.5.
        gcube_range_inds = np.array((-0.5, 0.5)) * gcube['n_cells']
        gcube_range_boxfrac = (
            np.array((-0.5, 0.5)) * max_boxsize / self.box_size)
        rescale(parts['pos'][:, :parts_created],
                gcube_range_inds, gcube_range_boxfrac, in_place=True)

        # Do consistency checks in separate function for clarity.
        self._verify_gcube_region(parts, parts_created, gcube['volume'])

    def load_glass_file(self, num):
        """
        Load the glass file for high resolution particles.

        Parameters
        ----------
        num : int
            Suffix of the specific glass file to use.

        Returns
        -------
        r_glass : ndarray(float) [N_glass, 3]
            Array containing the coordinates of all particles in the glass
            distribution. If grids are used for both Zone I and Zone II
            (i.e. we don't use glass anywhere), None is returned instead.

        """
        if self.zone1_type == 'grid' and self.zone2_type == 'grid':
            return None

        glass_file = f"{self.glass_files_dir}/ascii_glass_{num}"
        if not os.path.isfile(glass_file):
            raise OSError(f"Specified glass file {glass_file} does not exist!")

        dtype = {'names': ['x', 'y', 'z'], 'formats': ['f8']*3}
        glass = np.loadtxt(glass_file, dtype=dtype, skiprows=1)
        if comm_rank == 0:
            print(f"Loaded glass file with {num} particles.")

        return glass

    def load_mask_file(self):
        """
        Load the (previously computed) mask file that defines the zoom region.

        This is only relevant for zoom simulations. The mask file should have
        been generated with the `MakeMask` class in `make_mask/make_mask.py`.
        Subfunction of set_up_gcube().

        An error is raised if no mask file is specified or if the specified
        file does not exist.

        """
        if comm_rank == 0:
            if self.mask_file is None:
                raise AttributeError(
                    "You need to specify a mask file for a zoom simulation!"
                )
            if os.path.isfile(self.mask_file):
                raise OSError(
                    f"The specified mask file '{self.mask_file}' does not "
                    "exist!"
                )
        # Load data on rank 0 and then distribute to other MPI ranks
        if comm_rank == 0:
            print('\n------ Loading mask file ------')
            mask_data = {}
            with h5.File(self.mask_file, 'r') as f:
                
                # Centre of the high-res zoom in region
                centre = np.array(f['Coordinates'].attrs.get("geo_centre"))

                # Data specifying the mask for the high-res region
                mask_data['cell_coordinates'] = np.array(
                    f['Coordinates'][...], dtype='f8')
                mask_data['cell_size'] = (
                    f['Coordinates'].attrs.get("grid_cell_width"))

                # Also load the side length of the cube enclosing the mask,
                # and the volume of the target high-res region (at the
                # selection redshift).                
                mask_data['extent'] = (
                    f['Coordinates'].attrs.get("bounding_length"))
                mask_data['high_res_volume'] = (
                    f['Coordinates'].attrs.get("high_res_volume"))
            print(f"Finished loading data from {self.mask_file}.")
            print(f"   (bounding side = {mask_data['extent']:.3f} Mpc/h).")

        else:
            mask_data = None
            centre = None

        mask_data = comm.bcast(mask_data)
        self.centre = comm.bcast(centre)

        return mask_data

    def compute_fft_stats(self, max_boxsize, all_ntot):
        """ Work out what size of FFT grid we need for the IC gen. """
        if self.is_zoom:
            if self.is_slab:
                self.high_res_n_eff = self.n_particles
                self.high_res_L = self.box_size
            else:
                self.high_res_L = self.ic_region_buffer_frac * max_boxsize
                assert self.high_res_L < self.box_size, 'Zoom buffer region too big'
                self.high_res_n_eff \
                        = int(self.n_particles * (self.high_res_L**3./self.box_size**3.))
            print('--- HRgrid c=%s L_box=%.2f Mpc/h'%(self.centre, self.box_size))
            print('--- HRgrid L_grid=%.2f Mpc/h n_eff=%.2f**3 (x2=%.2f**3) FFT buff frac= %.2f'\
                    %(self.high_res_L, self.high_res_n_eff**(1/3.),
                        2.*self.high_res_n_eff**(1/3.), self.ic_region_buffer_frac))

            # How many multi grid FFT levels, this will update n_eff?
            if self.multigrid_ics:
                if self.high_res_L > self.box_size/2.:
                    print("--- Cannot use multigrid ICs, zoom region is > boxsize/2.")
                    self.multigrid_ics = 0
                else:
                    nlevels = 0
                    while self.box_size / (2.**(nlevels+1)) > self.high_res_L:
                        nlevels += 1
                    actual_high_res_L = self.box_size/(2.**nlevels)
                    assert actual_high_res_L > self.high_res_L, 'Incorrect actual high_res_L'
                    actual_high_res_n_eff = \
                            int(self.n_particles * (actual_high_res_L**3./self.box_size**3))
                        
                    print('--- HRgrid num multigrids=%i, lowest=%.2f Mpc/h n_eff: %.2f**3 (x2 %.2f**3)'\
                            %(nlevels,actual_high_res_L,actual_high_res_n_eff**(1/3.),
                                2*actual_high_res_n_eff**(1/3.)))
        else:
            self.high_res_n_eff = self.n_particles
            self.high_res_L = self.box_size

        # Minimum FFT grid that fits self.fft_times_fac times (defaut=2) the nyquist frequency.
        ndim_fft = self.ndim_fft_start
        N = (self.high_res_n_eff)**(1./3)
        while float(ndim_fft)/float(N) < self.fft_times_fac:
            ndim_fft *= 2
        print("--- Using ndim_fft = %d" % ndim_fft)

        # Determine number of cores to use based on memory requirements.
        # Number of cores must also be a factor of ndim_fft.
        print('--- Using nmaxpart= %i nmaxdisp= %i'%(self.nmaxpart, self.nmaxdisp))
        self.compute_ic_cores_from_mem(self.nmaxpart, self.nmaxdisp, ndim_fft, all_ntot,
                optimal=False)

        # What if we wanted the memory usage to be optimal?
        self.compute_optimal_ic_mem(ndim_fft, all_ntot)

    def compute_ic_cores_from_mem(self, nmaxpart, nmaxdisp, ndim_fft, all_ntot, optimal=False):
        ncores_ndisp = np.ceil(float((ndim_fft*ndim_fft * 2 * (ndim_fft/2+1))) / nmaxdisp)
        ncores_npart = np.ceil(float(all_ntot) / nmaxpart)
        ncores = max(ncores_ndisp, ncores_npart)
        while (ndim_fft % ncores) != 0:
            ncores += 1
  
        # If we're using one node, try to use as many of the cores as possible
        if ncores < self.ncores_node:
            ncores = self.ncores_node
            while (ndim_fft % ncores) != 0:
                ncores -= 1
        this_str = '[Optimal] ' if optimal else '' 
        print('--- %sUsing %i cores for IC gen (min %i for FFT and min %i for particles)'%\
                (this_str, ncores, ncores_ndisp, ncores_npart))
        if optimal == False: self.n_cores_ic_gen = ncores

    def compute_optimal_ic_mem(self, ndim_fft, all_ntot):
        """ This will compute the optimal memory to fit IC gen on cosma7. """

        bytes_per_particle = 66.         
        bytes_per_grid_cell = 20.

        total_memory = (bytes_per_particle*all_ntot) + (bytes_per_grid_cell*ndim_fft**3.)

        frac = (bytes_per_particle*all_ntot) / total_memory
        nmaxpart = (frac * self.mem_per_core) / bytes_per_particle

        frac = (bytes_per_grid_cell*ndim_fft**3.) / total_memory
        nmaxdisp = (frac * self.mem_per_core) / bytes_per_grid_cell
       
        total_cores = total_memory/self.mem_per_core

        print("--- [Optimal] nmaxpart= %i nmaxdisp= %i"%(nmaxpart, nmaxdisp))

        self.compute_ic_cores_from_mem(nmaxpart, nmaxdisp, ndim_fft, all_ntot, optimal=True)

    def make_particle_load(self, only_calc_ntot=False):
        """
        Main driver function to generate the particle load.

        For zoom-in simulations, it generates three classes of particles:
        (i)   those at the target resolution within "Zone I", the volume
              specified by the pre-defined mask (and generally slightly beyond
              it).
        (ii)  medium resolution particles within "Zone II", the remaining
              volume within a cubic box. These particles have a range of
              masses depending on how far away they are from Zone I. 
        (iii) low resolution particles within "Zone III", the remaining
              volume of the simulation box. Particles again have a range of
              masses depending on their distance from Zone II.

        For uniform-volume simulations, Zone I always covers the entire box.

        Parameters
        ----------
        only_calc_ntot : bool, optional
            Switch to skip the actual particle generation (default: False).

        Returns
        -------
        parts : dict
            The properties of the generated particles. It has two keys:
            - pos : ndarray(float) [N_part, 3]
                The x, y, and z coordinates of the particles generated on the
                local MPI rank.
            - m : ndarray(float) [N_part]
                The masses (in units of the total box mass) of each particle. 

        """
        if comm_rank == 0:
            print('\n-------------------------')
            print('Generating particle load.')
            print('-------------------------\n')

        # ================================================================
        #                       Battle plan: 
        # ----------------------------------------------------------------
        # 1.) Preparation (compute total box mass, only for log messages)"
        # 2.) Set up gcell grid and generate local gcells.
        # 3.) Generate local particles within gcell grid
        # 4.) Generate low-resolution particles (outside gcell cube)
        # ================================================================

        # ------- Act I: Setup (may move this out of here) -----------------

        # Get particle masses and softening lengths of high-res particles.
        self.compute_masses()

        # **TODO** this is not actually needed here, move somewhere else!
        self.softenings = self.compute_softening(verbose=True)

        # --------------------------------------------------------------------
        # --- Act II: Preparation (find structure and number of particles) ---
        # -------------------------------------------------------------------- 

        # Set up the gcube and generate local gcells
        self.gcube = self.set_up_gcube()

        # Prepare uniform grid cell structure within gcube
        # (sets up self.gcell_info)
        gcells = self.generate_gcells()

        # Prepare the cubic shell structure filling the outer box (Zone III)
        # (sets up self.scube dict)
        self.prepare_zone3_particles()

        if comm_rank == 0:
            print_particle_load_info()

        # If this is a "dry run" and we only want the particle number, quit.
        if only_calc_ntot:
            return

        # -------------------------------------------------------------------
        # ------ Act III: Creation (generate and verify particles) ----------
        # -------------------------------------------------------------------

        # Initiate particle arrays.
        pos = np.zeros((self.nparts['tot_local'], 3), dtype='f8') - 1e30
        masses = np.zeros(self.nparts['tot_local'], dtype='f8') - 1
        self.parts = {'pos': pos, 'm': masses}

        # Magic, part I: populate local gcells with particles (Zone I/II)
        self.generate_gcube_particles(gcells, self.parts)
         
        # Magic, part II: populate outer region with particles (Zone III)
        self.generate_zone3_particles(self.parts)

        # -------------------------------------------------------------------
        # --- Act IV: Transformation (shift coordinate system to target) ----
        # -------------------------------------------------------------------

        # Make sure that the particles are sensible before shifting
        self.verify_particles(self.parts)

        # Move particle load to final position in the simulation box
        self.shift_particles_to_target_position()


    def set_up_gcube(self):
        """
        Set up the frame of the gcube.

        This delineates the region filled with high(er) resolution particles,
        and is centred on the origin (same as the input mask).

        Parameters
        ----------
        None (this may change).

        Returns
        -------
        gcube : dict
            Dict specifying the structure of the gcube with the following keys:
            - ...

        """
        gcube = {}

        # Find the closest number of gcells that fill the box along one
        # side length. Recall that `self.n_particles` is already guaranteed
        # to be (close to) an integer multiple of `self.glass_num`, so the
        # rounding here is only to avoid numerical rounding issues.
        n_base = int(np.rint((self.num_particles / self.glass_num)**(1/3)))
        if (n_gcells**3 * self.glass_num != self.n_particles):
            raise ValueError(
                f"Want to use {n_base} cells per dimension, but this would "
                f"give {n_base**3 * self.glass_num} particles, instead of "
                f"the target number of {self.num_particles} particles."
            )

        # Side length of one gcell [h^-1 Mpc]
        gcube['cell_size'] = self.box_size / n_base

        if self.is_zoom:
            # In this case, we have fewer gcells (in general), but still
            # use the same size.

            # **TODO**: add option to directly pass in mask dict to class
            self.mask_data = self.load_mask_file()  # Also reads mask centre.

            n_gcells = int(np.ceil(mask_data['extent'] / gcube['cell_size']))
            gcube['n_cells'] = n_gcells + self.num_buffer_gcells * 2

            # Make sure that we don't assign more than the total number of
            # cells to the high-resolution region
            if gcube['n_cells'] > n_base:
                raise ValueError(
                    f"Cannot assign {gcube['n_cells']} cells per dimension to "
                    f"gcube if there are only {n_base} cells per "
                    "dimension in total!"
                )
        else:
            # Simple: whole box filled by high-resolution glass cells.
            gcube['n_gcells'] = n_base

        # Check that we didn't generate more high-res cells than we can handle
        gcube['num_cells'] = gcube['n_cells']**3
        if gcube['num_cells'] > (2**32) / 2:
            raise Exception(
                f"Total number of gcells ({gcube['num_cells']}) "
                f"is too large, can only have up to {(2**32)/2}."
            )

        # Compute the physical size of the gcube. Note that, for a
        # zoom simulation, this is generally not the same as the size of the
        # mask box, because of the applied buffer and quantization.
        gcube['sidelength'] = gcell_size * n_gcells
        gcube['volume'] = gcube_length**3
 
        if comm_rank == 0:
            print(
                f"Using a gcube with side length {gcube['sidelength']:.4f}, "
                f"Mpc/h, corresponding to a volume of {gcube['volume']:.4f} "
                f"(Mpc/h)^3 ({gcube['volume']/self.box_size**3 * 100:.3f} "
                f"per cent of the total simulation volume or "
                f"{gcube['volume']/mask_data['extent']**3 * 100:.3f} per cent "
                f"of the mask bounding cube).\n"
                f"Using {gcube['n_cells']} gcells of size "
                f"{gcube['cell_size']} Mpc/h per dimension "
                f"({gcube['num_cells']} gcells in total)."
            )

        return gcube

    def load_glass_files(self):
        """Load all required glass files into a list of coordinate arrays."""
        glass = {}
        if self.zone2_type == 'glass':
            # If we fill Zone II with glass cells, we typically need several
            # different sizes of them, for the different resolution levels.
            for num_glass in self.gcell_info['num_particles_per_cell']:
                if num_glass not in glass:
                    glass[num_glass] = self.load_glass_file(num_glass)
        elif self.zone1_type == 'glass':
            # If only Zone I is glass, we only need one size
            glass[self.glass_num] = self.load_glass_file(self.glass_num)

        return glass

    def _verify_gcube_region(self, parts, nparts_created, gvolume):
        """
        Perform consistency checks and print info about high-res region.
        This is a subfunction of `populate_gcells()`.
        """
        # Make sure that the coordinates are in the expected range
        if np.max(np.abs(parts['pos'][:nparts_created])) > 0.5:
            raise ValueError("Invalid Zone I/II coordinate values!")

        # Make sure that we have allocated the right mass (fraction) to
        # the cubic high resolution region.
        tot_hr_mass = comm.allreduce(np.sum(parts['masses'][:nparts_created]))
        ideal_hr_mass = gvolume / self.box_size**3
        if np.abs(tot_hr_mass - ideal_hr_mass) > 1e-6:
            raise ValueError(
                f"Particles in the cubic high-res region have a combined "
                f"mass (fraction) of {tot_hr_mass} instead of the expected "
                f"{ideal_hr_mass}!"
            )

        # Find and print the centre of mass of the Zone I/II particles
        # (N.B.: slicing creates views, not new arrays --> no memory overhead)
        com = centre_of_mass_mpi(
            parts['pos'][: nparts_gcube], parts['masses'][:, nparts_gcube])

        if comm_rank == 0:
            print(f"Centre of mass for high-res grid particles:\n"
                  f"[{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}] Mpc/h."
            )

    def generate_zone3_particles(self, parts):
        """
        Generate (very) low resolution boundary particles outside the gcube.
    
        Parameters
        ----------
        parts : dict
            Dict containing the particle coordinate and mass arrays

        Returns
        -------
        None

        """
        npart_local = self.nparts['zone3_local']
        npart_all = self.nparts['zone3_all']
        offset_zone3 = self.nparts['zone1_all'] + self.nparts['zone2_all']

        # No zone III ==> no problem. This may also be the case for zooms.
        if npart_all == 0:
            return

        if comm_rank == 0:
            print(f"\n---- Generating outer low-res particles (Zone III) ----")
            print(f"   (Num_global = {npart_all})")

        if npart_local > 0:
            if self.is_slab:
                cy.get_layered_particles_slab(
                    min_boxsize, self.box_size,
                    self.nq_info['starting_nq'], self.nq_info['nlev_slab'],
                    self.nq_info['dv_slab'], comm_rank, comm_size, npart_all_zone3, n_tot_hi,
                    coords_x, coords_y, coords_z, masses, self.nq_info['nq_reduce'],
                    self.nq_info['extra']
                )
            else:
                cy.layered_particles(
                    side, self.nq_info['nq'], comm_rank,
                    comm_size, npart_all_zone3, n_tot_hi, self.nq_info['extra'],
                    self.nq_info['total_volume'], coords_x, coords_y, coords_z,
                    masses
                )

            m_max = np.max(parts['masses'][offset_zone3 : ])
            m_min = np.min(parts['masses'][offset_zone3 : ])

            if self.verbose:
                print(
                    f"[{comm_rank}]: "
                    f"Generated {npart_local} (= {npart_local**(1/3):.2f}^3) "
                    f"particles.\nMin log mass (fraction): "
                    f"{np.log10(m_min):.2f}, max: {np.log10(m_max):.2f}."
                )

        else:
            # No zone III particles generated on this rank
            m_max = -1      # Min possible mass is 0
            m_min = 10      # Max possible mass is 1

        # Find and print cross-MPI range of low-res particle masses
        m_max = comm.allreduce(m_max, op=MPI.MAX)
        m_min = comm.allreduce(m_min, op=MPI.MIN)
        if comm_rank == 0:
            print(
                f"Total of {num_lr} particles in low-res region "
                f"(={num_lr**(1/3)}^3).\n"
                f"Minimum mass fraction = 10^{np.log10(m_min):.2f} "
                f"(m_min = {m_min * self.total_box_mass:.3e} M_Sun),\n"
                f"Maximum mass fraction = 10^{np.log10(m_max):.2f} "
                f"(m_max = {m_max * self.total_box_mass:.3e} M_Sun)."
            )

        # Safety check on coordinates
        if npart_local > 0:
            # Don't abs whole array to avoid memory overhead
            if (np.max(parts['coords'][:, offset_zone3:]) > 0.5 or
                np.min(parts['coords'][:, offset_zone3:]) < 0.5):
                raise ValueError(
                    f"Zone III particles outside allowed range (-0.5 -> 0.5)! "
                    f"\nMaximum absolute coordinate is "
                    f"{np.max(np.abs(parts['coords'][:, offset_zone3:]))}."
                )

    def verify_particles(self, parts):
        """Perform safety checks on all generated particles."""

        npart_local = self.nparts['tot_local']
        npart_global = self.nparts['tot_all']

        # Safety check on coordinates and masses
        if npart_local > 0:
            # Don't abs whole array to avoid memory overhead
            if np.min(parts['coords']) < 0.5 or np.max(parts['coords']) > 0.5:
                raise ValueError(
                    f"Zone III particles outside allowed range (-0.5 -> 0.5)! "
                    f"\nMinimum coordinate is {np.min(parts['coords'])}, "
                    f"maximum is {np.min(parts['coords'])}."
                )
            if np.min(parts['masses']) < 0 or np.max(parts['masses']) > 1:
                raise ValueError(
                    f"Masses should be in the range 0 --> 1, but we have "
                    f"min={np.min(parts['masses'])}, "
                    f"max={np.max(parts['masses'])}."
                )

        # Check that the total masses add up to 1.
        total_mass_fraction = comm.allreduce(np.sum(parts['masses']))
        mass_error_fraction = 1 - total_mass_fraction
        if np.abs(mass_error_fraction) > 1e-5:
            raise ValueError(
                f"Unacceptably large error in total mass fractions: "
                f"Expected 1.0, got {total_mass_fraction:.4e} (error: "
                f"{mass_error_fraction}."
            )
        if np.abs(mass_error_fraction > 1e-6) and comm_rank == 0:
            print(
                f"**********\n   WARNING!!! \n***********\n"
                f"Error in total mass fraction is {mass_error_fraction}."
            )

        # Find the (cross-MPI) centre of mass, should be near origin.
        com = centre_of_mass_mpi(parts['coords'], parts['masses'])
        if comm_rank == 0:
            print(f"Centre of mass for all particles (in units of the box "
                  f"size): [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}].")
            if (np.max(np.abs(com)) > 0.1):
                print(
                    f"**********\n   WARNING!!! \n***********\n"
                    f"Centre of mass is offset suspiciously far from the "
                    f"geometric box centre!"
                )

        # Announce success if we got here.
        if comm_rank == 0:
            print(
                f"Done creating and verifying {self.nparts['tot_all']} "
                f"(= {self.nparts['tot_all']**(1/3):.3f}^3) particles."
            )

    def shift_particles_to_target_position(self):
        """
        Move the centre of the high-res region to the desired point.

        This also applies periodic wrapping. On return, particles are in
        their final positions as required by IC-Gen (in range 0 --> 1).

        Returns
        -------
        None

        """
        # Shift particles to the specified centre (note that self.centre_phys
        # is already in the range 0 --> box size, so that's all we need)...
        if comm_rank == 0:
            print(f"Lagrangian centre of high-res region is at "
                  f"{self.centre_phys[0]:.3f} / {self.centre_phys[1]:.3f} / "
                  f"{self.centre_phys[2]:.3f} h^-1 Mpc."
            )
        cen = rescale(self.centre_phys, [0., self.box_size], [0., 1.])
        self.parts['coords'] += cen

        # ... and then apply periodic wrapping (simple: range is 0 --> 1).
        self.parts['coords'] %= 1.0

    def save_param_files(self):
        """
        Create IC-GEN and GADGET/SWIFT parameter files and submit scripts.

        ** THIS IS NOT ACTUALLY DOING ANYTHING RIGHT NOW. **
        """
        # Compute mass cut offs between particle types.
        hr_cut = 0.0
        lr_cut = 0.0

        if self.is_zoom:
            i_z = 1

            if self.n_species >= 2:
                hr_cut = np.log10(self.glass_particle_mass) + 0.01
                print('log10 mass cut from parttype 1 --> 2 = %.2f' % (hr_cut))
            if self.n_species == 3:
                lr_cut = np.log10(self.max_grid_mass) + 0.01
                print('log10 mass cut from parttype 2 --> 3 = %.2f' % (lr_cut))
        else:
            i_z = 0

        # Get softenings.
        eps = self.compute_softening()

        # Build a dict with all parameters that could possibly be required
        # for the parameter and submit files.
        param_dict = dict(
            hr_cut='%.3f'%hr_cut,
            lr_cut='%.3f'%lr_cut,
            is_zoom=i_z,
            f_name=self.f_name,
            n_species=self.n_species,
            ic_dir=self.ic_dir,
            box_size='%.8f'%self.box_size,
            starting_z='%.8f'%self.starting_z,
            finishing_z='%.8f'%self.finishing_z,
            n_particles=self.n_particles,
            coords_x='%.8f'%self.centre[0],
            coords_y='%.8f'%self.centre[1],
            coords_z='%.8f'%self.centre[2],
            high_res_L='%.8f'%self.high_res_L,
            high_res_n_eff=self.high_res_n_eff,
            panphasian_descriptor=self.panphasian_descriptor,
            constraint_phase_descriptor="%dummy",
            constraint_phase_descriptor_path="%dummy",
            constraint_phase_descriptor_levels="%dummy",
            constraint_phase_descriptor2="%dummy",
            constraint_phase_descriptor_path2="%dummy",
            constraint_phase_descriptor_levels2="%dummy",
            ndim_fft_start=self.ndim_fft_start,
            Omega0='%.8f'%self.Omega0,
            OmegaLambda='%.8f'%self.OmegaLambda,
            OmegaBaryon='%.8f'%self.OmegaBaryon,
            HubbleParam='%.8f'%self.HubbleParam,
            Sigma8='%.8f'%self.Sigma8,
            is_slab=self.is_slab,
            use_ph_ids=self.use_ph_ids,
            multigrid_ics=self.multigrid_ics,
            linear_ps=self.linear_ps,
            nbit=self.nbit,
            fft_times_fac=self.fft_times_fac,
            swift_ic_dir_loc=self.swift_ic_dir_loc,
            eps_dm_h='d%.8f'%eps_dm_h,
            eps_baryon_h='%.8f'%eps_baryon_h,
            softening_ratio_background=self.softening_ratio_background,
            eps_dm_physical_h='%.8f'%eps_dm_physical_h,
            eps_baryon_physical_h='%.8f'%eps_baryon_physical_h,
            template_set=self.template_set,
            gas_particle_mass=self.gas_particle_mass,
            swift_dir=self.swift_dir,
            n_nodes_swift='%i'%self.n_nodes_swift,
            num_hours_swift=self.num_hours_swift,
            swift_exec_location=self.swift_exec_location,
            num_hours_ic_gen=self.num_hours_ic_gen,
            n_cores_ic_gen='%i'%self.n_cores_ic_gen,
            eps_dm='%.8f'%(eps_dm_h/self.HubbleParam),
            eps_baryon='%.8f'%(eps_baryon_h/self.HubbleParam),
            eps_dm_physical='%.8f'%(eps_dm_physical_h/self.HubbleParam),
            eps_baryon_physical='%.8f'%(eps_baryon_physical_h/self.HubbleParam))

        # Make ICs param file.
        if self.make_ic_param_files:
            make_param_file_ics(param_dict)
            make_submit_file_ics(param_dict)

            print('\n------ Saving ------')
            print('Saved ics param and submit file.')

        if 'gadget4' in self.sim_types:
            raise Exception("Think about DMO for gadget4")
            # Make GADGET4 param file.
            # make_param_file_gadget4(self.gadget4_dir, self.f_name, self.box_size,
            #        self.starting_z, self.finishing_z, self.Omega0, self.OmegaLambda,
            #        self.OmegaBaryon, self.HubbleParam, s_high)

            ## Makde GADGET4 submit file.
            # make_submit_file_gadget4(self.gadget4_dir, self.f_name)

        if 'gadget' in self.sim_types:
            raise Exception("Think about DMO for gadget")
            # Make GADGET param file.
            # make_param_file_gadget(self.gadget_dir, self.f_name, self.box_size,
            #    s_high, s_low, s_low_low, self.Omega0, self.OmegaLambda,
            #    self.OmegaBaryon, self.HubbleParam, self.dm_only,
            #    self.starting_z, self.finishing_z, self.high_output)
            # print 'Saved gadget param file.'

        if self.make_swift_param_files:
            # Make swift param file (remember no h's for swift).
            make_param_file_swift(param_dict)
            make_submit_file_swift(param_dict)
            print('Saved swift param and submit file.')

    def save_submit_files(self, max_boxsize):
        """
        Generate submit files.

        **TODO** Complete or remove.
        """
        if 'gadget' in self.sim_types:
            raise Exception(
                "Creation of GADGET submit files is not yet implemented.")

    def compute_softening(self, verbose=False) -> dict:
        """
        Compute softening lengths, in units of Mpc/h.

        Returns
        -------
        eps : dict
            A dictionary with four keys: 'dm' and 'baryon' contain the
            co-moving softening lengths for DM and baryons (the latter is 0
            for DM-only simulations). 'dm_proper' and 'baryon_proper' contain
            the corresponding maximal proper softening lengths.
        """
        if self.dm_only:
            comoving_ratio = 1 / 20.  # = 0.050
            proper_ratio = 1 / 45.  # = 0.022
        else:
            comoving_ratio = 1 / 20.  # = 0.050
            proper_ratio = 1 / 45.  # = 0.022

        # Compute mean inter-particle separation, in Mpc/h (recall that
        # self.box_size is already in these units).
        n_per_dimension = self.n_particles ** (1 / 3.)
        mean_interparticle_separation = self.box_size / n_per_dimension

        # Softening lengths for DM
        eps_dm = mean_interparticle_separation * comoving_ratio
        eps_dm_proper = mean_interparticle_separation * proper_ratio

        # Softening lengths for baryons
        if self.dm_only:
            eps_baryon = 0.0
            eps_baryon_proper = 0.0
        else:
            # Adjust DM softening lengths according to baryon fraction
            fac = ((self.Omega0 - self.OmegaBaryon) / self.OmegaBaryon)**(1./3)
            eps_baryon = eps_dm / fac
            eps_baryon_proper = eps_dm_proper / fac

        if comm_rank == 0 and verbose:
            h = self.HubbleParam
            if not self.dm_only:
                print(f"Comoving softenings: DM={eps_dm:.6f}, "
                      f"baryons={eps_baryon:.6f} Mpc/h")
                print(f"Max proper softenings: DM={eps_dm_proper:.6f}, "
                      f"baryons={eps_baryon_proper:.6f} Mpc/h")
            print(f"Comoving softenings: DM={eps_dm / h:.6f} Mpc, "
                  f"baryond={eps_baryon / h:.6f} Mpc")
            print(f"Max proper softenings: DM={eps_dm_proper / h:.6f} Mpc, "
                  f"baryons={eps_baryon_proper / h:.6f} Mpc")

        eps = {'dm': eps_dm, 'baryon': eps_baryon,
               'dm_proper': eps_dm_proper, 'baryon_proper': eps_baryon_proper}
        return eps


    def generate_uniform_grid(self, n=None, num=None, centre=False):
        """
        Generate a uniform cubic grid of `n_particles` particles.

        The particles are arranged within a cube of side length 1. Their
        coordinates are returned as a [N_part, 3] array. 

        Parameters
        ----------
        n : int, optional
            The number of particles per dimension to place into the grid.
            If it is not specified, num must be.
        num : int, optional
            The total number of particles to place into the grid. It need
            not be a cube number; if it is not, the closest cube is used.
            If it is not specified, n must be.
        centre : bool, optional
            Place the coordinate origin at the centre of the cube, rather than
            at its lower vertex (default: False).

        Returns
        -------
        coords : ndarray(float) [Num_grid, 3]
            The particle coordinates.

        """
        if n is None:
            n = int(np.rint(n_particles**(1/3)))

        pos_1d = np.linspace(0, 1, num=n, endpoint=False)
        pos_1d += (pos_1d[1] - pos_1d[0])/2
        pos_3d = np.meshgrid(*[pos_1d]*3)

        # pos_3d is arranged as a 3D grid, so we need to flatten the three
        # coordinate arrays
        coords = np.zeros((n_particles, 3), dtype='f8')
        for idim in range(3):
            coords[:, idim] = pos_3d[idim].flatten()

        # Safety check to make sure that the coordinates are valid.
        assert np.all(coords >= 0.0) and np.all(coords <= 1.0), (
            'Inconsistency generating a uniform grid')

        # If requested, shift coordinates to cube centre
        if centre:
            coords -= 0.5

        return coords

    def print_particle_load_info(self):
        """Print information about to-be-generated particles (rank 0 only)."""

        # ** TODO ** Come back to this.

        if comm_rank != 0:
            return

        if self.is_zoom:
            if self.mask_file is not None:
                n_particles_target = (self.n_particles / self.box_size ** 3.) \
                                     * self.mask_data['high_res_volume']
            print('--- Total number of glass particles %i (%.2f cubed, %.2f percent)' % \
                  (self.n_tot_glass_part, self.n_tot_glass_part ** (1 / 3.), np.true_divide(
                      self.n_tot_glass_part, all_ntot)))
            print('--- Total number of grid particles %i (%.2f cubed, %.2f percent)' % \
                  (self.n_tot_grid_part, self.n_tot_grid_part ** (1 / 3.), np.true_divide(
                      self.n_tot_grid_part, all_ntot)))
            print('--- Total number of outer particles %i (%.2f cubed, %.2f percent)' % \
                  (all_ntot_lo, all_ntot_lo ** (1 / 3.), np.true_divide(
                      all_ntot_lo, all_ntot)))
            if self.mask_file is not None:
                print('--- Target number of ps %i (%.2f cubed), made %.2f times as many.' % \
                      (n_particles_target, n_particles_target ** (1 / 3.),
                       np.true_divide(all_ntot, n_particles_target)))

        self.compute_fft_stats(max_boxsize, all_ntot)
        print('--- Total number of particles %i (%.2f cubed)' % \
              (all_ntot, all_ntot ** (1 / 3.)))
        print('--- Total memory per rank HR grid=%.6f Gb, total of particles=%.6f Gb' % \
              (self.size_of_HR_grid_arrays / 1024. / 1024. / 1024.,
               (4 * all_ntot * 8. / 1024. / 1024. / 1024.)))
        print('--- Num ranks needed for less than <max_particles_per_ic_file> = %.2f' % \
              (np.true_divide(all_ntot, self.max_particles_per_ic_file)))

    def save_particle_load(self, randomize=False):
        """
        Save the generated particle load in HDF5 and/or Fortran format.

        This is an MPI-collective function, because it internally
        redistributes the particle load between MPI ranks so that all files
        are similar in size.    

        Parameters
        ----------
        randomize : bool
            Shuffle particle positions on each rank before writing (and before
            possible re-partitioning).

        Returns
        -------
        None

        """
        # If we don't save in either format, we can stop right here.
        if not self.save_as_hdf5 + self.save_as_fortran:
            return

        # Extract parameters from class attributes
        num_part_local = self.nparts['tot_local']
        num_part_global = self.nparts['tot_all']

        # Randomise arrays, if desired
        if self.randomize:
            if comm_rank == 0:
                print('Randomizing arrays...')
            idx = np.random.permutation(num_part_local)
            parts['pos'] = parts['pos'][idx, :]
            parts['masses'] = parts['masses'][idx]

        # Load balance across MPI ranks.
        self.repartition_particles()

        # Save particle load as a collection of HDF5 and/or Fortran files
        save_dir = (f"{self.ic_dir}/ic_gen_submit_files/{self.f_name}/"
                    f"particle_load")
        save_dir_hdf = save_dir + '/hdf5'
        save_dir_bin = save_dir + '/fbinary'

        if comm_rank == 0:
            if not os.path.exists(save_dir_hdf) and self.save_as_hdf5:
                os.makedirs(save_dir_hdf)
            if not os.path.exists(save_dir_bin) and self.save_as_fortran:
                os.makedirs(save_dir_bin)

        # Make sure to save no more than max_save files at a time.
        max_save = 50
        n_batch = int(np.ceil(comm_size / max_save))
        for ibatch in range(n_batch):
            if comm_rank % max_save == n_batch:
                if self.save_as_df5:
                    hdf5_loc = f"{save_dir_hdf5}/PL.{comm_rank}.hdf5" 
                    self.save_local_particles_as_hdf5(hdf5_loc)

                # Save as Fortran binary.
                if self.save_as_fortran:
                    fortran_loc = f"{save_dir_bin}/PL.{comm_rank}"
                    self.save_local_particles_as_binary(fortran_loc)

                if self.verbose:
                    print(
                        f"[Rank {comm_rank}] Finished saving {n_part} local "
                         "particles.")

            # Make all ranks wait so that we don't have too many writing
            # at the same time.
            comm.barrier()

    def repartition_particles(self):
        """Re-distribute particles across MPI ranks to achieve equal load."""
        if comm_size == 0:
            return

        # Find min/max number of particles per rank.
        num_part_min = comm.allreduce(num_part_local, op=MPI.MIN)
        num_part_max = comm.allreduce(num_part_local, op=MPI.MAX)

        # ** TODO **: move this to separate function prior to particle
        # generation. That way, we can immediately build the coordinate
        # arrays with the final size (if > locally generated size).
        num_part_desired = np.zeros(comm_size, dtype=int)
        num_part_desired = (num_part_global // comm_size).astype(int)
        num_part_allocated = np.sum(num_part_desired)
        num_part_desired[-1] += (num_part_global - num_part_allocated)

        if comm_rank == 0:
            n_per_rank = num_part_desired[0]**(1/3.)
            print(
                f"Load balancing {num_part_global} particles across "
                f"{comm_size} ranks ({n_per_rank:.2f}^3 per rank)...\n"
                f"Current load ranges from "
                f"{num_part_min} to {num_part_max} particles."
            )

            # Change this in future by allowing one rank to write >1 files
            if n_per_rank > self.max_particles_per_ic_file**(1/3.):
                print(f"***WARNING*** Re-partitioning will lead to more "
                      f"than {self.max_particles_per_ic_file}^3 particles "
                      f"per IC file!"
                )

        self.parts['masses'] = repartition(
            self.parts['masses'], num_part_desired,
            comm, comm_rank, comm_size
        ) 
        num_part_new = len(parts['masses'])

        # Because we repartition the coordinates individually, we first
        # need to expand the array if needed
        if num_part_new > self.parts['coords'].shape[0]:
            self.parts['coords'] = np.resize(
                self.parts['coords'], (num_part_new, 3))
            self.parts['coords'][self.nparts['tot_local']: , : ] = -1

        for idim in range(3):
            self.parts['coords'][: num_part_new, idim] = repartition(
            self.parts['coords'][:, idim], num_part_desired,
            comm, comm_rank, comm_size
        )
        if num_part_new < self.nparts['tot_local']:
            self.parts['coords'] = self.parts['coords'][:num_part_new, :]

        self.nparts['tot_local'] = num_part_new

        if comm_rank == 0:
            print('Done with load balancing.')

    def save_local_particles_as_hdf5(self, save_loc):
        """Write local particle load to HDF5 file `save_loc`"""
        n_part = self.n_part_local
        n_part_tot = self.n_part_total
 
        with h5.File(save_loc, 'w') as f:
            g = f.create_group('PartType1')
            g.create_dataset('Coordinates', (n_part, 3), dtype='f8')
            g['Coordinates'][:,0] = self.coords_x
            g['Coordinates'][:,1] = self.coords_y
            g['Coordinates'][:,2] = self.coords_z
            g.create_dataset('Masses', data=self.masses)
            g.create_dataset('ParticleIDs', data=np.arange(n_part))

            g = f.create_group('Header')
            g.attrs.create('nlist', n_part)
            g.attrs.create('itot', n_part_tot)
            g.attrs.create('nj', comm_rank)
            g.attrs.create('nfile', comm_size)
            g.attrs.create('coords', self.centre / self.box_size)
            g.attrs.create('radius', self.radius / self.box_size)
            g.attrs.create('cell_length', self.cell_length/self.box_size)
            g.attrs.create('Redshift', 1000)
            g.attrs.create('Time', 0)
            g.attrs.create('NumPart_ThisFile', [0, n_part, 0, 0, 0])
            g.attrs.create('NumPart_Total', [0, n_part_tot, 0, 0, 0])
            g.attrs.create('NumPart_TotalHighWord', [0, 0, 0, 0, 0])
            g.attrs.create('NumFilesPerSnapshot', comm_size)
            g.attrs.create('ThisFile', comm_rank)

    def save_local_particles_as_binary(self, save_loc):
        """Write local particle load to Fortran binary file `save_loc`"""
        f = FortranFile(save_loc, mode="w")

        # Write first 4+8+4+4+4 = 24 bytes
        f.write_record(
            np.int32(self.n_part_local),
            np.int64(self.n_part_total),
            np.int32(comm_rank),
            np.int32(comm_size),
            np.int32(0),
            
            # Now we pad the header with 6 zeros to make the header length
            # 48 bytes in total
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
        )

        # Write actual data
        f.write_record(self.coords_x.astype(np.float64))
        f.write_record(self.coords_y.astype(np.float64))
        f.write_record(self.coords_z.astype(np.float64))
        f.write_record(self.masses.astype("float32"))
        f.close()

# ---------------------------------------------------------------------------

def mpi_combine_arrays(a, send_to_all=True, unicate=False, root=0):
    """
    Combine (partial) arrays held by each MPI rank.
 
    The segments held on each rank are combined in rank order (unless
    `unicate` is True).

    Parameters
    ----------
    a : ndarray
        The local array to combine across ranks. It must be one-dimensional.
    send_to_all : bool, optional
        Switch to communicate the concatenated array back to each rank
        (default: True).
    unicate : bool, optional
        Remove duplicate entries from the concatenated array (default: False).
    root : int, optional
        The rank on which to gather the segments (default: 0). If send_to_all
        is True, this only affects the internal operation of the function.

    Returns
    -------
    a_full : ndarray
        The concatenated array. If send_to_all is False, this is None on all
        ranks except `root`.

    Examples
    --------
    >>> a = np.zeros(3) + comm_rank
    >>> a_all = mpi_combine_arrays(a)
    array([0., 0., 0., 1., 1., 1.])   # with 2 MPI ranks

    """
    n_elem_by_rank = comm.allgather(a.shape[0])
    offset_by_rank = np.concatenate(([0], np.cumsum(n_elem_by_rank)))

    n_tot = offset_by_rank[-1]
    a_all = np.zeros(n_tot, dtype=a.dtype) if comm_rank == root else None

    try:
        mpi_type = MPI_TYPE_MAP[a.dtype.name]
    except KeyError:
        print(f"Could not translate array type {a.dtype.name} to MPI.\n"
              "Converting array to float64...")
        a = a.astype(float64)
        mpi_type = MPI_TYPE_MAP[float64]

    comm.Gatherv(
        sendbuf=a,
        recvbuf=[a_all, n_elem_by_rank, offset_by_rank, mpi_type],
        root=root)

    if comm_rank == 0:
        if unicate:
            a_all = np.unique(a_all)
    else:
        a_all = None

    if send_to_all:
        a_all = comm.bcast(a_all)

    return a_all


def find_next_cube(self, num):
    """Find the lowest number >=num that has a cube root."""
    return int(np.ceil(num**(1/3.))** 3.)


def rescale(self, x, old_range, new_range, in_place=False):
    """
    Rescale a coordinate, or an array thereof, from an old to a new range.
    
    Parameters
    ----------
    x : scalar or ndarray
        The coordinate(s) re-scale
    old_range : tuple(2)
        The original coordinate range (can be different from the range of
        x values).
    new_range : tuple(2)
        The target coordinates corresponding to the coordinates specified
        by old_range.
    in_place : bool, optional
        Switch to modify the input coordinates in place instead of returning
        a new variable (default: False).

    Returns
    -------
    x_new : float or ndarray(float)
        The re-scaled coordinates.

    """
    old_span = old_range[1] - old_range[0]
    new_span = new_range[1] - new_range[0]
    scale_factor = new_span / old_span

    if in_place:
        x -= old_range[0]
        x *= scale_factor
        x += new_range[0]
    else:
        return scale_factor * (x - old_range[0]) + new_range[0]


MPI_TYPE_MAP = {
    'int8': MPI.CHAR,
    'int16': MPI.SHORT,
    'int32': MPI.INT,
    'int64': MPI.LONG,
    'int128': MPI.LONG_LONG,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE,
    'bool': MPI.BOOL,
}


def cell_centre_from_index(index, n_cells):
    """
    Compute the central coordinates of cubic cells in a box from their indices.

    Indices are assumed to increase continuously along the axes, with cell
    ix, iy, iz (in cells from the lower vertex) having index
    ix + iy*nx + iz*nx *ny (where nx and ny are the total number of cells
    in the x and y direction, respectively).

    Parameters
    ----------
    index : int or ndarray(int)
        The index or indices of to-be-evaluated cells. Values must be in the
        range [0, prod(n_cells)[.
    n_cells : array, tuple, or list (int) [3]
        The total number of cells along each axis.

    Returns
    -------
    pos : float or ndarray(float)
        The central coordinates of the input cells, in units of one cell
        side length and with the origin at the centre of the box.

    Examples
    --------
    >>> cell_centre_from_index(14, [2, 3, 5])
    array([-1. , -0.5, -0.5])

    >>> idx = [4, 3, 7]
    >>> cell_centre_from_index(idx, [2, 3, 5])
    array([[-0.5,  1. , -2. ],
           [ 0.5,  0. , -2. ],
           [ 0.5, -1. , -1. ]])
    """
    if np.max(index) >= np.prod(n_cells) or np.min(index) < 0:
        raise ValueError("Invalid index in input to cell_centre_from_index!")

    indices_3d = np.unravel_index(index, n_cells, 'F')
    pos = np.zeros((len(index), 3))
    for idim in range(3):
        pos[:, idim] = indices_3d[idim]

    cen = np.array(n_cells) / 2
    return pos - cen + 0.5


def cell_index_from_centre(pos, n_cells):
    """
    Compute the scalar index of cubic cells in a box from their coordinates.

    Indices are assumed to increase continuously along the axes, with cell
    ix, iy, iz (in cells from the lower vertex) having index
    ix + iy*nx + iz*nx *ny (where nx and ny are the total number of cells
    in the x and y direction, respectively).

    Parameters
    ----------
    pos : ndarray(float) [N_cells, 3]
        The coordinates of an arbitrary point within each input cell, in units
        of one cell side length and with the origin at the centre of the box.
    n_cells : array, tuple, or list (int) [3]
        The total number of cells along each axis.

    Returns
    -------
    index : ndarray(int)
        The scalar index of each input cell.

    Examples
    --------
    pos = np.array([[-0.6, 1.1, -2.0], [0.5, 0.0, -2.2]])
    >>> cell_index_from_centre(pos, [2, 3, 5])
    array([4, 3])

    """
    cen = np.array(n_cells) / 2
    indices_3d = np.floor(pos + cen).astype(int)
    
    indices_3d_list = [indices_3d[:, i] for i in range(3)]

    # Internally raises an error if indices are out of range
    return np.ravel_multi_index(indices_3d_list, n_cells, order='F')


def centre_of_mass_mpi(coords, masses):
    """
    Compute the centre of mass for input particles, across MPI.

    The return value is only significant for rank 0.
    """
    # Form local CoM first, to avoid memory overheads...
    com_local = np.average(coords, weights=masses, axis=0)
    m_tot = np.sum(masses)

    # ... then form cross-MPI (weighted) average of all local CoMs.
    com_x = comm.reduce(com_local[0] * m_tot)
    com_y = comm.reduce(com_local[1] * m_tot)
    com_z = comm.reduce(com_local[2] * m_tot)
    m_tot = comm.reduce(m_tot)

    return np.array((com_x, com_y, com_z)) / m_tot


if __name__ == '__main__':
    only_calc_ntot = False
    if len(sys.argv) > 2:
        only_calc_ntot = True if int(sys.argv[2]) == 1 else False

    ParticleLoad(sys.argv[1], only_calc_ntot=only_calc_ntot)
