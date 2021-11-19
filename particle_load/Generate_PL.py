import os
import sys
import re
import yaml
import h5py as h5
import numpy as np
import subprocess
from scipy.spatial import distance
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from mpi4py import MPI
from ParallelFunctions import repartition
#import MakeGrid as cy
import MakeParamFile as mpf
from scipy.io import FortranFile
import time

import numpy as np
import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)
import auxiliary_tools as cy

from pdb import set_trace

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# ** TO DO **
# - Implement multiple-files-per-rank writing
# - Change internal units from h^-1 Mpc to Mpc, *only* add h^-1 for IC-Gen.
# - Tidy up output
# - Re-arrange 

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
    def __init__(self, param_file: str,
                 randomize: bool = False, only_calc_ntot: bool = False,
                 verbose: bool = False, save_data: bool = True,
                 create_param_files: bool = True,
                 create_submit_files: bool = True
    ) -> None:

        self.verbose = 1

        # Read and process the parameter file.
        self.read_param_file(param_file)
        self.sim_box = self.initialize_sim_box()

        self.compute_box_mass()
        self.get_target_resolution()

        # Generate particle load.
        self.nparts = {}
        self.scube = {}
        
        self.parts = self.make_particle_load(only_calc_ntot=only_calc_ntot)

        # Generate param and submit files, if desired (NOT YET IMPLEMENTED).
        if comm_rank == 0:
            self.create_param_and_submit_files(
                create_param=create_param_files,
                create_submit=create_submit_files
            )

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
            'sim_name',
            'is_zoom',
            'cosmology',
            'box_size',  # Change, do not require for zoom
            'zone1_gcell_load',
        ]
        for att in required_params:
            if att not in params:
                raise KeyError(f"Parameter {att} must be specified!")

        # Get the parameters for the specified cosmology
        self.cosmology_name = params['cosmology']
        self.cosmo = get_cosmology_params(params['cosmology'])

        # Extract parameters for particle load and subsequent codes
        self.config = self.get_config_params(params)
        self.extra_params = self.get_extra_params(params)        

    def get_config_params(self, params):
        """
        Get parameters that affect the particle load generation.

        See the template parameter file for a description of each parameter.
    
        Parameters
        ----------
        params : dict
            The parsed YAML input parameter file.

        Returns
        -------
        cparams : dict
            The full set of configuration parameters including defaults.
        """
        defdict = {
            # Basics
            'sim_name': None,
            'is_zoom': None,
            'box_size': None,
            'mask_file': None,
            'uniform_particle_number': None,
            'target_mass': None,

            # In-/output options
            'output_formats': "Fortran, HDF5",
            'output_dir': './ic_gen_data/',
            'glass_files_dir': './glass_files',
            'max_numpart_per_file': 400**3,

            # Zone separation options
            'gcube_n_buffer_cells': 2,

            # Particle type options
            'zone1_gcell_load': 1331, 
            'zone1_type': "glass",
            'zone2_type': "glass",
            'zone2_mass_factor': 8.0,
            'zone3_ncell_factor': 0.5,
            'zone3_min_n_cells': 20,
            'zone3_max_n_cells': 1000,
            'min_gcell_load': 8,
        }

        cparams = {}
        for key in defdict:
            cparams[key] = params[key] if key in params else defdict[key]

        return cparams

    def get_extra_params(self, params):
        """
        Get extra parameters for passing through to other codes.

        See the template parameter file for a description of each parameter.
    
        Parameters
        ----------
        params : dict
            The parsed YAML input parameter file.

        Returns
        -------
        xparams : dict
            The full set of extra parameters including defaults.

        """
        defdict = {
            # Basic switches
            'generate_param_files': True,
            'generate_submit_files': True,
            'code_types': 'ICGen, SWIFT',
            
            # General simulation options
            'z_initial': 127.0,
            'z_final': 0.0,
            'sim_type': 'dmo',
            'dm_only_sim': True,

            # IC-Gen specific parameters
            'num_species': 2,
            'icgen_fft_to_gcube_ratio': 1.0,
            'icgen_nmaxpart': 36045928,
            'icgen_nmaxdisp': 791048437,
            'icgen_runtime_hours': 4,
            'icgen_PH_nbit': 21,
            'fft_min_Nyquist_factor': 2.0,
            'fft_n_start': None,
            'icgen_multigrid': True,
            'icgen_num_cores': 28,
            'panphasian_descriptor': None,
            'icgen_constraint_phase_descriptor': '%dummy',
            'icgen_constraint_phase_descriptor2': '%dummy',
            'icgen_constraint_phase_descriptor_levels': '%dummy',
            'icgen_constraint_phase_descriptor2_levels': '%dummy',
            'icgen_constraint_phase_descriptor_path': '%dummy',
            'icgen_constraint_phase_descriptor2_path': '%dummy',

            # Softening parameters
            'comoving_eps_ratio': 1/20,
            'proper_eps_ratio': 1/45,            
            'background_eps_to_mips_ratio': 0.02,
            
            # SWIFT specific options
            'swift_dir': None,
            'swift_num_nodes': 1,
            'swift_runtime_hours': 72,
            'swift_ics_dir': '.',
            'swift_exec': '../swiftsim/examples/swift',
            'swift_num_nodes': 1,
            'swift_runtime_hours': 72,

            # GADGET specific options
            'gadget_num_cores': 32,
            'gadget_exec': './gadget/P-Gadget3-DMO-NoSF',            

            # System-specific parameters
            'memory_per_core': 18.2e9,
            'num_cores_per_node': 28,
        }
        
        xparams = {}
        for key in defdict:
            xparams[key] = params[key] if key in params else defdict[key]

        return xparams

    def initialize_sim_box(self):
        """Initialize the structure to hold info about full simulation box."""
        sim_box = {
            'l_mpchi': self.config['box_size'],
            'volume_mpchi': self.config['box_size']**3
        }
        return sim_box

    def get_target_resolution(self):
        """
        Compute the target resolution.

        If the uniform(-equivalent) number of particles is specified, this is
        used. Otherwise, the closest possible number is calculated from the
        specified target particle mass and number of particles per zone1 gcell.

        """
        num_part_equiv = self.config["uniform_particle_number"]
        zone1_gcell_load = self.config["zone1_gcell_load"]

        if num_part_equiv is None:

            # Compute from target particle mass
            m_target = self.config["target_mass"]
            if m_target is None:
                raise ValueError("Must specify target mass!")

            m_target = float(m_target)
            n_per_gcell = np.cbrt(zone1_gcell_load)

            m_frac_target = m_target / self.sim_box['mass_msun']
            num_equiv_target = 1. / m_frac_target

            n_equiv_target = np.cbrt(num_equiv_target)            
            n_gcells_equiv = int(np.rint(n_equiv_target / n_per_gcell))
            num_part_equiv = n_gcells_equiv**3 * zone1_gcell_load

        else:
            m_target = self.sim_box['mass_msun'] / num_part_equiv
        
        # Sanity check: number of particles must be an integer multiple of the
        # glass file particle number.
        if np.abs(num_part_equiv / zone1_gcell_load % 1) > 1e-6:
            raise ValueError(
                f"The full-box-equivalent particle number "
                f"({num_part_equiv}) must be an integer multiple of the "
                f"Zone I gcell load ({zone1_gcell_load})!"
            )
        self.sim_box['num_part_equiv'] = num_part_equiv
        self.sim_box['n_part_equiv'] = np.cbrt(num_part_equiv)

        print(f"Target resolution is {m_target:.2e} M_Sun, eqiv. to "
              f"n = {self.sim_box['n_part_equiv']}^3.")

    def compute_box_mass(self):
        """Compute the total masses in the simulation volume."""
        if comm_rank == 0:
            h = self.cosmo['hubbleParam']
            omega0 = self.cosmo['Omega0']
            omega_baryon = self.cosmo['OmegaBaryon']

            cosmo = FlatLambdaCDM(
                H0=h*100., Om0=omega0, Ob0=omega_baryon)
            rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc ** 3).value

            box_volume = self.sim_box['volume_mpchi'] / h
            m_tot = omega0 * rho_crit * box_volume

        else:
            m_tot = None

        # Send masses to all MPI ranks and store as class attributes
        self.sim_box['mass_msun'] = comm.bcast(m_tot)
        if self.verbose:
            print(f"Total box mass is {self.sim_box['mass_msun']:.2e} M_Sun")

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

        Note
        ----
        For the gcells, we work in length units of the gcell sidelength.

        """
        gcube = self.gcube

        # Assign indices of local gcells, such that all cells with
        # index % comm_size == comm_rank end up on local MPI rank
        gcell_idx = np.arange(comm_rank, gcube['num_cells'], comm_size)
        n_gcells = len(gcell_idx)

        # Calculate the central coordinates of local gcells
        gcell_pos = cell_centre_from_index(gcell_idx, [gcube['n_cells']] * 3)
        gcell_types = np.ones(n_gcells, dtype='i4') * -1

        if self.config['is_zoom']:

            # Rescale coordinates of mask cells to gcell coordinates. Recall
            # that the mask cell coordinates already have the same origin.
            f_scale = 1. / gcube['cell_size']
            mask_pos = self.mask_data['cell_coordinates'] * f_scale
            mask_cell_size = self.mask_data['cell_size'] * f_scale

            # Check that the values make sense
            cmin = np.min(mask_pos - mask_cell_size / 2)
            cmax = np.max(mask_pos + mask_cell_size / 2)
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
                print(f"Extent of mask cell centres [gcell units]:")
                for idim, name in enumerate(['x', 'y', 'z']):
                    print(f"  {mask_pos[:, idim].min():.3f} <= "
                          f"{name} <= {mask_pos[:, idim].max():.3f}")
                print(f"Mask cell size = {mask_cell_size:.2f} gcells.\n")

            # .............................................................

            # Assign a type (resolution level) to each local gcell.
            # Those that overlap with the mask are type 0 (i.e. target res.),
            # others have >= 1 depending on distance from the mask.
            
            cy.assign_mask_cells(
                mask_pos, mask_cell_size, gcell_pos, gcell_types)
            n_type0 = np.count_nonzero(gcell_types == 0)
            n_type0_tot = comm.allreduce(n_type0)
            n_type0_min = comm.reduce(n_type0, op=MPI.MIN)
            n_type0_max = comm.reduce(n_type0, op=MPI.MAX)
            if comm_rank == 0:
                print(
                    f"Assigned {n_type0_tot} gcells to highest resolution.\n"
                    f"   ({n_type0_min} - {n_type0_max} per rank)\n"
                )

            # ..............................................................

            num_gcell_types = self.assign_zone2_types(
                gcell_pos, gcell_idx, gcell_types)

        else:
            # If this is not a zoom simulation, all gcells are type 0.
            num_gcell_types = 1
            gcell_types[:] = 0

        # Total memory size gcell structure
        memory_size_in_byte = (
            sys.getsizeof(gcell_types) + sys.getsizeof(gcell_idx) +
            sys.getsizeof(gcell_pos)
        )

        # Final step: work out particle load info for all cells
        self.gcell_info = self.prepare_gcube_particles(
            gcell_types, num_gcell_types)
        return {'index': gcell_idx, 'pos': gcell_pos, 'types': gcell_types,
                'num': n_gcells, 'memsize': memory_size_in_byte}

    def assign_zone2_types(self, pos, index, types):
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
        num_types : int
            The total (cross-MPI) number of assigned gcell types.

        """
        stime = time.time()
        num_tot_zone2 = comm.allreduce(np.count_nonzero(types == -1))
        if comm_rank == 0:
            print(f"Assigning resolution level to {num_tot_zone2} gcells "
                  f"outside target high-res region...")

        # Initialize loop over (undetermined) number of gcell types.
        num_to_assign = num_tot_zone2
        source_type = 0
        num_assigned = 0

        while num_to_assign > 0:

            # Check that we don't have any cells higher than source type...
            if np.max(types) > source_type:
                raise ValueError(
                    f"Assigning neighbours to source type {source_type}, "
                    f"but already have cells with type up to {np.max(types)}!")

            if comm_rank == 0 and self.verbose > 1:
                print(f"Tagging neighbours of cell type {source_type}. "
                      f"Have so far tagged {num_assigned} gcells, "
                      f"{num_to_assign} still to go..."
                )

            # Find direct neighbours of all (local) gcells of current type
            ind_source = np.nonzero(types == source_type)[0]
            ngb_indices = find_neighbour_cells(
                pos[ind_source, :], self.gcube['n_cells'])

            # Combine skin cells across MPI ranks, removing duplicates
            ngb_indices = mpi_combine_arrays(ngb_indices, unicate=True)

            # Find local, unassigned gcells amongst the neighbours
            idx = np.where((np.isin(index, ngb_indices)) & (types == -1))[0]
            types[idx] = source_type + 1

            # Update number of assigned and still-to-be-assigned gcells
            num_assigned_now = comm.allreduce(len(idx))
            if num_assigned_now == 0 and comm_rank == 0:
                raise ValueError(
                    f"Have assigned {num_assigned_now} neighbour gcells "
                    f"for source type {source_type}, but also have "
                    f"{num_to_assign} more gcells unassigned!"
                )
            num_assigned += num_assigned_now
            num_to_assign = comm.allreduce(np.count_nonzero(types == -1))
            if num_to_assign == 0:
                break

            source_type += 1

        if num_assigned != num_tot_zone2:
            raise ValueError(
                f"Assigned {num_assigned} zone-II gcells, not {num_tot_zone2}")
        if comm_rank == 0:
            print(f"   ... done, assigned {source_type} different levels "
                  f"to Zone II gcells ({time.time() - stime:.2e} sec.)")

        return source_type + 2       # includes type 0; num = max + 1

    def find_scube_structure(self, target_n_cells, tolerance_n_cells=5,
                             max_extra=10, eps=0.01):
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

        Class attributes updated
        ------------------------
        scube : dict
            Properties of the cube (with a central hole...) formed by the
            set of cubic shells in Zone III.

        Notes
        -----
        All lengths here are expressed in units of the simulation box size.
        
        """ 
        stime = time.time()
        if target_n_cells < 5:       # "Dat kan niet."
            raise ValueError(
                f"Cannot build scube with target n_cells = {target_n_cells}.")
        if target_n_cells < 25 and comm_rank == 0:
            print(f"WARNING: target_n_cells = {target_n_cells}, limited "
                  f"range available for variations.")

        gcube_length = self.gcube['sidelength']
        self.scube['base_shell_l_inner'] = self.gcube['sidelength']

        # Set up arrays of all to-be-evaluated combinations of n_cells and
        # n_extra. For extra, we first set up a slighly offset float array
        # and reduce it later to test n_extra = 0 twice (see below).
        n_1d = np.arange(max(target_n_cells - tolerance_n_cells, 10),
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
        ns = (np.floor(ns_ideal)).astype(int) + delta_ns
    
        #set_trace()
        # Volume enclosed by next-to-outermost shell
        v_inner = (gcube_length * factors**(ns-1))**3

        # Volume of outermost shell, accounting for different cell number
        n_outer = n + n_extra
        num_cells_outer = 6 * n_outer * n_outer - 12 * n_outer + 8
        cellsizes_outer = gcube_length*factors**(ns - 1) / (n_outer - 2)
        v_outer = num_cells_outer * cellsizes_outer**3

        # Fractional difference in volume from ideal value
        # (recall that the simulation box has length = volume = 1 here)
        v_tot = v_inner + v_outer
        v_diff = (v_tot - 1) / (1 - gcube_length**3)

        # Find the combination with the smallest v_diff
        ind_best = np.argmin(np.abs(v_diff))
        if v_diff[ind_best] > eps:
            raise Exception(
                f"Could not find acceptable scube parameters\n"
                f"(best combination differs by {v_diff[ind_best] * 100:.2e} "
                f"per cent in volume, max allowed is {eps * 100})."
            )

        self.scube['n_cells'] = n[ind_best]
        self.scube['n_shells'] = ns[ind_best]
        self.scube['n_extra'] = n_extra[ind_best]
        self.scube['volume'] = v_tot[ind_best] - gcube_length**3
        self.scube['delta_volume_fraction'] = v_diff[ind_best]
        self.scube['l_ratio'] = n[ind_best] / (n[ind_best] - 2)

        # Sanity checks
        if (self.scube['base_shell_l_inner'] *
            (self.scube['l_ratio']**(self.scube['n_shells']-1)) > 1):
            raise ValueError(
                f"Outermost scube shell has an inner radius > 1!")

        # "Leap mass" (=volume fraction) added to each particle (cell)
        # in the outermost shell to get to the exact mass within the sim box
        # (box has length 1, so shortfall -- which may be -ve! -- is v_tot-1;
        # gcube volume is subtracted from both and cancels out).
        n_cells_outer = self.scube['n_cells'] + self.scube['n_extra']
        num_cells_outer = 6*n_cells_outer**2 - 12*n_cells_outer + 8
        self.scube['leap_mass'] = (1 - v_tot[ind_best]) / (num_cells_outer)

        # Compute number of shells and Zone III particles for this rank.
        # (we will assign successive shells to different MPI ranks).
        tot_nshells = self.scube['n_shells']
        nc = self.scube['n_cells']
        num_shells_local = tot_nshells // comm_size
        if comm_rank < tot_nshells % comm_size:
            num_shells_local += 1
        self.scube['n_shells_local'] = num_shells_local    # Can probably go

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

        self.scube['m_min'] = (
            self.scube['base_shell_l_inner'] / (self.scube['n_cells'] - 2))**3

        scell_size_outer = (
            self.scube['base_shell_l_inner'] *
            self.scube['l_ratio']**self.scube['n_shells'] /
            (self.scube['n_cells'] + self.scube['n_extra'])
        )
        self.scube['m_max'] = scell_size_outer**3 + self.scube['leap_mass']

        # Report what we found.
        if comm_rank == 0:
            print(
                f"\nFound optimal scube structure for Zone III "
                f"({time.time() - stime:.2e} sec.):\n"
                f"  N_cells = {n[ind_best]} (target: {target_n_cells}, "
                f"{6*nc*nc - 12*nc + 8} particles per shell)\n"
                f"  Shell length ratio = {self.scube['l_ratio']:.3f}\n"
                f"  N_extra = {n_extra[ind_best]} "
                f"({6*(nc+1)**2 - 12*(nc+1) + 8} particles in outer shell)\n"
                f"  N_shells = {ns[ind_best]} "
                f"(ideal: {ns_ideal[ind_best]:.3f})\n"
                f"  Volume = {self.scube['volume']:.4f}\n"
                f"  Volume deviation fraction = {v_diff[ind_best]:.2e}"
                f" (leap mass fraction: {self.scube['leap_mass']:.3e})\n"
                f"  Total particle number = {self.scube['num_part_all']:,}\n"
                f"  Min mass fraction = {self.scube['m_min']:.2e}, max = "
                f"{self.scube['m_max']:.2e}\n"
            )

        return self.scube['num_part_local']

    def prepare_gcube_particles(self, gcell_types, num_gcell_types):
        """ 
        Work out how to populate gcells with particles.

        This function calculates how many particles of which mass should be
        generated within each gcell.

        Parameters
        ----------
        gcell_types : ndarray(int) [N_cells]
            Types of local gcells.
        num_gcell_types : int
            The total number of different gcell types on all MPI ranks.

        Returns
        -------
        gcell_info : dict
            Information about particle load per assigned gcell type:
            - num_parts_per_gcell : ndarray(int)
                The number of particles per gcell of each type.
            - particle_masses : ndarray(float)
                Masses (in units of total box mass) of particles within each
                gcell type.
            - num_gcells : ndarray(int)
                Number of (local) gcells of each type.
            - num_types : int
                The total number of allocated gcell types.

        Class attributes stored
        ----------------------- 
        nparts : dict
            - 'zone1_local', 'zone2_local', 'tot_local' : int
                The local (this rank) number of particles in zone1, zone2, and
                total (last one will be updated later for zone3).
            - 'zone1_all', 'zone2_all', 'tot_all' : int
                The total (cross-MPI) number of particles in zone1, zone2, and
                total (last one will be updated later for zone3).

        """
        # Extract relevant config parameters
        zone1_gcell_load = self.config['zone1_gcell_load']
        zone1_type = self.config['zone1_type']
        zone2_type = self.config['zone2_type']
        zone2_mass_drop_factor = self.config['zone2_mass_factor']
        min_gcell_load = self.config['min_gcell_load']

        # Number of high resolution particles this rank will have.
        gcell_info = {
            'num_parts_per_cell': np.zeros(num_gcell_types, dtype=int) - 1,
            'particle_masses': np.zeros(num_gcell_types) - 1,
            'num_cells': np.zeros(num_gcell_types, dtype=int) - 1,
            'num_types': num_gcell_types
        }

        gcell_volume_fraction = self.gcube['cell_size']**3

        # Analyse each gcell type in turn
        for itype in range(num_gcell_types):

            # Find and count local gcells of currently processed type
            ind_itype = np.nonzero(gcell_types == itype)[0]
            num_gcells_itype = len(ind_itype)
            gcell_info['num_cells'][itype] = num_gcells_itype
            if num_gcells_itype == 0:
                continue

            # Find ideal number of particles for each of these gcells
            if itype == 0:
                zone_type = zone1_type
                target_gcell_load = zone1_gcell_load
            else:
                zone_type = zone2_type
                reduction_factor = zone2_mass_drop_factor * itype
                target_gcell_load = int(np.ceil(
                    max(min_gcell_load, zone1_gcell_load / reduction_factor)))

            # Apply quantization constraints to find actual particle number
            if zone_type == 'glass':
                gcell_load = find_nearest_glass_number(
                    target_gcell_load, self.config['glass_files_dir'])
            else:
                gcell_load = find_next_cube(target_gcell_load)
            gcell_info['num_parts_per_cell'][itype] = gcell_load

            # Mass (fraction) of each particle in current gcell type
            mass_itype = gcell_volume_fraction / gcell_load
            gcell_info['particle_masses'][itype] = mass_itype

        # Some safety checks
        if np.min(gcell_info['num_cells'] < 0):
            raise ValueError(f"Negative entries in gcell_info['num_cells']...")
        num_gcells_processed = comm.reduce(np.sum(gcell_info['num_cells']))
        if comm_rank == 0 and num_gcells_processed != gcell_types.shape[0]:
            raise ValueError(
                f"Processed {num_gcells_processed} gcells, but there are "
                f"{self.gcube['num_cells']}!"
            )

        # Gather total and global particle numbers
        num_parts_by_type = (
            gcell_info['num_cells'] * gcell_info['num_parts_per_cell'])
        self.nparts['zone1_local'] = num_parts_by_type[0]
        self.nparts['zone2_local'] = np.sum(num_parts_by_type[1:])
        self.nparts['tot_local'] = np.sum(num_parts_by_type)
        self.nparts['zone1_all'] = comm.allreduce(self.nparts['zone1_local'])
        self.nparts['zone2_all'] = comm.allreduce(self.nparts['zone2_local'])
        self.nparts['tot_all'] = (
            self.nparts['zone1_all'] + self.nparts['zone2_all'])
        if self.verbose:
            num_parts_by_type_all = (
                np.zeros_like(num_parts_by_type) if comm_rank==0 else None)
            comm.Reduce([num_parts_by_type, MPI.LONG],
                        [num_parts_by_type_all, MPI.LONG],
                        op = MPI.SUM, root = 0
            )
            if comm_rank == 0:
                print(
                    f"Total number of Zone I/II particles by resolution "
                    f"level:\n   {num_parts_by_type_all}"
                )

        # Calculate and store/print some info on particle distributions 
        gcell_load_range = [
            np.min(gcell_info['num_parts_per_cell']),
            np.max(gcell_info['num_parts_per_cell'])
        ]
        self._find_global_resolution_range(gcell_load_range)
        self._find_global_mass_distribution(gcell_info['particle_masses'])

        return gcell_info

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
        scube['lowest_equiv_n_in_gcube'] : int
            Number of particles per dimension if all gcells were at the
            lowest assigned resolution.

        """
        gcube = self.gcube

        # Find the particle number per dimension that would fill the gcube
        # at the highest and lowest resolutions
        lowest_gcell_load = comm.allreduce(gcell_load_range[0], op=MPI.MIN)
        highest_gcell_load = comm.reduce(gcell_load_range[1], op=MPI.MAX)
        num_part_equiv_low = lowest_gcell_load * gcube['num_cells']
        num_part_equiv_high = highest_gcell_load * gcube['num_cells']

        # Print some statistics about the load factors
        if comm_rank == 0:
            print("")
            for ii, name in enumerate(['Target', 'Lowest']):
                if ii == 0:
                    neq = num_part_equiv_high
                    load = highest_gcell_load
                else:
                    neq = num_part_equiv_low
                    load = lowest_gcell_load
                nbox = neq * 1 / gcube['volume']
                print(f"{name} gcell load is {load} ({load**(1/3):.2f}^3) "
                      f"particles, corresponding to")
                print(
                    f"   {neq:,} ({neq**(1/3):.2f}^3) particles in the gcube,\n"
                    f"   {nbox:.2e} ({nbox**(1/3):.2f}^3) particles in the "
                    f"entire simulation box, and \n"
                    f"   {neq / gcube['volume']:,.3f} particles per (cMpc/h)^3."
                )       

        print(f"Lowest gcell load is {lowest_gcell_load}.")
        self.scube['lowest_equiv_n_in_gcube'] = np.cbrt(
            lowest_gcell_load * gcube['num_cells'])

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
        zone3_ncell_factor = self.config['zone3_ncell_factor']

        if not self.config['is_zoom']:
            self.nparts['zone3_local'] = 0
            self.nparts['zone3_all'] = 0
            return

        # Ideal number of scells per dimension: a bit lower than the equivalent
        # of the largest inter-particle spacing in the gcube.
        target_n_scells = int(np.clip(
            self.scube['lowest_equiv_n_in_gcube'] * zone3_ncell_factor,
            self.config['zone3_min_n_cells'], self.config['zone3_max_n_cells']
        ))
        if comm_rank == 0 and self.verbose > 1:
            print(f"\nZone III cubic shells should have a target number of "
                  f"{target_n_scells} cells per dimension.")

        # Find the actual number of scells including geometric constraints
        # of the scube, and the scube parameters and total particle number.
        npart_zone3 = self.find_scube_structure(target_n_scells)

        self.nparts['zone3_local'] = npart_zone3
        self.nparts['tot_local'] += npart_zone3
        self.nparts['zone3_all'] = comm.allreduce(npart_zone3)
        self.nparts['tot_all'] += self.nparts['zone3_all']

    def _find_global_mass_distribution(self, local_masses):
        """
        Find and store the global distribution of particle masses.

        This is a subfunction of prepare_gcube_particles(). 
        """
        m_to_msun = self.sim_box['mass_msun']  # m is the box mass fraction

        # Gather all globally occurring Zone I+II particle masses
        all_masses = np.unique(np.concatenate(comm.allgather(local_masses)))
        zone1_m = np.min(all_masses)
        if self.config['is_zoom']:
            ind_zone2 = np.nonzero(all_masses > zone1_m)[0]
            zone2_m_min = np.min(all_masses[ind_zone2])
            zone2_m_max = np.max(all_masses[ind_zone2])

        if comm_rank == 0:
            print(
                f"Zone I particles have a mass fraction "
                f"{zone1_m:.2e} ({zone1_m * m_to_msun:.2e} M_Sun)."
            )
            if self.config['is_zoom']:
                print(
                    f"Zone II particle mass fraction range is "
                    f"{zone2_m_min:.2e} -> {zone2_m_max:.2e}\n   "
                    f"({zone2_m_min * m_to_msun:.2e} -> "
                    f"{zone2_m_max * m_to_msun:.2e} M_Sun)."
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
        stime = time.time()

        gcube = self.gcube
        zone1_type = self.config['zone1_type']
        zone2_type = self.config['zone2_type']

        if comm_rank == 0:
            np_all = self.nparts['zone1_all'] + self.nparts['zone1_all']
            print(f"\n---- Generating {np_all:,} gcube particles "
                  f"(Zone I/II) ----\n")

        # Load all the glass files we need into a list of coordinate arrays
        glass = self.load_glass_files()
        if comm_rank == 0:
            print("")    
        
        # Loop over each gcell type and fill them with particles
        num_parts_created = 0
        for itype in range(self.gcell_info['num_types']):
            ind_type = np.nonzero(gcells['types'] == itype)[0]
            ncell_type = len(ind_type)
            if ncell_type != self.gcell_info['num_cells'][itype]:
                raise Exception(
                    f"Expected {self.gcell_info['num_cells']} "
                    f"gcells of type {itype}, but found {ncell_type}!"
                )

            gcell_load_type = self.gcell_info['num_parts_per_cell'][itype]
            particle_mass_type = self.gcell_info['particle_masses'][itype]

            is_glass = ((itype == 0 and zone1_type == 'glass') or
                        (itype > 0 and zone2_type == 'glass'))
            if is_glass:
                kernel = glass[gcell_load_type] - 0.5
            else:
                kernel = make_uniform_grid(num=gcell_load_type, centre=True)    

            cy.fill_gcells_with_particles(
                gcells['pos'][ind_type, :], kernel, parts['pos'],
                parts['m'], particle_mass_type, num_parts_created
            )
            np_type = ncell_type * gcell_load_type
            num_parts_created += np_type

            # Brag about what we have done.
            if self.verbose > 1:
                mass_msun = particle_mass_type * self.sim_box['mass_msun']
                print(
                    f"[{comm_rank}: Type {itype}] num_part = {np_type} "
                    f"({np_type**(1/3):.3f}^3) in {ncell_type} cells "
                    f"({gcell_load_type}/cell)\n             mass fraction "
                    f"= {particle_mass_type:.3e} "
                    f"(DMO mass {mass_msun:.2e} M_sun)."
                )

        num_p_gcube = self.nparts['zone1_local'] + self.nparts['zone2_local']
        if num_parts_created != num_p_gcube:
            raise ValueError(
                f"Should have created {num_p_gcube} particles, but did "
                f"create {num_parts_created}!"
            )
        num_part_total = comm.allreduce(num_parts_created)
        if comm_rank == 0:
            print(
                f"Generated {num_part_total:,} ({num_part_total**(1/3):.2f}^3) "
                f"particles in Zone I/II."
            )

        # Scale coordinates to units of the simulation box size (i.e. such
        # that the edges of the simulation box (not gcube!) are at coordinates 
        # -0.5 and 0.5.
        gcube_range_inds = np.array((-0.5, 0.5)) * gcube['n_cells']
        gcube_range_boxfrac = (np.array((-0.5, 0.5)) * gcube['sidelength'])
        rescale(parts['pos'][:, :num_parts_created],
                gcube_range_inds, gcube_range_boxfrac, in_place=True)

        # Do consistency checks in separate function for clarity.
        self._verify_gcube_region(parts, num_parts_created, gcube['volume'])
        if comm_rank == 0:
            print(f"--> Finished generating gcube particles in "
                  f"{time.time() - stime:.3e} sec.")

    def load_mask_file(self):
        """
        Load the (previously computed) mask file that defines the zoom region.

        Most of this is only relevant for zoom simulations; for uniform volume
        ICs we only return (0.5, 0.5, 0.5) as the "centre". For zooms, te mask
        file should have been generated with the `MakeMask` class in 
        `make_mask/make_mask.py`; an error is raised if no mask file is
        specified or if the specified file does not exist.
                
        Parameters
        ----------
        None

        Returns
        -------
        mask_data : dict
            Dict containing the coordinates and sizes of all mask cells,
            as well as the extent of their cubic bounding box.
            ** TODO ** Should also load the box size here.

        centre : ndarray(float) [3]
            The central point of the mask in the simulation box.

        """
        stime = time.time()
        lbox_mpchi = self.sim_box['l_mpchi']
        mask_file = self.config['mask_file']

        if not self.config['is_zoom']:
            print(f"Uniform volume simulation, centre: "
                  f"{self.sim_box['l_mpchi'] * 0.5:.2f} h^-1 Mpc in x/y/z.")
            return None, np.array((0.5, 0.5, 0.5))

        if comm_rank == 0:
            if self.config['mask_file'] is None:
                raise AttributeError(
                    "You need to specify a mask file for a zoom simulation!"
                )
            if not os.path.isfile(mask_file):
                raise OSError(
                    f"The specified mask file '{mask_file}' does not exist!")

        # Load data on rank 0 and then distribute to other MPI ranks
        if comm_rank == 0:
            mask_data = {}
            with h5.File(mask_file, 'r') as f:
                
                # Centre of the high-res zoom in region
                centre = np.array(
                    f['Coordinates'].attrs.get("geo_centre")) / lbox_mpchi

                # Data specifying the mask for the high-res region
                mask_data['cell_coordinates'] = np.array(
                    f['Coordinates'][...], dtype='f8') / lbox_mpchi
                mask_data['cell_size'] = (
                    f['Coordinates'].attrs.get("grid_cell_width")) / lbox_mpchi

                # Also load the side length of the cube enclosing the mask,
                # and the volume of the target high-res region (at the
                # selection redshift).                
                mask_data['extent'] = (
                    f['Coordinates'].attrs.get("bounding_length")) / lbox_mpchi
                mask_data['high_res_volume'] = (
                    f['Coordinates'].attrs.get("high_res_volume")
                    / lbox_mpchi**3)
        else:
            mask_data = None
            centre = None

        mask_data = comm.bcast(mask_data)
        centre = comm.bcast(centre)

        if comm_rank == 0:
            centre_mpchi = centre * lbox_mpchi 
            num_mask_cells = mask_data['cell_coordinates'].shape[0]
            print(f"Finished loading data from mask file "
                  f"({time.time() - stime:.2e} sec.)")
            print(f"  Target centre: "
                  f"{centre[0]:.2f} / {centre[1]:.2f} / {centre[2]:.2f}; "
                  f"({centre_mpchi[0]:.2f} / {centre_mpchi[1]:.2f} / "
                  f"{centre_mpchi[2]:.2f}) h^-1 Mpc")
            print(f"  Bounding side: {mask_data['extent']:.3f} x box length\n"
                  f"  Number of cells: {num_mask_cells} (cell size: "
                  f"{mask_data['cell_size'] * lbox_mpchi:.2f} h^-1 Mpc)\n"
            )

        return mask_data, centre

    def compute_fft_params(self):
        """
        Work out what size of FFT grid we need for IC-Gen.

        Parameters
        ----------
        None

        Returns
        -------
        fft : dict
            Dict with three parameters specifying the high-res FFT grid.

        """
        # VIPs only
        if comm_rank != 0:
            return

        n_fft_start = self.extra_params['fft_n_start']
        f_Nyquist = self.extra_params['fft_min_Nyquist_factor']

        fft = self.compute_fft_highres_grid()
        print(
            f"--- HRgrid:\n   c={self.centre}, "
            f"L_box={self.sim_box['l_mpchi']:.2f} Mpc/h\n"
            f"L_grid={fft['l_mesh_mpchi']:.2f} Mpc/h, "
            f"n_eff={fft['n_eff']:.2f} (x2 = {fft['n_eff']*2:.2f})\n"
            f"FFT buffer fraction="
            f"{self.extra_params['icgen_fft_to_gcube_ratio']:.2f}"
        )

        # Find the minimum FFT grid size that is at least a factor
        # self.fft_min_Nyquist_factor larger than the Nyquist criterion
        # (i.e. the effective number of particles per dimension) and is a
        # multiple of n_fft_start
        n_fft_required = (fft['n_eff'] * f_Nyquist)
        pow2 = int(np.ceil(np.log(n_fft_required / n_fft_start) / np.log(2)))
        n_fft = n_fft_start * 2**pow2
        nyq_ratio = n_fft / fft['n_eff']
        print(f"Using FFT grid with {n_fft} points per side\n"
              f"   ({nyq_ratio:.2f} times the target-res Nyquist frequency).")

        fft['n_mesh'] = n_fft
        return fft

    def compute_fft_highres_grid(self):
        """Compute the properties of the FFT high-res grid."""
        if comm_rank != 0:
            return None

        # Extract the number of particles if the whole box were at target res
        num_part_box = self.sim_box['num_part_equiv']

        # This is trivial for periodic box simulations
        if not self.config['is_zoom']:
            fft_params = {
                'num_eff': num_part_box,
                'n_eff': int(np.rint(np.cbrt(num_part_box))),
                'l_mesh_mpchi': self.sim_box['l_mpchi']
            }
            return fft_params

        # Rest is only for "standard" (cubic) zooms        
        l_mesh = (self.extra_params['icgen_fft_to_gcube_ratio'] *
                  self.gcube['sidelength'])
        l_mesh_mpchi = l_mesh * self.sim_box['l_mpchi']
        if l_mesh > 1.:
            raise ValueError(
                f"Buffered zoom region is too big ({l_mesh:.3e})!")
        num_eff = int(num_part_box * l_mesh**3)

        # How many multi-grid FFT levels (this will update n_eff)?
        # Unclear whether this actually does anything...

        if self.extra_params['icgen_multigrid'] and l_mesh > 0.5:
            print(f"*** Cannot use multigrid ICs, mesh region is "
                  f"{l_mesh:.2f} > 0.5")
            self.extra_params['icgen_multigrid'] = False

        if self.extra_params['icgen_multigrid']:
            # Re-calculate size of high-res cube to be a power-of-two
            # fraction of the simulation box size
            n_levels = int(np.log(1/l_mesh) / np.log(2))
            print(f"Direct n_levels: {n_levels}")
            n_levels = 0
            while 1 / (2.**(n_levels+1)) > l_mesh:
                n_levels += 1
            print(f"Loop n_levels: {n_levels}")

            actual_l_mesh = 1. / (2.**n_levels)
            actual_l_mesh_mpchi = self.sim_box['l_mpchi'] / (2.**n_levels)
            if (actual_l_mesh_mpchi < l_mesh_mpchi):
                raise Exception("Multi-grid l_mesh is too small!")

            actual_num_eff = int(num_part_box * actual_l_mesh**3)
            
            print(
                f"--- HRgrid num multigrids={n_levels}, "
                f"lowest = {actual_l_mesh_mpchi:.2f} Mpc/h, "
                f"n_eff = {actual_num_eff**(1/3):.2f}^3 "
                f"(x2: {2*actual_num_eff**(1/3):.2f}^3)"
            )
        
        fft_highres_grid_params = {
                'num_eff': num_eff,
                'n_eff': int(np.rint(np.cbrt(num_eff))), # Exact up to precision
                'l_mesh_mpchi': l_mesh,
        }
        return fft_highres_grid_params

    def get_icgen_core_number(self, fft_params, optimal=False):
        """
        Determine number of cores to use based on memory requirements.
        Number of cores must also be a factor of ndim_fft.

        Parameters
        ----------
        fft_params : dict
            The parameters of the high-res FFT grid
        optimal : bool
            Switch to enable an "optimal" number of cores (default: False)

        Returns
        -------
        n_cores : int
            The minimally required number of cores.

        Class attributes stored
        -----------------------
        ic_params['icgen_n_cores'] : int
            The number of cores required for IC-Gen.

        """
        # For VIPs only.
        if comm_rank != 0:
            return

        if optimal:
            origin = 'optimized'
            n_max_part, n_max_disp = self.compute_optimal_ic_mem()
        else:
            origin = 'default'
            n_max_part = self.extra_params['icgen_nmaxpart']
            n_max_disp = self.extra_params['icgen_nmaxdisp']
        print(f"Using {origin} n_max_part={n_max_part}, "
              f"n_max_disp={n_max_disp}.")

        n_dim_fft = fft_params['n_mesh']
        num_cores_per_node = self.extra_params['num_cores_per_node']

        # Find number of cores that satisfies both FFT and particle constraints
        n_cores_from_fft = int(
            np.ceil((2*n_dim_fft**2 * (n_dim_fft/2 + 1))) / n_max_disp)
        n_cores_from_npart = int(np.ceil(self.nparts['tot_all'] / n_max_part))
        n_cores = max(n_cores_from_fft, n_cores_from_npart)

        # Increase n_cores until n_dim_fft is an integer multiple
        while (n_dim_fft % n_cores) != 0:
            n_cores += 1
  
        # If we're using one node, try to use as many of the cores as possible
        # (but still such that ndim_fft is an integer multiple). Since the
        # starting ncores works, we can safely increase n_fft_per_core
        if n_cores < num_cores_per_node:
            n_cores = num_cores_per_node
            while (n_dim_fft % n_cores) != 0:
                n_cores -= 1
 
        print(f"--- Using {n_cores} cores for IC-gen (minimum need for "
              f"{n_cores_from_fft} for FFT, {n_cores_from_npart} "
              f"for particles).")
        print(f"n_dim_fft = {n_dim_fft}")

        return n_cores

    def compute_optimal_ic_mem(self, n_fft):
        """
        Compute the optimal memory to fit IC-Gen on cosma7.

        Parameters
        ----------
        n_fft : int
            The side length of the FFT grid

        Returns
        -------
        max_parts : int
            The maximal number of particles that can be handled per core
        max_fft : int
            The maximal number of FFT grid points that can be handled per core

        """
        bytes_per_particle = 66
        bytes_per_fft_cell = 20
 
        num_parts = self.nparts['tot_all']
        num_fft = n_fft**3
        total_memory = ((bytes_per_particle * num_parts) +
                        (bytes_per_fft_cell * num_fft))

        num_cores = total_memory / self.extra_params['memory_per_core']
        max_parts = num_parts / num_cores
        max_fft = num_fft / num_cores
        print(f"--- Optimal max_parts={max_parts}, max_fft = {max_fft}")

        return max_parts, max_fft

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

        self.mask_data, self.centre = self.load_mask_file()

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
            self.print_particle_load_info(gcells)

        # If this is a "dry run" and we only want the particle number, quit.
        if only_calc_ntot:
            return

        # -------------------------------------------------------------------
        # ------ Act III: Creation (generate and verify particles) ----------
        # -------------------------------------------------------------------

        # Initiate particle arrays.
        pos = np.zeros((self.nparts['tot_local'], 3), dtype='f8') - 1e30
        masses = np.zeros(self.nparts['tot_local'], dtype='f8') - 1
        parts = {'pos': pos, 'm': masses}

        # Magic, part I: populate local gcells with particles (Zone I/II)
        self.generate_gcube_particles(gcells, parts)
         
        # Magic, part II: populate outer region with particles (Zone III)
        self.generate_zone3_particles(parts)

        # -------------------------------------------------------------------
        # --- Act IV: Transformation (shift coordinate system to target) ----
        # -------------------------------------------------------------------

        # Make sure that the particles are sensible before shifting
        self.verify_particles(parts)

        # Move particle load to final position in the simulation box
        self.shift_particles_to_target_position(parts)

        return parts

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - -  Functions called by make_particle_load - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
        stime = time.time()
        gcube = {}

        num_part_box = self.sim_box['num_part_equiv']
        zone1_gcell_load = self.config['zone1_gcell_load']
        num_buffer_gcells = self.config['gcube_n_buffer_cells']

        # Find the closest number of gcells that fill the box along one
        # side length. Recall that num_part_box is already guaranteed
        # to be (close to) an integer multiple of zone1_gcell_load, so the
        # rounding here is only to avoid numerical rounding issues.
        n_base = int(np.rint((num_part_box / zone1_gcell_load)**(1/3)))
        if (n_base**3 * zone1_gcell_load != num_part_box):
            raise ValueError(
                f"Want to use {n_base} cells per dimension, but this would "
                f"give {n_base**3 * zone1_gcell_load} particles, instead of "
                f"the target number of {num_part_box} particles."
            )

        # Side length of one gcell
        gcube['cell_size_mpchi'] = self.sim_box['l_mpchi'] / n_base
        gcube['cell_size'] = 1. / n_base   # In sim box size units

        if self.config['is_zoom']:
            # In this case, we have fewer gcells (in general), but still
            # use the same size.
            mask_cube_size = self.mask_data['extent']
            n_gcells = int(np.ceil(mask_cube_size / gcube['cell_size']))
            gcube['n_cells'] = n_gcells + num_buffer_gcells * 2

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
            gcube['n_cells'] = n_base
            mask_cube_size = 1.0

        gcube['num_cells'] = gcube['n_cells']**3
        if gcube['num_cells'] > (2**32) / 2:          # Dat kan niet.
            raise Exception(
                f"Total number of gcells ({gcube['num_cells']}) "
                f"is too large, can only have up to {(2**32)/2}."
            )

        # Compute the physical size of the gcube. Note that, for a
        # zoom simulation, this is generally not the same as the size of the
        # mask box, because of the applied buffer and quantization.
        gcube['sidelength_mpchi'] = gcube['cell_size_mpchi'] * gcube['n_cells']
        gcube['sidelength'] = gcube['cell_size'] * gcube['n_cells']
        gcube['volume_mpchi'] = gcube['sidelength_mpchi']**3
        gcube['volume'] = gcube['sidelength']**3
 
        if comm_rank == 0:
            print(
                f"Finished setting up gcube ({time.time() - stime:.2e} sec.)\n"
                f"  Side length: {gcube['sidelength_mpchi']:.4f} h^-1 Mpc "
                f"(= {gcube['sidelength']:.3e} x box size)\n"
                f"  Volume: {gcube['volume_mpchi']:.4f} (h^-1 Mpc)^3 "
                f"({gcube['volume'] * 100:.3f} % of the simulation volume,\n      "
                f"{gcube['volume']/(mask_cube_size**3) * 100:.3f} % "
                f"of the mask bounding cube)\n"
                f"  {gcube['n_cells']} gcells per dimension, of size "
                f"{gcube['cell_size_mpchi']:.3f} Mpc/h\n"
                f"  {gcube['num_cells']} gcells in total\n"
            )

        return gcube

    def load_glass_files(self):
        """
        Load all required glass files into a dict of coordinate arrays.

        Returns
        -------
        glass : dict
            Dict containing all required glass files, with their individual
            total particle number as key.

        """
        zone1_type = self.config['zone1_type']
        zone2_type = self.config['zone2_type']

        glass = {}
        num_parts_per_gcell_type = self.gcell_info['num_parts_per_cell']
        glass_dir = self.config['glass_files_dir']
        
        for iitype, num_parts in enumerate(num_parts_per_gcell_type):
            if num_parts in glass or num_parts == 0:
                continue
            if ((iitype == 0 and zone1_type != 'glass') or
                (iitype > 0 and zone2_type != 'glass')):
                continue
            glass[num_parts] = load_glass_from_file(num_parts, glass_dir)

        return glass

    def _verify_gcube_region(self, parts, nparts_created, gvolume):
        """
        Perform consistency checks and print info about high-res region.
        This is a subfunction of `populate_gcells()`.
        """
        # Make sure that the coordinates are in the expected range
        if np.max(np.abs(parts['pos'][:nparts_created])) > 0.5:
            raise ValueError("Invalid Zone I/II coordinate values!")

        # Make sure that we have allocated the right mass (fraction)
        tot_hr_mass = comm.allreduce(np.sum(parts['m'][:nparts_created]))
        if np.abs(tot_hr_mass - gvolume) > 1e-6:
            raise ValueError(
                f"Particles in Zone I/II have a combined "
                f"mass (fraction) of {tot_hr_mass} instead of the expected "
                f"{gvolume}!"
            )
        if comm_rank == 0:
            m_dev = (tot_hr_mass - gvolume) / gvolume
            print(f"Verified total Zone I/II mass fractions\n"
                  f"   ({tot_hr_mass:.2e}, ideal = {gvolume:.2e}, "
                  f"deviation = {m_dev * 100:.2e}%).")

        # Find and print the centre of mass of the Zone I/II particles
        # (N.B.: slicing creates views, not new arrays --> no memory overhead)
        com = centre_of_mass_mpi(
            parts['pos'][: nparts_created], parts['m'][:nparts_created])

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
        stime = time.time()
        npart_local = self.nparts['zone3_local']
        npart_all = self.nparts['zone3_all']
        offset_zone3 = self.nparts['zone1_all'] + self.nparts['zone2_all']

        # No zone III ==> no problem. This may also be the case for zooms.
        if npart_all == 0:
            return

        if comm_rank == 0:
            print(f"\n---- Generating {npart_all:,} outer low-res particles "
                  f"(Zone III) ----\n")

        if npart_local > 0:
            cy.fill_scube_layers(self.gcube, self.scube, self.nparts,
                                 parts, comm_rank, comm_size)

            m_max = np.max(parts['m'][offset_zone3 : ])
            m_min = np.min(parts['m'][offset_zone3 : ])

            if self.verbose:
                print(
                    f"[Rank {comm_rank}]: "
                    f"Generated {npart_local:,} ({npart_local**(1/3):.2f}^3) "
                    f"Zone III particles,\n          mass (fraction) range "
                    f"{m_min:.2e} --> {m_max:.2e}."
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
                f"Generated {npart_all:,} ({npart_all**(1/3):.2f}^3) "
                f"particles in Zone III.\n"
                f"   Mass fraction range: {m_min:.2e} --> {m_max:.2e} "
                f"({m_min * self.sim_box['mass_msun']:.3e} --> "
                f"{m_max * self.sim_box['mass_msun']:.3e} M_Sun)."
            )

        # Safety check on coordinates
        if npart_local > 0:
            # Don't abs whole array to avoid memory overhead
            if (np.max(parts['pos'][offset_zone3:, :]) > 0.5 or
                np.min(parts['pos'][offset_zone3:, :]) < -0.5):
                raise ValueError(
                    f"Zone III particles outside allowed range (-0.5 -> 0.5)! "
                    f"\nMaximum absolute coordinate is "
                    f"{np.max(np.abs(parts['pos'][offset_zone3:, :]))}."
                )

        if comm_rank == 0:
            print(f"--> Finished generating Zone III particles in "
                  f"{time.time() - stime:.3e} sec.")

    def verify_particles(self, parts):
        """Perform safety checks on all generated particles."""

        if comm_rank == 0:
            print("")
        npart_local = self.nparts['tot_local']
        npart_global = self.nparts['tot_all']

        # Safety check on coordinates and masses
        if npart_local > 0:
            # Don't abs whole array to avoid memory overhead
            if np.min(parts['pos']) < -0.5 or np.max(parts['pos']) > 0.5:
                raise ValueError(
                    f"Zone III particles outside allowed range (-0.5 -> 0.5)! "
                    f"\nMinimum coordinate is {np.min(parts['pos'])}, "
                    f"maximum is {np.min(parts['pos'])}."
                )
            if np.min(parts['m']) < 0 or np.max(parts['m']) > 1:
                raise ValueError(
                    f"Masses should be in the range 0 --> 1, but we have "
                    f"min={np.min(parts['m'])}, "
                    f"max={np.max(parts['m'])}."
                )

        # Check that the total masses add up to 1.
        total_mass_fraction = comm.allreduce(np.sum(parts['m']))
        mass_error_fraction = 1 - total_mass_fraction
        if np.abs(mass_error_fraction) > 1e-5:
            raise ValueError(
                f"Unacceptably large error in total mass fractions:\n"
                f"   Expected 1.0, got {total_mass_fraction:.4e} (error: "
                f"{mass_error_fraction}."
            )
        if np.abs(mass_error_fraction > 1e-6) and comm_rank == 0:
            print(
                f"**********\n   WARNING!!! \n***********\n"
                f"Error in total mass fraction is {mass_error_fraction}."
            )

        # Find the (cross-MPI) centre of mass, should be near origin.
        com = centre_of_mass_mpi(parts['pos'], parts['m'])
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
                f"Done verifying {self.nparts['tot_all']:,} "
                f"({self.nparts['tot_all']**(1/3):.2f}^3) particles."
            )

    def shift_particles_to_target_position(self, parts):
        """
        Move the centre of the high-res region to the desired point.

        This also applies periodic wrapping. On return, particles are in
        their final positions as required by IC-Gen (in range 0 --> 1).

        Returns
        -------
        None

        """
        # Shift particles to the specified centre (note that self.centre
        # is already in the range 0 --> 1, so that's all we need)...
        parts['pos'] += self.centre

        # ... and then apply periodic wrapping (simple: range is 0 --> 1).
        parts['pos'] %= 1.0

        if comm_rank == 0:
            print(f"\nShifted particles such that the Lagrangian centre of "
                  f"high-res region is at\n   "
                  f"{self.centre[0]:.3f} / {self.centre[1]:.3f} / "
                  f"{self.centre[2]:.3f}."
            )

    def create_param_and_submit_files(
        self, create_param=True, create_submit=True):
        """
        Create appropriate parameter and submit files for IC-GEN and SWIFT.
        """
        codes = self.extra_params['code_types']
        print(f"Generate files for codes '{codes}'")

        fft_params = self.compute_fft_params()
        n_cores_icgen = self.get_icgen_core_number(fft_params, optimal=False)
        eps = self.compute_softenings()

        all_params = self.compile_param_dict(fft_params, eps, n_cores_icgen)

        if create_param:
            mpf.make_all_param_files(all_params, codes.lower())
        if create_submit:
            mpf.make_all_submit_files(all_params, codes.lower())

    def compile_param_dict(self, fft_params, eps, n_cores_icgen):
        """Compile a dict of all parameters required for param/submit files."""

        extra_params = self.extra_params
        cosmo_h = self.cosmo['hubbleParam']
        param_dict = {}

        # Compute mass cuts between particle types.
        cut_type1_type2 = 0.0    # Dummy value if not needed
        cut_type2_type3 = 0.0    # Dummy value if not needed
        if self.config['is_zoom']:
            num_species = extra_params['num_species']
            if num_species >= 2:
                cut_type1_type2 = np.log10(
                    np.mean(self.gcell_info['particle_masses'][0:2]))
                print(f"log10 mass fraction cut from parttype 1 --> 2 = "
                      f"{cut_type1_type2:.2f}")
            if num_species == 3:
                cut_type2_type3 = np.log10(
                    (self.gcell_info['particle_masses'][-1] +
                     self.scube['m_min']) / 2
                )
                print(f"log10 mass fraction cut from parttype 2 --> 3 = "
                      f"{cut_type2_type3:.2f}")
            if num_species > 3:
                print(f"NumSpecies > 3 not supported, ignoring extra.")

        centre_mpchi = self.centre * self.sim_box['l_mpchi']

        param_dict['f_name'] = self.config['sim_name']
        param_dict['box_size_mpchi'] = self.sim_box['l_mpchi'] 
        param_dict['centre_x_mpchi'] = centre_mpchi[0]
        param_dict['centre_y_mpchi'] = centre_mpchi[1]
        param_dict['centre_z_mpchi'] = centre_mpchi[2]
        param_dict['l_gcube_mpchi'] = self.gcube['sidelength_mpchi']
        param_dict['is_zoom'] = self.config['is_zoom']

        for key in self.cosmo:
            param_dict[f'cosmo_{key}'] = self.cosmo[key]

        param_dict['ics_z_init'] = extra_params['z_initial']

        # IC-Gen specific parameters
        param_dict['icgen_dir'] = self.config['output_dir']
        param_dict['icgen_n_cores'] = n_cores_icgen
        param_dict['icgen_runtime_hours'] = extra_params['icgen_runtime_hours']
        param_dict['icgen_use_PH_ids'] = True
        param_dict['icgen_PH_nbit'] = extra_params['icgen_PH_nbit']
        param_dict['icgen_num_species'] = extra_params['num_species']
        param_dict['icgen_cut_t1t2'] = cut_type1_type2
        param_dict['icgen_cut_t2t3'] = cut_type2_type3
        param_dict['icgen_linear_powspec_file'] = (
            self.cosmo['linear_powerspectrum_file'])
        param_dict['icgen_panphasian_descriptor'] = (
            extra_params['panphasian_descriptor'])
        param_dict['icgen_n_part_for_uniform_box'] = (
            self.sim_box['num_part_equiv'])
        for key_suffix in ['', '2', '_path', '_levels', '2_path', '2_levels']:
            param_dict[f'icgen_constraint_phase_descriptor{key_suffix}'] = (
            extra_params[f'icgen_constraint_phase_descriptor{key_suffix}'])
        param_dict['icgen_multigrid'] = extra_params['icgen_multigrid']
        param_dict['icgen_n_fft_mesh'] = fft_params['n_mesh']
        param_dict['icgen_highres_num_eff'] = fft_params['num_eff']
        param_dict['icgen_highres_n_eff'] = fft_params['num_eff']**(1/3)
        param_dict['icgen_highres_l_mpchi'] = fft_params['l_mesh_mpchi']

        # Simulation-specific parameters
        param_dict['sim_eps_dm_mpchi'] = eps['dm']
        param_dict['sim_eps_dm_mpc'] = eps['dm'] / cosmo_h
        param_dict['sim_eps_to_mips_background'] = (
            extra_params['background_eps_to_mips_ratio'])
        param_dict['sim_eps_dm_pmpchi'] = eps['dm_proper']
        param_dict['sim_eps_dm_pmpc'] = eps['dm_proper'] / cosmo_h
        param_dict['sim_eps_baryon_mpchi'] = eps['baryons']
        param_dict['sim_eps_baryon_mpc'] = eps['baryons'] / cosmo_h
        param_dict['sim_eps_baryon_pmpchi'] = eps['baryons_proper']
        param_dict['sim_eps_baryon_pmpc'] = eps['baryons_proper'] / cosmo_h
        param_dict['sim_aexp_initial'] = 1 / (1 + extra_params['z_initial'])
        param_dict['sim_aexp_final'] = 1 / (1 + extra_params['z_final'])
        param_dict['sim_type'] = extra_params['sim_type']

        # SWIFT-specific parameters
        param_dict['swift_dir'] = extra_params['swift_dir']
        param_dict['swift_ics_dir'] = extra_params['swift_ics_dir']
        param_dict['swift_num_nodes'] = extra_params['swift_num_nodes']
        param_dict['swift_runtime_hours'] = extra_params['swift_runtime_hours']
        param_dict['swift_exec'] = extra_params['swift_exec']
        param_dict['swift_gas_splitting_threshold_1e10msun'] = (
            self.find_gas_splitting_mass())
    
        return param_dict       

    def find_gas_splitting_mass(self):
        """Compute the mass threshold for splitting gas particles."""
        f_baryon = self.cosmo['OmegaBaryon'] / self.cosmo['Omega0']
        m_tot_gas = self.sim_box['mass_msun'] * f_baryon
        m_tot_gas /= (self.cosmo['hubbleParam'] * 1e10)  # to 1e10 M_Sun [no h]
        return m_tot_gas / self.sim_box['num_part_equiv'] * 4

    def save_submit_files(self, max_boxsize):
        """
        Generate submit files.

        **TODO** Complete or remove.
        """
        if 'gadget' in self.sim_types:
            raise Exception(
                "Creation of GADGET submit files is not yet implemented.")

    def compute_softenings(self, verbose=False) -> dict:
        """
        Compute softening lengths, in units of Mpc/h.

        This is not required for the actual particle load generation, only
        to make the simulation parameter files.

        Returns
        -------
        eps : dict
            A dictionary with four keys: 'dm' and 'baryons' contain the
            co-moving softening lengths for DM and baryons (the latter is 0
            for DM-only simulations). 'dm_proper' and 'baryons_proper' contain
            the corresponding maximal proper softening lengths.
        """
        comoving_ratio = self.extra_params['comoving_eps_ratio']
        proper_ratio = self.extra_params['proper_eps_ratio']

        # Compute mean inter-particle separation (ips), in h^-1 Mpc.
        mean_ips = self.sim_box['l_mpchi'] / self.sim_box['n_part_equiv']

        # Softening lengths for DM
        eps_dm = mean_ips * comoving_ratio
        eps_dm_proper = mean_ips * proper_ratio

        # Softening lengths for baryons
        if self.extra_params["dm_only_sim"]:
            eps_baryon = 0.0
            eps_baryon_proper = 0.0
        else:
            # Adjust DM softening lengths according to baryon fraction
            fac = (self.cosmo['OmegaBaryon'] / self.cosmo['OmegaDM'])**(1/3)
            eps_baryon = eps_dm * fac
            eps_baryon_proper = eps_dm_proper * fac

        if comm_rank == 0 and verbose:
            print(f"Computed softening lengths:")
            h = self.cosmo['hubbleParam']
            if not self.extra_params["dm_only_sim"]:
                print(f"   Comoving softenings: DM={eps_dm:.6f}, "
                      f"baryons={eps_baryon:.6f} Mpc/h")
                print(f"   Max proper softenings: DM={eps_dm_proper:.6f}, "
                      f"baryons={eps_baryon_proper:.6f} Mpc/h")
            print(f"   Comoving softenings: DM={eps_dm / h:.6f} Mpc, "
                  f"baryond={eps_baryon / h:.6f} Mpc")
            print(f"   Max proper softenings: DM={eps_dm_proper / h:.6f} Mpc, "
                  f"baryons={eps_baryon_proper / h:.6f} Mpc\n")

        eps = {'dm': eps_dm, 'baryons': eps_baryon,
               'dm_proper': eps_dm_proper, 'baryons_proper': eps_baryon_proper}
        return eps

    def print_particle_load_info(self, gcells):
        """Print information about to-be-generated particles (rank 0 only)."""
        max_parts_per_file = self.config["max_numpart_per_file"]

        if comm_rank != 0:
            return

        np1 = self.nparts['zone1_all']
        np2 = self.nparts['zone2_all']
        np3 = self.nparts['zone3_all']
        np_tot = self.nparts['tot_all']
        print(f"--- Total number of Zone I particles: "
              f"{np1:,} ({np1**(1/3):.2f}^3, {np1 / np_tot * 100:.2f} %).")
        print(f"--- Total number of Zone II particles: "
              f"{np2:,} ({np2**(1/3):.2f}^3, {np2 / np_tot * 100:.2f} %).")
        print(f"--- Total number of Zone III particles: "
              f"{np3:,} ({np3**(1/3):.2f}^3, {np3 / np_tot * 100:.2f} %).")
        if self.config['is_zoom']:
            np_target = (self.sim_box['num_part_equiv'] *
                         self.mask_data['high_res_volume'])
            print(
                f"--- Target number of particles: "
                f"{np_target:.2e} ({np_target**(1/3):.2f}^3, "
                f"{np_target / np_tot * 100:.2f} % of actual total.)"
            )

        print(f"\n--- Total number of particles: "
              f"{np_tot:,} ({np_tot**(1/3):.2f}^3)")

        mem_gcells = memstring(gcells['memsize'])
        mem_parts = memstring(4 * 8 * np_tot)       # 4 * 8 byte / particle
        print(f"--- Total memory use per rank: {mem_gcells} for gcells, "
              f"{mem_parts} for particles.")
        print(f"--- Number of files needed for max "
              f"{max_parts_per_file:,} particles per file: "
              f"{int(np.ceil(np_tot / max_parts_per_file))}")

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

        ** TODO ** Modify this so that each rank breaks its own particles
        down into chunks with at most the max allowed number of particles.

        """
        output_formats = self.config['output_formats'].lower()
        save_as_hdf5 = 'hdf5' in output_formats
        save_as_fortran_binary = 'fortran' in output_formats
        # If we don't save in either format, we can stop right here.
        if not save_as_hdf5 and not save_as_fortran_binary:
            return

        # Randomise arrays, if desired
        if randomize:
            if comm_rank == 0:
                print('Randomizing arrays...')
            idx = np.random.permutation(self.nparts['tot_local'])
            self.parts['pos'] = self.parts['pos'][idx, :]
            self.parts['m'] = self.parts['m'][idx]

        self.parts['m'][:10000] *= (-1)
            
        # Load balance across MPI ranks.
        self.repartition_particles()

        # Save particle load as a collection of HDF5 and/or Fortran files
        save_dir = (f"{self.config['output_dir']}/ic_gen_submit_files/"
                    f"{self.config['sim_name']}/particle_load")
        save_dir_hdf5 = save_dir + '/hdf5'
        save_dir_bin = save_dir + '/fbinary'

        if comm_rank == 0:
            if not os.path.exists(save_dir_hdf5) and save_as_hdf5:
                os.makedirs(save_dir_hdf5)
            if not os.path.exists(save_dir_bin) and save_as_fortran_binary:
                os.makedirs(save_dir_bin)

        # Make sure to save no more than max_save files at a time.
        max_save = 50
        n_batch = int(np.ceil(comm_size / max_save))
        for ibatch in range(n_batch):
            if comm_rank % n_batch == ibatch:
                if save_as_hdf5:
                    hdf5_loc = f"{save_dir_hdf5}/PL.{comm_rank}.hdf5" 
                    self.save_local_particles_as_hdf5(hdf5_loc)

                if save_as_fortran_binary:
                    fortran_loc = f"{save_dir_bin}/PL.{comm_rank}"
                    self.save_local_particles_as_binary(fortran_loc)

                if self.verbose:
                    print(
                        f"[Rank {comm_rank}] Finished saving "
                        f"{self.nparts['tot_local']} local particles.")

            # Make all ranks wait so that we don't have too many writing
            # at the same time.
            comm.barrier()

    def repartition_particles(self):
        """Re-distribute particles across MPI ranks to achieve equal load."""
        if comm_size == 0:
            return

        num_part_all = self.nparts['tot_all']
        num_part_local = self.nparts['tot_local']
        num_part_min = comm.allreduce(num_part_local, op=MPI.MIN)
        num_part_max = comm.allreduce(num_part_local, op=MPI.MAX)

        # ** TODO **: move this to separate function prior to particle
        # generation. That way, we can immediately build the coordinate
        # arrays with the final size (if > locally generated size).
        num_part_desired = int(num_part_all // comm_size)
        num_part_leftover = num_part_all % comm_size
        num_per_rank = np.zeros(comm_size, dtype=int) + num_part_desired
        num_per_rank[: num_part_leftover] += 1
        
        if comm_rank == 0:
            n_per_rank = num_per_rank[0]**(1/3.)
            print(
                f"Load balancing {num_part_all} particles across "
                f"{comm_size} ranks ({n_per_rank:.2f}^3 per rank)...\n"
                f"Current load ranges from "
                f"{num_part_min} to {num_part_max} particles."
            )

            # Change this in future by allowing one rank to write >1 files
            if np.max(num_per_rank) > self.config['max_numpart_per_file']:
                print(f"***WARNING*** Re-partitioning will lead to (max) "
                      f"{np.max(num_per_rank):,} particles per file, more "
                      f"than the indicated maximum of "
                      f"{self.config['max_numpart_per_file']}!"
                )

        self.parts['m'] = repartition(
            self.parts['m'], num_per_rank, comm, comm_rank, comm_size) 
        num_part_new = len(self.parts['m'])

        # Because we repartition the coordinates individually, we first
        # need to expand the array if needed
        if num_part_new > self.parts['pos'].shape[0]:
            self.parts['pos'] = np.resize(
                self.parts['pos'], (num_part_new, 3))
            self.parts['pos'][self.nparts['tot_local']: , : ] = -1

        for idim in range(3):
            self.parts['pos'][: num_part_new, idim] = repartition(
            self.parts['pos'][:, idim], num_per_rank,
            comm, comm_rank, comm_size
        )
        if num_part_new < self.nparts['tot_local']:
            self.parts['pos'] = self.parts['pos'][:num_part_new, :]

        self.nparts['tot_local'] = num_part_new

        if comm_rank == 0:
            print('Done with load balancing.')

    def save_local_particles_as_hdf5(self, save_loc):
        """Write local particle load to HDF5 file `save_loc`"""
        n_part = self.nparts['tot_local']
        n_part_tot = self.nparts['tot_all']
 
        with h5.File(save_loc, 'w') as f:
            g = f.create_group('PartType1')
            g.create_dataset('Coordinates', (n_part, 3), dtype='f8')
            g['Coordinates'][:,0] = self.parts['pos'][:, 0]
            g['Coordinates'][:,1] = self.parts['pos'][:, 1]
            g['Coordinates'][:,2] = self.parts['pos'][:, 2]
            g.create_dataset('Masses', data=self.parts['m'])
            g.create_dataset('ParticleIDs', data=np.arange(n_part))

            g = f.create_group('Header')
            g.attrs.create('nlist', n_part)
            g.attrs.create('itot', n_part_tot)
            g.attrs.create('nj', comm_rank)
            g.attrs.create('nfile', comm_size)
            g.attrs.create('coords', self.centre)
            g.attrs.create('cell_length', self.gcube['cell_size'])
            g.attrs.create('Redshift', 1000)
            g.attrs.create('Time', 0)
            g.attrs.create('NumPart_ThisFile', [0, n_part, 0, 0, 0])
            g.attrs.create('NumPart_Total', [0, n_part_tot, 0, 0, 0])
            g.attrs.create('NumPart_TotalHighWord', [0, 0, 0, 0, 0])
            g.attrs.create('NumFilesPerSnapshot', comm_size)
            g.attrs.create('ThisFile', comm_rank)

        if comm_rank == 0:
            print(f"Done saving local particles to '{save_loc}'.")

    def save_local_particles_as_binary(self, save_loc):
        """Write local particle load to Fortran binary file `save_loc`"""
        f = FortranFile(save_loc, mode="w")

        # Write first 4+8+4+4+4 = 24 bytes
        f.write_record(
            np.int32(self.nparts['tot_local']),
            np.int64(self.nparts['tot_all']),
            np.int32(comm_rank),    # Index of this file
            np.int32(comm_size),    # Total number of files
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
        f.write_record(self.parts['pos'][:, 0].astype(np.float64))
        f.write_record(self.parts['pos'][:, 1].astype(np.float64))
        f.write_record(self.parts['pos'][:, 2].astype(np.float64))
        f.write_record(self.parts['m'].astype("float32"))
        f.close()

        if comm_rank == 0:
            print(f"Done saving local particles to '{save_loc}'.")


# ---------------------------------------------------------------------------

def memstring(m_byte):
    """Human-readable string for memory data."""
    if m_byte < 1024:
        return f"{m_byte} B"
    m_byte /= 1024
    if m_byte < 1024:
        return f"{m_byte:.2f} kB"
    m_byte /= 1024
    if m_byte < 1024:
        return f"{m_byte:.2f} MB"
    m_byte /= 1024
    if m_byte < 1024:
        return f"{m_byte:.2f} GB"
    m_byte /= 1024
    return f"{m_byte:.2f} TB"


def make_uniform_grid(n=None, num=None, centre=False):
    """
    Generate a uniform cubic grid with the specified number of points.

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
        n = int(np.rint(num**(1/3)))

    pos_1d = np.linspace(0, 1, num=n, endpoint=False)
    pos_1d += (1/(2*n))
    pos_3d = np.meshgrid(*[pos_1d]*3)

    # pos_3d is arranged as a 3D grid, so we need to flatten the three
    # coordinate arrays
    coords = np.zeros((n**3, 3), dtype='f8')
    for idim in range(3):
        coords[:, idim] = pos_3d[idim].flatten()

    # Safety check to make sure that the coordinates are valid.
    assert np.all(coords >= 0.0) and np.all(coords <= 1.0), (
        'Inconsistency generating a uniform grid')

    # If requested, shift coordinates to cube centre
    if centre:
        coords -= 0.5

    return coords


def find_nearest_glass_number(num, glass_files_dir):
    """Find the closest number of glass particles in a file to `num`.""" 
    files = os.listdir(glass_files_dir)
    num_in_files = [int(_f.split('_')[2]) for _f in files if 'ascii' in _f]
    num_in_files = np.array(num_in_files, dtype='i8')
    index_best = np.abs(num_in_files - num).argmin()
    return num_in_files[index_best]


def load_glass_from_file(num, glass_dir):
    """
    Load the particle distribution from a specified glass file.

    Parameters
    ----------
    num : int
        Suffix flor the specific glass file to use (number of particles).
    glass_dir : str
        Directory containing the different glass files.

    Returns
    -------
    r_glass : ndarray(float) [N_glass, 3]
        Array containing the coordinates of all particles in the glass
        distribution.

    ** TODO ** Check whether the explicit named dtype is really necessary.

    """
    glass_file = f"{glass_dir}/ascii_glass_{num}"
    if not os.path.isfile(glass_file):
        raise OSError(f"Specified glass file {glass_file} does not exist!")

    r_glass = np.loadtxt(glass_file)
    if comm_rank == 0:
        print(f"Loaded glass file with {num} particles.")

    return r_glass

def find_neighbour_cells(pos_source, n_cells, max_dist=1.):
    """
    Find indices of cells in a cubic grid that are near source cells.

    Neighbour cells are defined by their distance from a source cell,
    they can be local (on same rank) or remote.

    This function may be called with an empty source list, e.g. if the
    caller is an MPI-collective function. In this case, None is returned
    immediately.

    Parameters
    ----------
    pos_source : ndarray(float) [N_source, 3]
        The central coordinates of source cells, in gcell size units.
    n_cells : int
        The total number of cells per dimension; in general different from
        N_source^(1/3).
    max_dist : float, optional
        The maximum distance between cell centres for a gcell to be tagged
        as a neighbour. Default: 1.0, i.e. only the six immediate
        neighbour cells are tagged.

    Returns
    -------
    ngb_inds : ndarray(int) [N_ngb]
        The global scalar indices of all identified neighbours (may be empty).

    """
    if max_dist < 1:
        raise ValueError(
            f"max_dist = {max_dist}, so we cannot possibly find any "
            f"neighbour cells."
        )
    n_source = pos_source.shape[0]
    if n_source == 0:
        return np.zeros(0, dtype=int)

    # Build a kernel that contains the positions of all neighbour cell centres
    # relative to the target centre. Need to scale the grid up by
    # 2 * n_max/(n_max-1), because make_uniform_grid() puts the edges,
    # rather than the cell centres, at +/- 0.5.
    max_kernel_length = 2*int(np.floor(max_dist)) + 1
    kernel = make_uniform_grid(n=max_kernel_length, centre=True)
    kernel *= (2 * max_kernel_length/(max_kernel_length-1))

    kernel_r = np.linalg.norm(kernel, axis=1)
    ind_true_kernel = np.nonzero(kernel_r <= max_dist)[0]
    kernel = kernel[ind_true_kernel, :]
    n_kernel = len(ind_true_kernel)

    # Apply the kernel coordinate offsets to every source position
    pos_stretched = np.repeat(pos_source, n_kernel, axis=0)
    ngb_pos = pos_stretched + np.tile(kernel, (n_source, 1))

    # Find the indices of neighbours that are actually within the gcube
    ind_in_gcube = np.nonzero(
        np.max(np.abs(ngb_pos), axis=1) <= n_cells // 2)[0]
    ngb_inds = cell_index_from_centre(
        ngb_pos[ind_in_gcube, :], n_cells)

    return ngb_inds


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

    recvbuf = [a_all, n_elem_by_rank, offset_by_rank[:-1], mpi_type]
    comm.Gatherv(sendbuf=a, recvbuf=recvbuf, root=root)

    if comm_rank == 0:
        if unicate:
            a_all = np.unique(a_all)
    else:
        a_all = None

    if send_to_all:
        a_all = comm.bcast(a_all)

    return a_all


def find_next_cube(num):
    """Find the lowest number >=num that has a cube root."""
    return int(np.ceil(np.cbrt(num))**3)

def find_previous_cube(num, min_root=1):
    """Find the highest number <=num that has a cube root."""
    return int(max(np.floor(np.cbrt(num)), min_root)**3)

def find_nearest_cube(num, min_root=1):
    """Find the cube number closest to num."""
    root_low = int(max(np.floor(np.cbrt(num)), min_root))
    cube_low = root_low**3
    cube_high = (root_low + 1)**3
    diff_low = np.abs(cube_low - num)
    diff_high = np.abs(cube_high - num)
    if diff_low < diff_high:
        return cube_low
    else:
        return cube_high


def rescale(x, old_range, new_range, in_place=False):
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
    return np.ravel_multi_index(indices_3d_list, [n_cells]*3, order='F')


def centre_of_mass_mpi(coords, masses):
    """
    Compute the centre of mass for input particles, across MPI.

    Parameters
    ----------
    coords : ndarray(float) [N, 3]
        The coordinates of all N particles on the local rank.
    masses : ndarray(float) [N]
        The masses of all N particles on the local rank.

    Returns
    -------
    com : ndarray(float) [3]
        The cross-MPI centre of mass of all particles, on rank 0 only. On
        other ranks, the return value is not significant.

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


def get_cosmology_params(name):
    """Get cosmology parameters for a named cosmology."""    
    cosmo = {}
    if name == 'Planck2013':
        cosmo['Omega0'] = 0.307
        cosmo['OmegaLambda'] = 0.693
        cosmo['OmegaBaryon'] = 0.04825
        cosmo['hubbleParam'] = 0.6777
        cosmo['sigma8'] = 0.8288
        cosmo['linear_powerspectrum_file'] = 'extended_planck_linear_powspec'
    elif name == 'Planck2018':
        cosmo['Omega0'] = 0.3111
        cosmo['OmegaLambda'] = 0.6889
        cosmo['OmegaBaryon'] = 0.04897
        cosmo['hubbleParam'] = 0.6766
        cosmo['sigma8'] = 0.8102
        cosmo['linear_powerspectrum_file'] = 'EAGLE_XL_powspec_18-07-2019.txt'
    else:
        raise ValueError(f"Invalid cosmology '{name}'!")

    cosmo['OmegaDM'] = cosmo['Omega0'] - cosmo['OmegaBaryon']

    return cosmo


if __name__ == '__main__':
    only_calc_ntot = False
    if len(sys.argv) > 2:
        only_calc_ntot = True if int(sys.argv[2]) == 1 else False

    ParticleLoad(sys.argv[1], only_calc_ntot=only_calc_ntot)
