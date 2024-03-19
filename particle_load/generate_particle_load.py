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
import argparse

# Append modules directory to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "tools"
    )
)

# Append parent directory to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir
    )
)
from local import local
from tools import utils

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

except ImportError:
    print("Could not load mpi4py, running in serial mode")
    comm_rank = 0
    comm_size = 1
    from dummy import DummyComm, DummyMPI
    comm = DummyComm()
    MPI = DummyMPI()
    
import parallel_functions as pf
from scipy.io import FortranFile
import time
import kernel_replications as kr


import param_file_routines as pr

import pyximport
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)
import auxiliary_tools as cy

from pdb import set_trace

# Append modules directory to PYTHONPATH
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "tools"
    )
)
from timestamp import TimeStamp


rng = np.random.default_rng()

# ** TO DO **
# - Tidy up output

class ParticleLoad:
    """
    Class to generate and save a particle load from an existing mask.

    Parameters
    ----------
    args : object
        Structure containing the command-line arguments from argparse.
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
    def __init__(self, args, param_file: str = None,
                 randomize: bool = False, only_calc_ntot: bool = False,
                 verbose: bool = False, save_data: bool = True,
                 save_metadata: bool = True,
                 create_param_files: bool = True,
                 create_submit_files: bool = True
    ) -> None:

        self.verbose = 1

        if comm_rank == 0:
            print("----------------------------")
            print("  PARTICLE LOAD GENERATOR   ")
            print("----------------------------")

        ts = TimeStamp()
        
        # Read and process the parameter file.
        if param_file is None:
            param_file = args.param_file
        self.read_param_file(param_file, override_params=args.params)
        self.sim_box = self.initialize_sim_box()
        self.mask_data, self.centre = self.load_mask_file()

        if comm_rank == 0:
            print("Computing simulation and (target) particle masses...")
        self.compute_box_mass()
        self.get_target_resolution()

        ts.set_time('Setup')
        
        # Generate particle load.
        self.nparts = {}
        self.scube = {}
        if args.dry_run:
            only_calc_ntot = True
        self.parts, tss = self.make_particle_load(only_calc_ntot=only_calc_ntot)

        ts.import_times(tss)
        ts.set_time('Make particle load')
        
        # Generate param and submit files, if desired (NOT YET IMPLEMENTED).
        if comm_rank == 0:
            self.create_param_and_submit_files(
                create_param=self.extra_params['generate_param_files'],
                create_submit=self.extra_params['generate_submit_files']
            )

        # Save particle load
        if save_data and not only_calc_ntot:
            tss = self.save_particle_load(randomize=randomize)
            ts.import_times(tss)
        ts.set_time('Save particle load')
        if save_metadata and not only_calc_ntot:
            tss = self.save_metadata()
            ts.import_times(tss)
        ts.set_time('Save metadata')

        if comm_rank == 0:
            ts.print_time_usage('Finished')
            
    def read_param_file(
            self, param_file: str, override_params: dict = None)-> None:
        """Read in parameters for run."""

        # Read params from YAML file on rank 0, then broadcast to other ranks.
        if comm_rank == 0:
            params = yaml.safe_load(open(param_file))
            if isinstance(override_params, dict):
                for key in override_params:
                    params[key] = override_params[key]
        else:
            params = None            
        params = comm.bcast(params, root=0)

        # Define and enforce required parameters.
        required_params = [
            'is_zoom',
            'sim_name',
            'identify_gas',
            'cosmology',
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

        # Override power spectrum file if desired
        if self.extra_params['icgen_power_spectrum_file'] is not None:
            self.cosmo['linear_powerspectrum_file'] = (
                self.extra_params['icgen_power_spectrum_file'])
        
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
            'is_zoom': False,
            'box_size': None,
            'mask_file': None,
            'uniform_particle_number': None,
            'uniform_particle_n': None,
            'target_mass': None,
            'target_mass_type': None,
            'identify_gas': False,

            # In-/output options
            'output_formats': "Fortran",
            'glass_files_dir': './glass_files',
            'max_numpart_per_file': 400**3,
            'icgen_object_dir': None,
            'icgen_work_dir': None,

            # Zone separation options
            'gcube_n_buffer_cells': 2,
            'gcube_min_size_mpc': 0.0,

            # Particle type options
            'zone1_gcell_load': 1331, 
            'zone1_type': "glass",
            'zone2_type': "glass",
            'zone2_mpart_factor_per_mpc': 1.0,
            'zone2_min_mpart_over_zone1': 1.5,
            'zone2_max_mpart_over_zone1': None,
            'zone2_max_mpart_msun': None,
            'zone3_ncell_factor': 0.5,
            'zone3_min_n_cells': 20,
            'zone3_max_n_cells': 1000,
            'min_gcell_load': 8,
            'dm_to_gas_number_ratio': None,
            'dm_to_gas_mass_ratio': None,
            'generate_extra_dm_particles': False,
            'extra_dm_particle_scheme': None,
        }

        cparams = {}
        for key in defdict:
            cparams[key] = params[key] if key in params else defdict[key]

        if cparams['icgen_work_dir'] is None:
            if cparams['icgen_object_dir'] is None:
                raise ValueError(
                    "Must specify either IC_Gen object or workdir.")
            cparams['icgen_work_dir'] = (
                f"{cparams['icgen_object_dir']}/{params['sim_name']}")

        utils.set_none(cparams, 'dm_to_gas_number_ratio')
        utils.set_none(cparams, 'dm_to_gas_mass_ratio')
        utils.set_none(cparams, 'extra_dm_particle_scheme')
        utils.set_none(cparams, 'zone2_max_mpart_over_zone1')
        utils.set_none(cparams, 'zone2_max_mpart_msun')

        if (cparams['identify_gas'] and not
            cparams['generate_extra_dm_particles']):
            cparams['assign_gas'] = True
        else:
            cparams['assign_gas'] = False

        if not cparams['identify_gas']:
            cparams['generate_extra_dm_particles'] = False

        if cparams['generate_extra_dm_particles']:
            if cparams['dm_to_gas_number_ratio'] is None:
                raise ValueError("Must specify the DM oversampling factor!")

        if cparams['zone2_min_mpart_over_zone1'] < 1:
            raise ValueError(
                "Zone-II particles must be more massive than Zone-I!")

        # If an object directory is specified and has the specified mask file,
        # use that
        if (cparams['icgen_object_dir'] is not None and
            cparams['mask_file'] is not None):
            obj_mask_file = (
                f"{cparams['icgen_object_dir']}/{cparams['mask_file']}")
            if os.path.isfile(obj_mask_file):
                cparams['mask_file'] = obj_mask_file

        if cparams['target_mass_type'] is None:
            if cparams['identify_gas']:
                cparams['target_mass_type'] = 'mean'
            else:
                cparams['target_mass_type'] = 'dmo'
                    
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
            'code_types': 'IC_Gen_6p9',
            
            # General simulation options
            'z_initial': None,
            'z_final': 0.0,
            'sim_type': 'dmo',
            'dm_only_run': True,

            # IC-Gen specific parameters
            'icgen_exec': None,
            'icgen_powerspec_dir': None,
            'icgen_module_setup': None,
            'icgen_num_species': None,
            'icgen_fft_to_gcube_ratio': 1.0,
            'icgen_nmaxpart': 46656000,   # 'C7mem' build of IC_Gen
            'icgen_nmaxdisp': 454164480,  # 'C7mem' build of IC_Gen
            'icgen_runtime_hours': 4,
            'icgen_power_spectrum_file': None,
            'icgen_use_PH_IDs': True,
            'icgen_PH_nbit': 21,
            'fft_min_Nyquist_factor': 2.0,
            'fft_n_base': 3,
            'fft_n_min': 1536,
            'icgen_multigrid': True,
            'panphasian_descriptor': None,
            'icgen_num_constraints': 0,
            'icgen_p6_multipoles': 6.0,
            'icgen_constraint_phase_descriptor': '%dummy',
            'icgen_constraint_phase_descriptor2': '%dummy',
            'icgen_constraint_phase_descriptor_levels': '%dummy',
            'icgen_constraint_phase_descriptor2_levels': '%dummy',
            'icgen_constraint_phase_descriptor_path': '%dummy',
            'icgen_constraint_phase_descriptor2_path': '%dummy',

            # System-specific parameters
            'slurm_partition': local['slurm_partition'],
            'slurm_account': local['slurm_account'],
            'slurm_email': local['slurm_email'],
            'memory_per_core': local['memory_per_core'],
            'num_cores_per_node': local['cpus_per_node'],
        }
        
        xparams = {}
        for key in defdict:
            xparams[key] = params[key] if key in params else defdict[key]

        if xparams['icgen_num_species'] is None:
            xparams['icgen_num_species'] = 2 if self.config['is_zoom'] else 1

        if xparams['generate_param_files']:
            if 'ic_gen' in xparams['code_types'].lower():
                if xparams['panphasian_descriptor'] is None:
                    raise ValueError(
                        "Panphasian descriptor must be specified to set up "
                        "the IC_Gen param file!")
                if xparams['icgen_powerspec_dir'] is None:
                    raise ValueError(
                        "Directory of power spectrum files must be specified "
                        "to set up the IC_Gen param file!")
        if xparams['generate_submit_files']:
            if 'ic_gen' in xparams['code_types'].lower():
                if xparams['icgen_exec'] is None:
                    raise ValueError(
                        "IC_Gen executable must be specified to set up "
                        "the IC_Gen submit file!")

        return xparams

    def initialize_sim_box(self):
        """Initialize the structure to hold info about full simulation box."""
        h = self.cosmo['hubbleParam']

        sim_box = {}
        # For zooms, we will load the box size from the mask file in a moment.
        if not self.config['is_zoom']:
            sim_box['l_mpc'] = self.config['box_size']
            sim_box['volume_mpc'] = sim_box['l_mpc']**3

        return sim_box

    def compute_box_mass(self):
        """Compute the total masses in the simulation volume."""
        if comm_rank == 0:
            h = self.cosmo['hubbleParam']
            omega0 = self.cosmo['Omega0']
            omega_baryon = self.cosmo['OmegaBaryon']
            cosmo = FlatLambdaCDM(
                H0=h*100., Om0=omega0, Ob0=omega_baryon)

            rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc ** 3).value
            m_tot = omega0 * rho_crit * self.sim_box['volume_mpc']
        else:
            m_tot = None

        # Send masses to all MPI ranks and store as class attributes
        self.sim_box['mass_msun'] = comm.bcast(m_tot)
        if self.verbose and comm_rank == 0:
            print(f"Critical density is {rho_crit:.4e} M_Sun / Mpc^3")
            print(f"Total box mass is {self.sim_box['mass_msun']:.2e} M_Sun")

    def get_target_resolution(self):
        """
        Compute the target resolution.

        If the uniform(-equivalent) number of particles is specified, this is
        used. Otherwise, the closest possible number is calculated from the
        specified target particle mass and number of particles per zone1 gcell.

        """
        num_basepart_equiv = self.config["uniform_particle_number"]
        zone1_gcell_load = self.config["zone1_gcell_load"]

        # Compute from cube root number if needed and possible
        if num_basepart_equiv is None:
            if self.config['uniform_particle_n'] is not None:
                num_basepart_equiv = self.config['uniform_particle_n']**3

        if num_basepart_equiv is None:

            # --- Compute number from target particle mass ----
            m_target = self.config["target_mass"]
            if m_target is None:
                raise ValueError("Must specify target mass!")
            m_target = float(m_target)

            # Need to find mass of 'raw' particles (if all were same mass)
            if self.config['assign_gas']:
                # Generate enough for DM + gas particles

                dm_m_factor = self.config['dm_to_gas_mass_ratio']
                dm_n_factor = self.config['dm_to_gas_number_ratio']
                omega = self.cosmo['OmegaBaryon'] / self.cosmo['OmegaDM']
                omega_i = 1. / omega
                if dm_n_factor is not None:
                    if self.config['target_mass_type'] == 'gas':
                        m_target *= (1 + omega_i) / (1 + dm_n_factor)
                    elif self.config['target_mass_type'] == 'dm':
                        m_target *= (1 + omega) / (1 + 1 / dm_n_factor)

                elif dm_m_factor is not None:
                    if self.config['target_mass_type'] == 'gas':
                        m_target *= (1 + omega_i) / (1 + omega_i / dm_m_factor)
                    elif self.config['target_mass_type'] == 'dm':
                        m_target *= (1 + omega) / (1 + omega * dm_m_factor)

                else:
                    raise ValueError(
                        "Must specify either DM/gas number or mass ratio!")

            # If we get here, we divide gas/DM (if at all) by splitting

            elif self.config['target_mass_type'] == 'gas':
                m_target *= self.cosmo['Omega0'] / self.cosmo['OmegaBaryon']

            elif self.config['target_mass_type'] == 'dm':
                m_target *= self.cosmo['Omega0'] / self.cosmo['OmegaDM']

                # If we split off more than one DM particle per gas particle,
                # their (target) mass is reduced, so we need to start from
                # correspondingly more massive base particles.
                if self.config['generate_extra_dm_particles']:
                    m_target *= (self.config['dm_to_gas_number_ratio'])

            n_per_gcell = np.cbrt(zone1_gcell_load)

            m_frac_target = m_target / self.sim_box['mass_msun']
            num_equiv_target = 1. / m_frac_target

            n_equiv_target = np.cbrt(num_equiv_target)            
            n_gcells_equiv = int(np.rint(n_equiv_target / n_per_gcell))
            num_basepart_equiv = n_gcells_equiv**3 * zone1_gcell_load

            if comm_rank == 0:
                print(f"Ideal base particle mass is {m_target:.3e} M_Sun, "
                      f"corresponding to n_equiv = {n_equiv_target:.2f}.")
            
        else:
            m_target = self.sim_box['mass_msun'] / num_basepart_equiv
        
        # Sanity check: number of particles must be an integer multiple of the
        # glass file particle number.
        if np.abs(num_basepart_equiv / zone1_gcell_load % 1) > 1e-6:
            raise ValueError(
                f"The full-box-equivalent base particle number "
                f"({num_basepart_equiv}) must be an integer multiple of the "
                f"Zone I gcell load ({zone1_gcell_load})!"
            )
        
        num_part_equiv = num_basepart_equiv
        if self.config['generate_extra_dm_particles']:
            num_part_equiv *= (self.config['dm_to_gas_number_ratio'] + 1)

        self.sim_box['num_part_equiv'] = num_part_equiv
        self.sim_box['n_part_equiv'] = int(np.rint(np.cbrt(num_part_equiv)))

        self.sim_box['num_basepart_equiv'] = num_basepart_equiv
        self.sim_box['n_basepart_equiv'] = int(
            np.rint(np.cbrt(num_basepart_equiv)))

        m_base = self.sim_box['mass_msun'] / num_basepart_equiv

        if comm_rank == 0:
            print(f"Base resolution is {m_base:.3e} M_Sun, eqiv. to "
                  f"n = {self.sim_box['n_basepart_equiv']}^3 (full n = "
                  f"{self.sim_box['n_part_equiv']}^3)."
            )

    def load_mask_file(self):
        """
        Load the (previously computed) mask file that defines the zoom region.

        Most of this is only relevant for zoom simulations; for uniform volume
        ICs we only return (0.5, 0.5, 0.5) as the "centre". For zooms, the mask
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
        centre : ndarray(float) [3]
            The central point of the mask in the simulation box.

        Note
        ----
        All returned quantities are in units of the simulation box length.

        """
        stime = time.time()
        mask_file = self.config['mask_file']

        if comm_rank == 0:
            print(f"Reading mask file '{mask_file}'...", end='')

        if not self.config['is_zoom']:
            print(f"Uniform volume simulation, centre: "
                  f"{self.sim_box['l_mpc'] * 0.5:.2f} Mpc in x/y/z.")
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
                lbox_mpc = f['Params'].attrs.get('box_size')
                self.sim_box['l_mpc'] = lbox_mpc
                self.sim_box['volume_mpc'] = self.sim_box['l_mpc']**3

                centre = np.array(
                    f['Coordinates'].attrs.get("geo_centre")) / lbox_mpc

                # Data specifying the mask for the high-res region
                mask_data['cell_coordinates'] = np.array(
                    f['Coordinates'][...], dtype='f8') / lbox_mpc
                mask_data['cell_size'] = (
                    f['Coordinates'].attrs.get("grid_cell_width")) / lbox_mpc

                # Also load the side length of the cube enclosing the mask,
                # and the volume of the target high-res region (at the
                # selection redshift).                
                mask_data['extent'] = (
                    f['Coordinates'].attrs.get("bounding_length")) / lbox_mpc
                mask_data['high_res_volume'] = (
                    f['Params'].attrs.get("high_res_volume")
                    / lbox_mpc**3
                )

        else:
            mask_data = None
            centre = None
            self.sim_box = None

        mask_data = comm.bcast(mask_data)
        centre = comm.bcast(centre)
        self.sim_box = comm.bcast(self.sim_box)

        if comm_rank == 0:
            centre_mpc = centre * lbox_mpc 
            num_mask_cells = mask_data['cell_coordinates'].shape[0]
            print(f" done ({time.time() - stime:.2e} sec.)")
            print(f"  Simulation box size: {self.sim_box['l_mpc']:.3f} Mpc.")
            print(f"  Target centre: "
                  f"{centre[0]:.2f} / {centre[1]:.2f} / {centre[2]:.2f}; "
                  f"({centre_mpc[0]:.2f} / {centre_mpc[1]:.2f} / "
                  f"{centre_mpc[2]:.2f}) Mpc")
            print(f"  Bounding side: {mask_data['extent']:.3f} x box length\n"
                  f"  Number of mask cells: {num_mask_cells} (cell size: "
                  f"{mask_data['cell_size'] * lbox_mpc:.2f} Mpc)\n"
            )

        return mask_data, centre

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
            print('')
            print('Generating particle load...')
            print('............................\n')

        ts = TimeStamp()
        ts.add_counters(['Other'])
        
        # --------------------------------------------------------------------
        # --- Act I: Preparation (find structure and number of particles) ---
        # -------------------------------------------------------------------- 

        # Set up the gcube and generate local gcells
        self.gcube = self.set_up_gcube()
        ts.set_time('Set up gcube')
        
        # Prepare uniform grid cell structure within gcube
        # (sets up self.gcell_info)
        gcells = self.generate_gcells()
        ts.set_time('Generate gcells')
        
        # Prepare the cubic shell structure filling the outer box (Zone III)
        # (sets up self.scube dict)
        self.prepare_zone3_particles()
        ts.set_time('Prepare Zone-III particles')
        
        if comm_rank == 0:
            self.print_particle_load_info(gcells)

        # If this is a "dry run" and we only want the particle number, quit.
        if only_calc_ntot:
            return None, ts

        # -------------------------------------------------------------------
        # ------ Act II: Creation (generate and verify particles) ----------
        # -------------------------------------------------------------------

        # Initiate particle arrays.
        ts.increase_time('Other')
        pos = np.zeros((self.nparts['tot_local'], 3), dtype='f8') - 1e30
        ts.set_time('Create particle position arrays')
        masses = np.zeros(self.nparts['tot_local'], dtype='f8') - 1
        ts.set_time('Create particle mass arrays')
        parts = {'pos': pos, 'm': masses}
        ts.set_time('Dictize particle arrays')

        # Magic, part I: populate local gcells with particles (Zone I/II)
        self.generate_gcube_particles(gcells, parts, ts)
        #ts.set_time('Generate Zone-I/II particles')
        
        # Magic, part II: populate outer region with particles (Zone III)
        self.generate_zone3_particles(parts)
        ts.set_time('Generate Zone-III particles')

        # Make sure that the particles are sensible before shifting
        self.verify_particles(parts)
        ts.set_time('Overall particle verification')

        # -------------------------------------------------------------------
        # --- Act III: Transformation (shift coordinate system to target) ---
        # -------------------------------------------------------------------

        # Move particle load to final position in the simulation box
        self.shift_particles_to_target_position(parts)
        ts.set_time('Shift particles to target positions')
        
        return parts, ts

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

        num_part_box = self.sim_box['num_basepart_equiv']
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
        gcube['cell_size_mpc'] = self.sim_box['l_mpc'] / n_base
        gcube['cell_size'] = 1. / n_base   # In sim box size units
        gcube['cell_volume'] = gcube['cell_size']**3

        if self.config['is_zoom']:
            # In this case, we have fewer gcells (in general), but still
            # use the same size.
            mask_cube_size = self.mask_data['extent']

            buffer_size = gcube['cell_size'] * 2 * num_buffer_gcells
            gcube_min_size = max(
                mask_cube_size + buffer_size,
                self.config['gcube_min_size_mpc'] / self.sim_box['l_mpc']
            )
            gcube['n_cells'] = int(
                np.ceil(gcube_min_size / gcube['cell_size']))

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
        gcube['sidelength_mpc'] = gcube['cell_size_mpc'] * gcube['n_cells']
        gcube['sidelength'] = gcube['cell_size'] * gcube['n_cells']
        gcube['volume_mpc'] = gcube['sidelength_mpc']**3
        gcube['volume'] = gcube['sidelength']**3
 
        if comm_rank == 0:
            print(
                f"Finished setting up gcube ({time.time() - stime:.2e} sec.)\n"
                f"  Side length: {gcube['sidelength_mpc']:.4f} Mpc "
                f"(= {gcube['sidelength']:.3e} x box size)\n"
                f"  Volume: {gcube['volume_mpc']:.4f} Mpc^3\n"
                f"     {gcube['volume']*100:.3f} % of the simulation volume,\n"
                f"     {gcube['volume']/(mask_cube_size**3) * 100:.3f} % "
                f"of the mask bounding cube\n"
                f"  {gcube['n_cells']} gcells per dimension, of size "
                f"{gcube['cell_size_mpc']:.3f} Mpc\n"
                f"  {gcube['num_cells']} gcells in total.\n"
            )

        return gcube

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

    def find_scube_structure(self, target_n_cells, tolerance_n_cells=5,
                             max_extra=10, eps=0.015):
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
        self.scube['particle_masses'] = np.zeros(self.scube['n_shells']) - 1

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

        # Find mass in outermost shell. N.B.: need to base this on inner
        # size, since the outer size of that shell does in general not
        # increase by the same factor (unless n_extra = 0)
        scell_size_outer = (
            self.scube['base_shell_l_inner'] *
            self.scube['l_ratio']**(self.scube['n_shells'] - 1) /
            (self.scube['n_cells'] + self.scube['n_extra'] - 2)
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
        zone2_mass_factor = (self.config['zone2_mpart_factor_per_mpc'] *
                             self.gcube['cell_size_mpc'])

        # Number of high resolution particles this rank will have.
        gcell_info = {
            'num_parts_per_cell': np.zeros(num_gcell_types, dtype=int) - 1,
            'num_baseparts_per_cell': np.zeros(num_gcell_types, dtype=int) - 1,
            'particle_masses': np.zeros(num_gcell_types) - 1,
            'num_cells': np.zeros(num_gcell_types, dtype=int) - 1,
            'num_types': num_gcell_types
        }

        # Find allowed range of loads for Zone-II gcells
        # (do this now already, because zone1_gcell_load is assumed to be
        # correct, i.e. the exact value that will actually be used).
        zone2_load_range = self.find_zone2_load_range(zone1_gcell_load)

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
                load_range = [0, sys.maxsize]
            else:
                zone_type = zone2_type
                reduction_factor = zone2_mass_factor * itype
                target_gcell_load = zone1_gcell_load / reduction_factor
                load_range = zone2_load_range

            # Apply quantization constraints to find actual particle number
            if zone_type == 'glass':
                gcell_load = find_nearest_glass_number(
                    target_gcell_load, self.config['glass_files_dir'],
                    allowed_range=load_range)
            else:
                gcell_load = find_nearest_cube(
                    target_gcell_load, allowed_range=load_range)

            # Explicitly check that the gcell load is as specified for Zone-I
            if itype == 0:
                if (np.abs(gcell_load - target_gcell_load) / target_gcell_load
                    > 1e-6):
                    raise Exception(
                        f"Zone-I load not as anticipated "
                        f"{gcell_load:.3e} vs. {target_gcell_load:.3e}."
                    )

            # If we apply regular oversampling by splitting particles,
            # this increases the gcell load for type 0
            gcell_info['num_baseparts_per_cell'][itype] = gcell_load
            if itype == 0 and self.config['generate_extra_dm_particles']:
                gcell_load *= (1 + self.config['dm_to_gas_number_ratio'])

            gcell_info['num_parts_per_cell'][itype] = gcell_load

            # Mass (fraction) of each particle in current gcell type
            mass_itype = self.gcube['cell_volume'] / gcell_load
            gcell_info['particle_masses'][itype] = mass_itype

        # Some safety checks
        if np.min(gcell_info['num_cells'] < 0):
            raise ValueError(f"Negative entries in gcell_info['num_cells']...")
        num_gcells_processed = comm.reduce(np.sum(gcell_info['num_cells']))
        if comm_rank == 0 and num_gcells_processed != self.gcube['num_cells']:
            raise ValueError(
                f"Processed {num_gcells_processed} gcells, but there are "
                f"{self.gcube['num_cells']}!"
            )

        if comm_size > 1:
            # Propagate num_parts_per_cell and num_baseparts_per_cell across
            # ranks if we are using MPI
            gcell_info['num_parts_per_cell'] = comm.allreduce(
                gcell_info['num_parts_per_cell'],
                lambda *x: np.max(x, axis=0)
            )
            gcell_info['num_baseparts_per_cell'] = comm.allreduce(
                gcell_info['num_baseparts_per_cell'],
                lambda *x: np.max(x, axis=0)
            )
            gcell_info['particle_masses'] = comm.allreduce(
                gcell_info['particle_masses'],
                lambda *x: np.max(x, axis=0)
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
            if comm_size > 1:
                comm.Reduce([num_parts_by_type, MPI.LONG],
                            [num_parts_by_type_all, MPI.LONG],
                            op = MPI.SUM, root = 0
                )
            else:
                num_parts_by_type_all = num_parts_by_type
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

    def find_zone2_load_range(self, zone1_gcell_load):
        """Work out the minimum and maximum allowed Zone-II particle loads."""
        min_zone2_load = self.config['min_gcell_load']
        if self.config['zone2_max_mpart_over_zone1'] is not None:
            mass_ratio = float(self.config['zone2_max_mpart_over_zone1'])
            min_zone2_load = max(min_zone2_load, zone1_gcell_load / mass_ratio)

        if self.config['zone2_max_mpart_msun'] is not None:
            gcell_mass_msun = (
                self.gcube['cell_volume'] * self.sim_box['mass_msun'])
            min_zone2_load = max(
                min_zone2_load,
                gcell_mass_msun / float(self.config['zone2_max_mpart_msun'])
            )

        # There is always a maximum: must be below the mass of Zone I.
        max_zone2_load = (
            zone1_gcell_load / float(self.config['zone2_min_mpart_over_zone1'])
        )

        # Deal with stupid case of minimum being above maximum...
        if min_zone2_load > max_zone2_load:
            raise ValueError(
                f"Invalid range for Zone II gcell loads: "
                f"{min_zone2_load} > {max_zone2_load}!"
            )

        return [min_zone2_load, max_zone2_load]

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
        highest_gcell_load = comm.allreduce(gcell_load_range[1], op=MPI.MAX)
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
                    f"   {neq / gcube['volume_mpc']:,.3f} particles cMpc^-3."
                )       

            print(f"Lowest gcell load is {lowest_gcell_load}.")
        self.scube['lowest_equiv_n_in_gcube'] = np.cbrt(
            lowest_gcell_load * gcube['num_cells'])

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
                f"--- Cosmic mean number of particles in selection region: "
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

    def generate_gcube_particles(self, gcells, parts, ts):
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
            np_all = self.nparts['zone1_all'] + self.nparts['zone2_all']
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

            # We don't have to do anything if there are no cells of this type
            # (can happen when running on MPI)
            if ncell_type == 0:
                continue

            gcell_kernel_size = (
                self.gcell_info['num_baseparts_per_cell'][itype])
            gcell_load_type = self.gcell_info['num_parts_per_cell'][itype]
            particle_mass_type = self.gcell_info['particle_masses'][itype]

            is_glass = ((itype == 0 and zone1_type == 'glass') or
                        (itype > 0 and zone2_type == 'glass'))
            if is_glass:
                kernel_raw = glass[gcell_kernel_size] - 0.5
            else:
                kernel_raw = make_uniform_grid(
                    num=gcell_kernel_size, centre=True)

            # Final kernel may only be found after replication
            kernel = None

            if not self.config['identify_gas'] or itype > 0:
                kernel_masses = np.zeros(gcell_load_type) + particle_mass_type
                if itype == 0:
                    self.gcell_info['zone1_m_dm'] = -1
                    self.gcell_info['zone1_m_gas'] = -1
                    self.gcell_info['zone1_gas_mips_mpc'] = -1
            else:
                if self.config['generate_extra_dm_particles']:
                    # Replicate the kernel structure a given number of times,
                    # and assign gas/DM status to the replications
                    # (analogous to Richings et al. 2021).
                    kernel, kernel_masses, mass_ptypes = self.replicate_kernel(
                        kernel_raw, self.config['dm_to_gas_number_ratio'],
                        self.config['extra_dm_particle_scheme']
                    )
                else:
                    # We assign gas particles right here
                    kernel_masses, mass_ptypes = self.identify_gas(
                        gcell_load_type)

                self.gcell_info['zone1_m_dm'] = mass_ptypes['dm']
                self.gcell_info['zone1_m_gas'] = mass_ptypes['gas']

                # Mean gas inter-particle separation (for metadata file)
                num_gas = np.count_nonzero(kernel_masses == mass_ptypes['gas'])
                self.gcell_info['zone1_gas_mips_mpc'] = (
                    self.gcube['cell_size'] * self.sim_box['l_mpc'] /
                    np.cbrt(num_gas)
                )

            # If we have not assigned a final kernel yet, use raw one.
            if kernel is None:
                kernel = kernel_raw

            cy.fill_gcells_with_particles(
                gcells['pos'][ind_type, :], kernel, parts['pos'],
                parts['m'], kernel_masses, num_parts_created
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

        ts.set_time('Generate gcube particles')
            
        # Scale coordinates to units of the simulation box size (i.e. such
        # that the edges of the simulation box (not gcube!) are at coordinates 
        # -0.5 and 0.5.
        gcube_range_inds = np.array((-0.5, 0.5)) * gcube['n_cells']
        gcube_range_boxfrac = (np.array((-0.5, 0.5)) * gcube['sidelength'])
        if comm_rank == 0:
            print(f"Re-scaling particle coordinates...", end='')
        rescale(parts['pos'][:, :num_parts_created],
                gcube_range_inds, gcube_range_boxfrac, in_place=True)
        ts.set_time('Re-scale gcube particle coordinates')
        if comm_rank == 0:
            print(f" done (took {ts.get_time():.3e} sec.)")
            
        # Do consistency checks in separate function for clarity.
        self._verify_gcube_region(parts, num_parts_created, gcube['volume'], ts)
        #ts.set_time('Verify gcube particles')
        if comm_rank == 0:
            print(f"--> Finished generating gcube particles in "
                  f"{ts.get_time():.3e} sec.")

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
        num_parts_per_gcell_type = self.gcell_info['num_baseparts_per_cell']
        glass_dir = self.config['glass_files_dir']
        
        for iitype, num_parts in enumerate(num_parts_per_gcell_type):
            if num_parts in glass or num_parts <= 0:
                continue
            if ((iitype == 0 and zone1_type != 'glass') or
                (iitype > 0 and zone2_type != 'glass')):
                continue
            glass[num_parts] = load_glass_from_file(num_parts, glass_dir)

        return glass

    def identify_gas(self, num_tot):
        """
        Identify gas and DM particles within a gcell kernel.

        Parameters
        ----------
        num_tot : int
            The total number of particles in the kernel.

        Returns
        -------
        masses : ndarray(float)
            The masses (in units of the total box mass) of each particle in
            the kernel.
        mass_ptypes : dict
            The masses for 'gas' and 'dm' particles, respectively (under these
            keys).

        """
        # Find total mass in DM and baryons in a gcell
        f_m_baryon = self.cosmo['OmegaBaryon'] / self.cosmo['Omega0']
        f_m_dm = 1.0 - f_m_baryon
        m_gcell_tot = self.gcube['cell_size']**3
        m_gcell_gas = m_gcell_tot * f_m_baryon
        m_gcell_dm = m_gcell_tot * f_m_dm

        # Calculate gas and DM particle numbers, depending on setting
        dm_n_factor = self.config['dm_to_gas_number_ratio']
        if dm_n_factor is not None:
            n_frac_dm = dm_n_factor / (1 + dm_n_factor)
            num_dm = int(np.rint(num_tot * n_frac_dm))
        elif self.config['dm_to_gas_mass_ratio'] is not None:
            fp_gas = 1. / self.config['dm_to_gas_mass_ratio']

            # From combining N = N_gas + N_DM and M_i = N_i * m_i
            num_dm = int(np.rint((m_gcell_dm * num_tot * fp_gas) /
                                 (m_gcell_gas + m_gcell_dm * fp_gas)))
        else:
            raise ValueError(
                "To identify gas particles in the particle load, you need to "
                "specify either the number of mass fractions!"
            )

        num_gas = num_tot - num_dm
        m_dm = m_gcell_dm / num_dm
        m_gas = m_gcell_gas / num_gas

        if comm_rank == 0:
            m_to_msun = self.sim_box['mass_msun']
            print(
                f"Zone I: m_dm =  {m_dm:.5e} ({m_dm * m_to_msun:.5e} M_Sun),\n"
                f"        m_gas = {m_gas:.5e} ({m_gas * m_to_msun:.5e} M_Sun)"
                f"\n        N_DM = {num_dm}, N_gas = {num_gas}, "
                f"N_DM / N_gas = {num_dm / num_gas:.2f}."
            )

        # Assign particles in the kernel to each group
        ind_dm = np.random.choice(num_tot, size=num_dm, replace=False)
        ptypes = np.zeros(num_tot, dtype=int)
        ptypes[ind_dm] = 1
        ind_gas = np.nonzero(ptypes == 0)[0]

        kernel_masses = np.zeros(num_tot)
        kernel_masses[ind_dm] = m_dm
        kernel_masses[ind_gas] = m_gas

        return kernel_masses, {'dm': m_dm, 'gas': m_gas}

    def replicate_kernel(self, kernel_orig, num_rep, scheme):
        """
        Replicate a kernel a given number of times and assign particle types.

        The original particle positions must be in the range [-0.5, 0.5].
        Each replication is offset by a certain amount from the original,
        details depending on the replication number and specified scheme.

        Parameters
        ----------
        kernel_orig : ndarray(float)
            The coordinates of the original particles. This is updated and
            will also contain the replications upon return.
        num_rep : int
            The number of replications to be generated.
        scheme : string, optional
            Specification of where the replications should be positioned
            relative to the original particles. [TO BE CONTINUED]

        Returns
        -------
        kernel_masses : ndarray(float)
            The masses (in units of the total simulation mass) of each kernel
            particle including replications.
        mass_ptypes : dict
            The masses for 'gas' and 'dm' particles, respectively (under these
            keys).

        """
        num_orig = kernel_orig.shape[0]
        num_final = int(num_orig * (1 + num_rep))

        # Find total mass in DM and baryons in a gcell
        f_m_baryon = self.cosmo['OmegaBaryon'] / self.cosmo['Omega0']
        f_m_dm = 1.0 - f_m_baryon
        m_gcell_tot = self.gcube['cell_size']**3
        m_gcell_gas = m_gcell_tot * f_m_baryon
        m_gcell_dm = m_gcell_tot * f_m_dm

        # Calculate gas and DM particle numbers, depending on setting
        dm_num_factor = num_rep
        num_frac_dm = dm_num_factor / (1 + dm_num_factor)
        num_dm = int(num_orig * dm_num_factor)
        num_gas = num_orig

        m_dm = m_gcell_dm / num_dm
        m_gas = m_gcell_gas / num_gas

        if comm_rank == 0:
            m_to_msun = self.sim_box['mass_msun']
            print(
                f"Zone I: m_dm =  {m_dm:.5e} ({m_dm * m_to_msun:.5e} M_Sun),\n"
                f"        m_gas = {m_gas:.5e} ({m_gas * m_to_msun:.5e} M_Sun)"
                f"\n        ==> m_dm / m_gas = {m_dm / m_gas:.3f}."
                f"\n        N_dm = {num_dm}, N_gas = {num_gas}"
                f"\n        ==> N_dm / N_gas = {num_dm / num_gas:.2f}."
            )

        kernel_masses = np.zeros(num_final)
        kernel_masses[num_orig :] = m_dm
        kernel_masses[: num_orig] = m_gas

        # That was the easy part, now replicate the particles...

        # First, make space (use -1 as filler value)
        kernel = np.zeros((num_final, 3)) - 1
        kernel[: num_orig, ...] = np.copy(kernel_orig)

        # To calculate the replication shifts, we need the mean inter-particle
        # separation in the kernel (in units of the gcell)
        ips = 1.0 / np.cbrt(num_orig)

        # Since each replication scheme is different, outsource them each
        # to their own function...
        m_ratio = m_dm / m_gas
        if num_rep == 1:
            kr.replicate_kernel_bcc(kernel, ips, num_orig, m_ratio)

        elif num_rep == 3:
            if scheme in [None, 'face']:
                kr.replicate_kernel_n3_faces(kernel, ips, num_orig, m_ratio)
            elif scheme == 'edge':
                kr.replicate_kernel_n3_edges(kernel, ips, num_orig, m_ratio)

        elif num_rep == 4:
            if scheme in [None, 'face']:
                kr.replicate_kernel_n4_faces(kernel, ips, num_orig, m_ratio)
            elif scheme == 'edge':
                kr.replicate_kernel_n4_edges(kernel, ips, num_orig, m_ratio)
            elif scheme == 'square':
                kr.replicate_kernel_subsquare(kernel, ips, num_orig, m_ratio)
            else:
                raise ValueError(
                    f"Invalid scheme '{scheme}' for {num_rep} replications!")

        elif num_rep == 5:
            if scheme in [None, 'octahedron']:
                kr.replicate_kernel_octahedron(kernel, ips, num_orig, m_ratio)
            else:
                raise ValueError(
                    f"Invalid scheme '{scheme}' for {num_rep} replications!")

        elif num_rep == 6:
            if scheme is None:
                kr.replicate_kernel_n6(kernel, ips, num_orig, m_ratio)
            else:
                raise ValueError(
                    f"Invalid scheme '{scheme}' for {num_rep} replications!")

        elif num_rep == 7:
            if scheme in [None, 'subcube']:
                kr.replicate_kernel_subcube(kernel, ips, num_orig, m_ratio)
            elif scheme == 'diamond':
                kr.replicate_kernel_diamond(kernel, ips, num_orig, m_ratio)
            else:
                raise ValueError(
                    f"Invalid scheme '{scheme}' for {num_rep} replications!")

        else:
            raise ValueError(f"Invalid number of replications ({num_rep})!")

        return kernel, kernel_masses, {'dm': m_dm, 'gas': m_gas}

    def _verify_gcube_region(self, parts, nparts_created, gvolume, ts):
        """
        Perform consistency checks and print info about high-res region.
        This is a subfunction of `populate_gcells()`.
        """
        ts.increase_time('Other')
        # Make sure that the coordinates are in the expected range
        if not self.config['generate_extra_dm_particles']:
            if np.max(np.abs(parts['pos'][:nparts_created])) > 0.5:
                raise ValueError("Invalid Zone I/II coordinate values!")
        ts.set_time('   _verify: check coordinate ranges')
        
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
        ts.set_time('   _verify: check mass fractions')
            
        # Find and print the centre of mass of the Zone I/II particles
        # (N.B.: slicing creates views, not new arrays --> no memory overhead)
        com = centre_of_mass_mpi(
            parts['pos'][: nparts_created], parts['m'][:nparts_created])
        ts.set_time('   _verify: check centre of mass')
        
        if comm_rank == 0:
            print(f"Centre of mass for high-res grid particles:\n"
                  f"[{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}] Mpc/h."
            )
            print(f"Finished verifying gcube in {ts.get_time():.3e} sec.")

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
        offset_zone3 = self.nparts['zone1_local'] + self.nparts['zone2_local']

        # No zone III ==> no problem. This may also be the case for zooms.
        if npart_all == 0:
            return

        if comm_rank == 0:
            print(f"\n---- Generating {npart_all:,} outer low-res particles "
                  f"(Zone III) ----\n")

        if npart_local > 0:
            cy.fill_scube_layers(self.scube, self.nparts,
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
        ts = TimeStamp()
        
        if comm_rank == 0:
            print("\nVerifying particles...")
        npart_local = self.nparts['tot_local']
        npart_global = self.nparts['tot_all']

        # Safety check on coordinates and masses
        if npart_local > 0:
            # Don't abs whole array to avoid memory overhead
            if not self.config['generate_extra_dm_particles']:
                if np.min(parts['pos']) < -0.5 or np.max(parts['pos']) > 0.5:
                    raise ValueError(
                        f"Particles outside allowed range (-0.5 -> 0.5)! "
                        f"\nMinimum coordinate is {np.min(parts['pos'])}, "
                        f"maximum is {np.min(parts['pos'])}."
                    )
            if np.min(parts['m']) < 0 or np.max(parts['m']) > 1:
                raise ValueError(
                    f"Masses should be in the range 0 --> 1, but we have "
                    f"min={np.min(parts['m'])}, "
                    f"max={np.max(parts['m'])}."
                )
        ts.set_time('Check pos and mass ranges')
            
        # Check that the total masses add up to 1.
        total_mass_fraction = comm.allreduce(np.sum(parts['m']))
        ts.set_time('Check cross MPI mass total')
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
        ts.set_time('Check mass error')
            
        # Find the (cross-MPI) centre of mass, should be near origin.
        com = centre_of_mass_mpi(parts['pos'], parts['m'])
        ts.set_time('Check centre of mass')
        if comm_rank == 0:
            print(f"   Full centre of mass (in units of the box "
                  f"size): [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
            if (np.max(np.abs(com)) > 0.1):
                print(
                    f"**********\n   WARNING!!! \n***********\n"
                    f"Centre of mass is offset suspiciously far from the "
                    f"geometric box centre!"
                )

            print(f"   Verification successful.")
        ts.set_time('Final messages')
        #ts.print_time_usage('Finished verify_particles')
        
            
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
        ts = TimeStamp()
        parts['pos'] += self.centre
        ts.set_time('Shift positions')
        
        # ... and then apply periodic wrapping (simple: range is 0 --> 1).
        parts['pos'] %= 1.0
        ts.set_time('Do periodic wrapping')
        
        #if comm_rank == 0:
        #    print(f"Shifted particles such that the Lagrangian centre of "
        #          f"high-res region is at\n   "
        #          f"{self.centre[0]:.3f} / {self.centre[1]:.3f} / "
        #          f"{self.centre[2]:.3f}."
        #    )

        ts.set_time('Print message')
        #ts.print_time_usage('Finished shift_particles')
        
            
    # ------------- Routines for saving the particle load -------------------

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
        ts = TimeStamp()
        output_formats = self.config['output_formats'].lower()
        save_as_hdf5 = 'hdf5' in output_formats
        save_as_fortran_binary = 'fortran' in output_formats
        # If we don't save in either format, we can stop right here.
        if not save_as_hdf5 and not save_as_fortran_binary:
            if comm_rank == 0:
                print("Not saving particles.")
            return

        if comm_rank == 0:
            print("\nSaving particles...")
        
        # Randomise arrays, if desired
        if randomize:
            if comm_rank == 0:
                print('Randomizing arrays...')
            idx = np.random.permutation(self.nparts['tot_local'])
            self.parts['pos'] = self.parts['pos'][idx, :]
            self.parts['m'] = self.parts['m'][idx]

        # Load balance across MPI ranks.
        ts.set_time('Output setup')
        self.repartition_particles()
        ts.set_time('Repartitioning particles')
        
        # Save particle load as a collection of HDF5 and/or Fortran files
        save_dir = f"{self.config['icgen_work_dir']}/particle_load"
        save_dir_hdf5 = save_dir + '/hdf5'
        save_dir_bin = save_dir + '/fbinary'

        if comm_rank == 0:
            if not os.path.exists(save_dir_hdf5) and save_as_hdf5:
                os.makedirs(save_dir_hdf5)
            if not os.path.exists(save_dir_bin) and save_as_fortran_binary:
                os.makedirs(save_dir_bin)

        # Split local particle load into appropriate number of files
        num_files_per_rank = self.find_fortran_file_split()
        num_files_all = num_files_per_rank * comm_size
        
        # Split particles over files.
        # N.B.: IC_Gen expects equal particle numbers except in last file.
        num_parts_local = self.parts['m'].shape[0]
        num_parts_per_file = num_parts_local // num_files_per_rank
        separation_indices = np.arange(
            0, num_parts_local + 1, num_parts_per_file)
        separation_indices[-1] = num_parts_local

        ts.set_time('Prepare saving')
        
        # Make sure to save no more than max_save files at a time.
        max_save = 50
        n_batch = int(np.ceil(comm_size / max_save))
        for ibatch in range(n_batch):
            if comm_rank % n_batch == ibatch:
                if save_as_hdf5:
                    hdf5_loc = f"{save_dir_hdf5}/PL.{comm_rank}.hdf5" 
                    self.save_local_particles_as_hdf5(hdf5_loc)

                if save_as_fortran_binary:
                    for iifile in range(num_files_per_rank):
                        ifile = comm_rank * num_files_per_rank + iifile
                        fortran_loc = f"{save_dir_bin}/PL.{ifile}"
                        start_index = separation_indices[iifile]
                        end_index = separation_indices[iifile+1]
                        self.save_local_particles_as_binary(
                            ifile, num_files_all, fortran_loc,
                            start_index, end_index
                        )

                if self.verbose:
                    print(
                        f"   [Rank {comm_rank}] Finished saving "
                        f"{self.nparts['tot_local']} local particles.")

            # Make all ranks wait so that we don't have too many writing
            # at the same time.
            comm.barrier()

        ts.set_time('Write output')
        return ts
        
    def repartition_particles(self):
        """Re-distribute particles across MPI ranks to achieve equal load."""
        if comm_size == 1:
            print("   Single MPI rank, no repartitioning necessary.")
            return

        ts = TimeStamp()
        num_part_all = self.nparts['tot_all']
        num_part_local = self.nparts['tot_local']
        num_part_min = comm.allreduce(num_part_local, op=MPI.MIN)
        num_part_max = comm.allreduce(num_part_local, op=MPI.MAX)

        # ** TODO **: move this to separate function prior to particle
        # generation. That way, we can immediately build the coordinate
        # arrays with the final size (if > locally generated size).

        if num_part_all % comm_size == 0:
            # all files have same number:
            num_part_desired = num_part_all // comm_size
            num_per_rank = np.ones(comm_size, dtype=int) * num_part_desired
        else:
            # distribute evenly as much as possible:
            num_part_desired = num_part_all // comm_size
            # how many extra do we have:
            num_part_extra = num_part_all % comm_size
            num_per_rank = np.ones(comm_size, dtype=int) * num_part_desired
            # all files have same number except last, and last file has excess
            # particles, as required by ic_gen:
            num_per_rank[-1] += num_part_extra
        
        if comm_rank == 0:
            n_per_rank = num_per_rank[0]**(1/3.)
            print(
                f"   Load balancing {num_part_all} particles across "
                f"   {comm_size} ranks ({n_per_rank:.2f}^3 per rank)...\n"
                f"   Current load ranges from "
                f"   {num_part_min} to {num_part_max} particles."
            )
        ts.set_time('Setup')

        self.parts['m'] = pf.repartition(
            self.parts['m'], num_per_rank, comm, comm_rank, comm_size)
        ts.set_time('Mass repartitioning')
        num_part_new = len(self.parts['m'])

        # Because we repartition the coordinates individually, we first
        # need to expand the array if needed
        if num_part_new > self.parts['pos'].shape[0]:
            self.parts['pos'] = np.resize(
                self.parts['pos'], (num_part_new, 3))
            self.parts['pos'][self.nparts['tot_local']: , : ] = -1
        ts.set_time('Coordinate array shift')
        for idim in range(3):
            self.parts['pos'][: num_part_new, idim] = pf.repartition(
            self.parts['pos'][: self.nparts['tot_local'], idim], num_per_rank,
            comm, comm_rank, comm_size
        )
        ts.set_time('Coordinate repartitioning')

        if num_part_new < self.nparts['tot_local']:
            self.parts['pos'] = self.parts['pos'][:num_part_new, :]
        ts.set_time('Coordinate cutting')
            
        self.nparts['tot_local'] = num_part_new

        if comm_rank == 0:
            print('Done with load balancing.')

        ts.set_time('Finishing')
        #ts.print_time_usage('Finished repartitioning')
            
    def find_fortran_file_split(self):
        """Work out how many Fortran files to write per rank."""
        max_num_per_file = self.config['max_numpart_per_file']
        num_files = int(
            np.ceil(self.parts['m'].shape[0] / max_num_per_file))
        num_files = comm.allreduce(num_files, op=MPI.MAX)
        if comm_rank == 0:
            print(f"   Will write {num_files} Fortran files per rank.")

        return num_files

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
            print(f"   Done saving local particles to '{save_loc}'.")

    def save_local_particles_as_binary(
        self, index, tot_num_files, save_loc, start=None, end=None):
        """
        Write (part of) local particle load to Fortran binary file `save_loc`.

        Parameters
        ----------
        index : int
            The (numerical) index of the file to write (starting from 0).
        tot_num_files : int
            The total number of files, across all ranks.
        save_loc : str
            The file to write data to.
        start : int
            The first index of local particle array to write to this file.
        end : int
            The one-beyond-last index of local particle array to write.

        """
        if start is None:
            start = 0
        if end is None:
            end = self.nparts['tot_local']
        num_in_file = end - start
        
        f = FortranFile(save_loc, mode="w")

        # Write first 4+8+4+4+4 = 24 bytes
        f.write_record(
            np.int32(num_in_file),
            np.int64(self.nparts['tot_all']),
            np.int32(index),          # Index of this file
            np.int32(tot_num_files),  # Total number of files
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
        f.write_record(self.parts['pos'][start:end, 0].astype(np.float64))
        f.write_record(self.parts['pos'][start:end, 1].astype(np.float64))
        f.write_record(self.parts['pos'][start:end, 2].astype(np.float64))
        f.write_record(self.parts['m'][start:end].astype("float32"))
        f.close()

        if comm_rank == 0:
            print(f"   Done saving local particles to '{save_loc}'.")

    def save_metadata(self):
        """Save the metadata to an HDF5 file."""
        ts = TimeStamp()

        if comm_rank != 0:
            return ts

        save_loc = f"{self.config['icgen_work_dir']}/particle_load_info.hdf5"
        m_to_msun = self.sim_box['mass_msun']
        descr_m = (
            "Particle masses at each level in this zone, in units of the "
            "total simulation box mass.")
        descr_m_msun = (
            "Particle masses at each level in this zone, in units of M_Sun.")

        with h5.File(save_loc, 'w') as f:
            g = f.create_group('Header')
            g.attrs.create('NumPart_total', self.nparts['tot_all'])
            g.attrs.create('NumPart_per_zone',
                           np.array((self.nparts['zone1_all'],
                                     self.nparts['zone2_all'],
                                     self.nparts['zone3_all']))
            )
            g.attrs.create('SimulationBoxLength_Mpc', self.sim_box['l_mpc'])
            g.attrs.create('SimulationMass_MSun', self.sim_box['mass_msun'])
            g.attrs.create('NumPart_Equiv', self.sim_box['num_part_equiv'])
            g.attrs.create('N_Part_Equiv', self.sim_box['n_part_equiv'])
            g.attrs.create('NumBasePart_Equiv',
                self.sim_box['num_basepart_equiv'])
            g.attrs.create('N_BasePart_Equiv',
                self.sim_box['n_basepart_equiv'])
            g.attrs.create('CentreInParent', self.centre)
            
            for key in self.cosmo:
                g.attrs[key] = self.cosmo[key]

            g = f.create_group('ZoneI')
            g.attrs['m_gas'] = self.gcell_info['zone1_m_gas']
            g.attrs['m_gas_msun'] = self.gcell_info['zone1_m_gas'] * m_to_msun
            g.attrs['m_dm'] = self.gcell_info['zone1_m_dm']
            g.attrs['m_dm_msun'] = self.gcell_info['zone1_m_dm'] * m_to_msun
            g.attrs['MeanInterParticleSeparation_gas_Mpc'] = (
                self.gcell_info['zone1_gas_mips_mpc'])
            ds = g.create_dataset(
                'DMO_ParticleMasses',
                data=self.gcell_info['particle_masses'][0:1])
            ds.attrs.create('Description', descr_m)

            ds = g.create_dataset(
                'DMO_ParticleMasses_MSun',
                data=self.gcell_info['particle_masses'][0:1] * m_to_msun)
            ds.attrs.create('Description', descr_m_msun)

            if not self.config['is_zoom']:
                ts.set_time('Save metadata')
                return ts

            g = f.create_group('ZoneII')
            ds = g.create_dataset(
                'DMO_ParticleMasses',
                data=self.gcell_info['particle_masses'][1:])
            ds.attrs.create('Description', descr_m)
            ds = g.create_dataset(
                'DMO_ParticleMasses_MSun',
                data=self.gcell_info['particle_masses'][1:] * m_to_msun)
            ds.attrs.create('Description', descr_m_msun)

            g = f.create_group('ZoneIII')
            g.attrs.create('N_Cells', self.scube['n_cells'])
            g.attrs.create(
                'N_Cells_Outer', self.scube['n_cells'] + self.scube['n_extra'])
            g.attrs.create('N_Shells', self.scube['n_shells'])

            ds = g.create_dataset('DMO_ParticleMasses',
                                  data=self.scube['particle_masses'])
            ds.attrs.create('Description', descr_m)
            ds = g.create_dataset(
                'DMO_ParticleMasses_MSun',
                data=self.scube['particle_masses'] * m_to_msun)
            ds.attrs.create('Description', descr_m_msun)
            ds.attrs.create('LeapMass', self.scube['leap_mass'])

        ts.set_time('Save metadata')
        print(f"Saved particle load metadata to '{save_loc}'.")
        return ts
        
    # ------------- Routines for generating IC_Gen info -------------------

    def create_param_and_submit_files(
        self, create_param=True, create_submit=True):
        """
        Create appropriate parameter and submit files for IC-GEN and SWIFT.
        """
        codes = self.extra_params['code_types']
        print(f"\nPrepare information for code(s) '{codes}'...")

        fft_params = self.compute_fft_params()
        n_cores_icgen = self.get_icgen_core_number(fft_params)
        
        all_params = self.compile_param_dict(fft_params, n_cores_icgen)

        if create_param:
            pr.make_all_param_files(all_params, codes.lower())
        if create_submit:
            pr.make_all_submit_files(all_params, codes.lower())

    def compute_fft_params(self):
        """
        Work out what FFT grid dimensions we need for IC-Gen.

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
        
        num_part_box = self.sim_box['num_part_equiv']
        n_part_box = int(np.rint(np.cbrt(num_part_box)))
        
        # Zoom vs. non-zoom
        if self.config['is_zoom']:
            l_hr = (self.extra_params['icgen_fft_to_gcube_ratio'] *
                      self.gcube['sidelength'])
            if l_hr > 1:
                raise ValueError(
                f"Buffered zoom region is too big ({l_hr:.3e})!")
            num_eff = int(np.rint(num_part_box * l_hr**3))
            l_hr_mpc = l_hr * self.sim_box['l_mpc']

        else:
            num_eff = num_part_box
            l_hr = 1.0
            l_hr_mpc = self.sim_box['l_mpc']
        
        # Are we doing a multi-grid setup (only possible for zooms)?
        if self.use_multi_grid(l_hr):
            self.extra_params['icgen_multigrid'] = True

            # Find smallest grid that fully encloses the HR region
            # (grids shrink by successive factors of 2)
            n_levels = 1 + int(np.log(1/l_hr) / np.log(2))
            l_inner_mesh = 1 / (2**(n_levels-1))
            l_inner_mesh_mpc = l_inner_mesh * self.sim_box['l_mpc']
            n_eff_fft = int(np.ceil(np.cbrt(num_part_box * l_inner_mesh**3)))

        else:
            self.extra_params['icgen_multigrid'] = False
            l_inner_mesh_mpc = l_hr_mpc
            n_eff_fft = n_part_box
            n_levels = 1

        # Find next-largest allowed FFT size
        n_fft = self.find_fft_mesh_size(n_eff_fft)

        f_hr = l_hr
        print(
            f"   High-res grid: L={l_hr_mpc:.2f} Mpc "
            f"({100*l_hr**3:.2f}% of box volume).\n"
            f"   {n_levels} FFT levels, innermost grid has "
            f"L={l_inner_mesh_mpc} Mpc.\n"
            f"   FFT mesh: n={n_fft} (n_eff_part = {n_eff_fft}, "
            f"ratio = {n_fft / n_eff_fft:.2f})."

        )
        fft = {
            'n_mesh': n_fft,
            'num_eff_hr': num_eff,
            'l_hr_mpc': l_hr_mpc, 
        } 
        return fft

    def use_multi_grid(self, l_hr):
        """Check whether we use a multi-grid in IC_Gen."""
        if not self.config['is_zoom']:
            return False
        
        if l_hr > 0.5:
            print(f"*** Cannot use multigrid ICs, HR region size is "
                  f"{l_hr:.2f} > 0.5.")
            return False

        return True

    
    def find_fft_mesh_size(self, n_part):
        """
        Calculate the number of points per dimension for the FFTW mesh.

        Parameters
        ----------
        n_part : int
            The (effective) number of particles per dimension over the
            extent of the mesh.

        Returns
        -------
        n_fftw : int
            The side length of the FFTW mesh.

        """
        # Find the minimum FFT grid size that is at least a factor
        # `fft_min_Nyquist_factor` larger than the effective number of
        # particles per dimension
        f_Nyquist = self.extra_params['fft_min_Nyquist_factor']
        n_fft_base = self.extra_params['fft_n_base']
        n_fft_min = self.extra_params['fft_n_min']
        
        n_fft_required = max((n_part * f_Nyquist), n_fft_min)

        pow2 = int(np.ceil(np.log(n_fft_required / n_fft_base) / np.log(2)))
        n_fft = n_fft_base * 2**pow2

        return n_fft

    def get_icgen_core_number(self, fft_params):
        """
        Determine number of cores to use based on memory requirements.
        Number of cores must also be a factor of ndim_fft.

        Parameters
        ----------
        fft_params : dict
            The parameters of the high-res FFT grid

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

        # Apply a safety margin around the maximum values allowed by IC-Gen.
        # This might help prevent 'STATE' errors...
        num_max_part = int(self.extra_params['icgen_nmaxpart'])
        num_max_disp = int(self.extra_params['icgen_nmaxdisp'])

        n_dim_fft = fft_params['n_mesh']
        num_cores_per_node = int(self.extra_params['num_cores_per_node'])

        # Find number of cores that satisfies both FFT and particle constraints
        # Checking both independently may seem counter-intuitive, but is
        # correct because the constraint is the statically allocated memory
        # for the mesh and particle storage separately.
        num_cores_from_fft = n_dim_fft**2 * (n_dim_fft + 2) / num_max_disp
        num_cores_from_npart = self.nparts['tot_all'] / num_max_part
        num_cores_from_fft = int(np.ceil(num_cores_from_fft))
        num_cores_from_npart = int(np.ceil(num_cores_from_npart))

        num_cores = max(num_cores_from_fft, num_cores_from_npart)
        num_cores = self.find_allowed_core_number(
            num_cores, n_dim_fft, num_cores_per_node)

        print(f"   {num_cores} cores (need {num_cores_from_fft} for FFT, "
              f"{num_cores_from_npart} for particles).")

        return num_cores

    def compute_optimal_ic_mem(self, n_fft):
        """
        Compute the optimal particle and FFT number per MPI rank.

        Prints the numbers that optimize the required number of cores, as
        indication for possible custom-compilations of IC_Gen.
        
        Parameters
        ----------
        n_fft : int
            The side length of the FFT grid

        Returns
        -------
        None
        """
        bytes_per_particle = 66
        bytes_per_fft_cell = 20
 
        num_parts = self.nparts['tot_all']
        num_disps = n_fft**2 * (n_fft + 2)
        total_memory = ((bytes_per_particle * num_parts) +
                        (bytes_per_fft_cell * num_disps))

        num_cores = total_memory / self.extra_params['memory_per_core']
        num_cores = find_allowed_core_number(
            num_cores, n_fft, self.extra_params['num_cores_per_node'])
        
        max_parts = num_parts / num_cores
        max_disps = num_disps / num_cores
        print(f"--- Optimal max_parts={max_parts}, max_disps = {max_disps}")

        if (max_parts <= self.extra_params['icgen_nmaxpart'] * 0.95 and
            max_disps <= self.extra_params['icgen_nmaxdisp'] * 0.95):
            print("Standard parameters are sufficient.")

    def find_allowed_core_number(self, num_cores, n_fft, num_cores_per_node):
        """Find an allowed number of cores for a given target value."""

        # Increase n_cores until it is an integer divisor of n_dim_fft
        while (n_fft % num_cores) != 0:
            num_cores += 1

        # If we're using one node, try to use as many of the cores as possible
        # (but still such that ndim_fft is an integer multiple). Since the
        # starting `ncores` works, we will reduce n_cores at most to that.
        if num_cores < num_cores_per_node:
            num_cores = num_cores_per_node
            while (n_fft % num_cores) != 0:
                num_cores -= 1

        return num_cores
                    
    def compile_param_dict(self, fft_params, n_cores_icgen):
        """Compile a dict of all parameters required for param/submit files."""

        extra_params = self.extra_params
        cosmo_h = self.cosmo['hubbleParam']
        param_dict = {}

        # Compute mass cuts between particle types.
        cut_type1_type2 = 0.0    # Dummy value if not needed
        cut_type2_type3 = 0.0    # Dummy value if not needed
        if self.config['is_zoom']:
            num_species = extra_params['icgen_num_species']
            if num_species >= 2:
                cut_type1_type2 = np.log10(
                    np.mean(self.gcell_info['particle_masses'][0:2]))
                print(f"   log10 mass fraction cut from parttype 1 --> 2 = "
                      f"   {cut_type1_type2:.2f}")
            if num_species == 3:
                cut_type2_type3 = np.log10(
                    (self.gcell_info['particle_masses'][-1] +
                     self.scube['m_min']) / 2
                )
                print(f"   log10 mass fraction cut from parttype 2 --> 3 = "
                      f"   {cut_type2_type3:.2f}")
            if num_species > 3:
                print(f"   **NumSpecies > 3 not supported, ignoring extra.**")

        l_box_mpchi = self.sim_box['l_mpc'] * cosmo_h
        centre_mpchi = self.centre * l_box_mpchi

        param_dict['sim_name'] = self.config['sim_name']
        param_dict['box_size_mpchi'] = l_box_mpchi
        param_dict['centre_x_mpchi'] = centre_mpchi[0]
        param_dict['centre_y_mpchi'] = centre_mpchi[1]
        param_dict['centre_z_mpchi'] = centre_mpchi[2]
        param_dict['l_gcube_mpchi'] = self.gcube['sidelength_mpc'] * cosmo_h
        param_dict['is_zoom'] = self.config['is_zoom']

        for key in self.cosmo:
            param_dict[f'cosmo_{key}'] = self.cosmo[key]

        param_dict['ics_z_init'] = extra_params['z_initial']

        # IC-Gen specific parameters
        param_dict['icgen_exec'] = extra_params['icgen_exec']
        if extra_params['icgen_module_setup'] is None:
            param_dict['icgen_module_setup'] = ''
        else:
            param_dict['icgen_module_setup'] = (
                f"source {extra_params['icgen_module_setup']}")
        param_dict['icgen_work_dir'] = self.config['icgen_work_dir']
        param_dict['icgen_n_cores'] = n_cores_icgen
        param_dict['icgen_runtime_hours'] = extra_params['icgen_runtime_hours']
        param_dict['icgen_use_PH_ids'] = extra_params['icgen_use_PH_IDs']
        param_dict['icgen_PH_nbit'] = extra_params['icgen_PH_nbit']
        param_dict['icgen_num_species'] = extra_params['icgen_num_species']
        param_dict['icgen_cut_t1t2'] = cut_type1_type2
        param_dict['icgen_cut_t2t3'] = cut_type2_type3
        param_dict['icgen_linear_powspec_file'] = (
            f"{extra_params['icgen_powerspec_dir']}/"
            f"{self.cosmo['linear_powerspectrum_file']}"
        )
        param_dict['icgen_panphasian_descriptor'] = (
            extra_params['panphasian_descriptor'])
        param_dict['icgen_n_part_for_uniform_box'] = (
            self.sim_box['num_part_equiv'])
        param_dict['icgen_num_constraints'] = (
            extra_params['icgen_num_constraints'])
        for key_suffix in ['', '2', '_path', '_levels', '2_path', '2_levels']:
            param_dict[f'icgen_constraint_phase_descriptor{key_suffix}'] = (
            extra_params[f'icgen_constraint_phase_descriptor{key_suffix}'])
        param_dict['icgen_multigrid'] = extra_params['icgen_multigrid']
        param_dict['icgen_n_fft_mesh'] = fft_params['n_mesh']
        param_dict['icgen_highres_num_eff'] = fft_params['num_eff_hr']
        param_dict['icgen_highres_n_eff'] = np.cbrt(fft_params['num_eff_hr'])
        param_dict['icgen_highres_l_mpchi'] = fft_params['l_hr_mpc']*cosmo_h
        
        param_dict['slurm_partition'] = extra_params['slurm_partition']
        param_dict['slurm_account'] = extra_params['slurm_account']
        param_dict['slurm_email'] = extra_params['slurm_email']
        param_dict['num_cores_per_node'] = extra_params['num_cores_per_node']

        param_dict['icgen_p6_multipoles'] = extra_params['icgen_p6_multipoles']

        return param_dict       

    def find_gas_splitting_mass(self):
        """Compute the mass threshold for splitting gas particles."""
        f_baryon = self.cosmo['OmegaBaryon'] / self.cosmo['Omega0']
        m_tot_gas = self.sim_box['mass_msun'] * f_baryon
        m_tot_gas /= (self.cosmo['hubbleParam'] * 1e10)  # to 1e10 M_Sun [no h]
        return m_tot_gas / self.sim_box['num_basepart_equiv'] * 4

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


def find_nearest_glass_number(
    num, glass_files_dir, allowed_range=[1, sys.maxsize]):
    """Find the closest number of glass particles in a file to `num`.""" 
    files = os.listdir(glass_files_dir)
    num_in_files = [int(_f.split('_')[2]) for _f in files if 'ascii' in _f]
    num_in_files = np.array(num_in_files, dtype='i8')

    # Check to make sure that we don't have unrealistic expectations
    if np.max(num_in_files) < allowed_range[0]:
        raise ValueError(
            f"There are no glass files above the minimum allowed load of "
            f"{allowed_range[0]} (highest: {np.max(num_in_files)})!"
        )
    if np.min(num_in_files) > allowed_range[1]:
        raise ValueError(
            f"There are no glass files below the maximum allowed load of "
            f"{allowed_range[1]} (lowest: {np.min(num_in_files)})!"
        )

    # We limit the list of available files to those in the correct load range
    ind_allowed_files = np.nonzero(
        (num_in_files >= allowed_range[0]) & (num_in_files <= allowed_range[1])
    )[0]
    if len(ind_allowed_files) == 0:
        raise Exception(
            "Something is wrong - could not find any allowed glass files.")
    num_in_files = num_in_files[ind_allowed_files]

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
    if comm_size == 1:
        ret_array = np.unique(a) if unicate else a.copy()
        return ret_array

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

def find_nearest_cube(num, allowed_range=[1, sys.maxsize]):
    """
    Find the cube number closest to num.

    Parameters
    ----------
    allowed_range : [int, int], optional
        The return value must lie in this range.

    Returns
    -------
    cube : int
        The nearest cube integer to num.

    """
    allowed_range = [find_next_cube(allowed_range[0]),
                     find_previous_cube(allowed_range[1])
    ]
    root_range = np.rint(np.cbrt(allowed_range)).astype(int)

    root = np.clip(np.cbrt(num), root_range[0], root_range[1])
    root_low = int(np.floor(root))
    root_high = int(np.ceil(root))

    cube_low = root_low**3
    cube_high = root_high**3

    if np.abs(cube_low - num) < np.abs(cube_high - num):
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
    ts = TimeStamp()
    # Form local CoM first, to avoid memory overheads...
    com_local = np.average(coords, weights=masses, axis=0)
    ts.set_time('Local average')
    m_tot = np.sum(masses)
    ts.set_time('Local sum')
    
    # ... then form cross-MPI (weighted) average of all local CoMs.
    com_x = comm.reduce(com_local[0] * m_tot)
    com_y = comm.reduce(com_local[1] * m_tot)
    com_z = comm.reduce(com_local[2] * m_tot)
    m_tot = comm.reduce(m_tot)
    ts.set_time('MPI')
    #if comm_rank == 0:
    #    ts.print_time_usage('Finished centre_of_mass_mpi')
    
    if comm_rank == 0:
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
    elif name == 'DES3yr':
        cosmo['Omega0'] = 0.306
        cosmo['OmegaLambda'] = 0.694
        cosmo['OmegaBaryon'] = 0.0486
        cosmo['hubbleParam'] = 0.681
        cosmo['sigma8'] = 0.807
        cosmo['linear_powerspectrum_file'] = 'power_spec_from_CLASS_colibre.txt'
    elif name == 'DES3yr_neutrinos':
        cosmo['Omega0'] = 0.306078
        cosmo['OmegaLambda'] = 0.693922
        cosmo['OmegaBaryon'] = 0.0486
        cosmo['hubbleParam'] = 0.681
        cosmo['sigma8'] = 0.807
        cosmo['linear_powerspectrum_file'] = 'power_spec_from_CLASS_colibre.txt'       
    else:
        raise ValueError(f"Invalid cosmology '{name}'!")

    cosmo['OmegaDM'] = cosmo['Omega0'] - cosmo['OmegaBaryon']

    return cosmo


def parse_arguments():
    """Parse the input arguments into a structure."""

    parser = argparse.ArgumentParser(
	description="Set up the particle load for a zoom simulation.")
    parser.add_argument(
        'param_file', help='Parameter file with settings.')
    parser.add_argument(
        '-p', '--params',
        help='[Optional] Override one or more entries in the parameter file.'
             'The format is name1: value1[, name2: value2, ...]'
    )
    parser.add_argument(
        '-d', '--dry_run',
        help='Dry run -- only calculate the number of particles and memory '
             'requirements, but do not actually generate particles.',
        action='store_true'
    )

    args = parser.parse_args()

    # Process parameter override values here...
    args.params = utils.process_param_string(args.params)

    # Some sanity checks
    if not os.path.isfile(args.param_file):
        raise OSError(f"Could not find parameter file {args.param_file}!")

    return args


if __name__ == '__main__':
    args = parse_arguments()
    ParticleLoad(args)
