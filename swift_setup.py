"""Script to set up a SWIFT simulation for specified ICs."""

import numpy as np
import argparse
import os
import h5py as h5
import yaml
from shutil import copy
from tools.param_file_routines import make_custom_copy
from local import local

from pdb import set_trace


# TODO: update N_Part_Equiv to reflect DM particles only when in dual mode.

def main():
    """Set up SWIFT run."""
    args = parse_arguments()

    # Extract required data from ICs file.
    ic_metadata = get_ic_metadata(args)

    # Add additional required parameters
    params = compile_params(ic_metadata, args)
    
    # Set up run directory
    set_up_rundir(args)

    # Adapt and write simulation parameter file
    generate_param_file(params, args)

    # Adapt (re-/)submit scripts
    generate_submit_scripts(params, args)


def parse_arguments():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Set up a SWIFT run for given ICs."
    )

    # ICs file and run dir are non-optional    
    parser.add_argument(
        'run_dir',
        help="Directory in which to run the simulation."
    )
    parser.add_argument(
        '-i', '--ics_file',
        help="The name of the final, SWIFT-compatible ICs file."
    )
    
    parser.add_argument(
        '-s', '--sim_name',
        help='Name of the simulation. If not supplied, the name of the '
             'ICs file will be used.'
    )
    parser.add_argument(
        '-vr', action='store_true', help='Run with VR on the fly?')
    parser.add_argument(
        '-t', '--sim_type', default='dmo',
        help="Type of simulation to run: DMO [default], EAGLE, or COLIBRE."
    )
    parser.add_argument(
        '-g', '--swift_gas', action='store_true',
        help='Split off gas within SWIFT (default: no).'
    )
    parser.add_argument(
        '-p', '--param_template',
        help="Template parameter file to adapt, if different from generic "
             "template for the specified simulation type."
    )
    parser.add_argument(
        '-n', '--num_nodes', type=int, default=1,
        help="Number of nodes on which to run (default: 1)."
    )
    parser.add_argument(
        '-r', '--run_time', type=float, default=72,
        help="Job run time [hours]. SWIFT will stop half an hour earlier.")
    parser.add_argument(
        '-x', '--exec_dir', default='../../swiftsim/builds/std_vr',
        help="Directory containing the SWIFT executable, either absolute or "
             "relative to run_dir (default: ../../builds/std_vr)."
    )
    parser.add_argument(
        '-m', '--module_file', default='~/.module_load_swift',
        help="File to source in order to set up the SWIFT environment "
             "(default: ~/.module_load_swift)."
    )
    parser.add_argument(
        '--output_time_file', default='./swift_templates/output_times.dat',
        help='Output times file to use (default: '
             './swift_templates/output_times.dat).'
    )
    parser.add_argument(
        '--slurm_template', default='./swift_templates/slurm_script',
        help='Template file for SLURM job submission (default: '
             './swift_templates/slurm_script).'
    )
    parser.add_argument(
        '-y', '--table_dir', default='../..',
        help='Directory containing SWIFT tables for EAGLE-like runs '
             '(default: ../..).'
    )
    
    args = parser.parse_args()
    if args.ics_file is None:
        raise ValueError("Must specify the run directory!")
    if args.sim_type is None:
        raise ValueError("Must specify the simulation type!")

    args.sim_type = args.sim_type.lower()
    if args.sim_type not in ['dmo', 'eagle', 'colibre']:
        raise ValueError(
            "Simulation type '{args.sim_type}' is not (yet) supported.")
    if args.swift_gas and args.sim_type not in ['eagle', 'colibre']:
        raise ValueError(
            f"Cannot run gas simulation with type '{args.sim_type}'!")
    
    if args.param_template is None:
        if args.sim_type == 'dmo':
            args.param_template = './swift_templates/params_dmo.yml'
        elif args.sim_type == 'eagle':
            args.param_template = './swift_templates/params_eagle.yml'
        elif args.sim_type == 'colibre':
            args.param_template = './swift_templates/params_colibre.yml'
    if args.sim_name is None:
        args.sim_name = args.ics_file.split('/')[-1].replace('.hdf5', '')
    if args.run_time < 0.5:
        print(f"Overriding input run time to 0.6 hours.")
        args.run_time = 0.6
        
    return args


def get_ic_metadata(args):
    """Extract the relevant metadata from the ICs file."""
    ics_file = args.ics_file

    data = {}
    with h5.File(ics_file, 'r') as f:
        g = f['Metadata']
        for key in g.attrs.keys():
            data[key] = g.attrs[key]

        h = f['Header']
        data['HubbleParam'] = h.attrs['HubbleParam']
        data['AExp_ICs'] = h.attrs['Time']

        # We need the box size to find the mean inter-particle separation
        # and hence softening --> take out h factor
        data['BoxSize'] = h.attrs['BoxSize'] / data['HubbleParam']
        
        try:
            p0 = f['PartType0']
            data['is_hydro'] = True
            data['num_gas'] = h.attrs['NumPart_Total'][0]
            data['num_dm'] = h.attrs['NumPart_Total'][1]
            print(f"Found gas particles in ICs.")
            if args.swift_gas:
                raise ValueError(
                    "Cannot generate gas if it is already there!")
            if args.sim_type not in ['eagle', 'colibre']:
                raise ValueError(
                    "Cannot run simulations with gas in type "
                    f"'{args.sim_type}'!"
                )
        except KeyError:
            data['is_hydro'] = False
            print("ICs do not contain gas particles.")
    
        data['m_pt1'] = f['PartType1']['Masses'][0] / data['HubbleParam']
        print(f"Read PT1 mass as {data['m_pt1']}")
            
    set_default(data, 'dm_to_baryon_mass_ratio',
                data['OmegaDM'] / data['OmegaBaryon'])
    set_default(data, 'ics_contain_h_factors', 1)
    set_default(data, 'ics_include_gas', 0)

    return data


def compile_params(ic_data, args):
    """
    Compile the full param dict.

    Parameters
    ---------
    ic_data : dict
        Parameters read in from the ICs file.
    args : obj
        The input parameters.

    Returns
    -------
    params : dict
        Dict containing all required parameters.

    """
    params = {}
    for key in ic_data:
        params[key] = ic_data[key]

    params['slurm_num_nodes'] = args.num_nodes
    params['slurm_ntasks_per_node'] = 1 if args.num_nodes == 1 else 2
    params['sim_name'] = args.sim_name
    params['slurm_partition'] = local['slurm_partition']
    params['slurm_account'] = local['slurm_account']
    params['slurm_email'] = local['slurm_email']
    params['slurm_time_string'] = get_time_string(args.run_time)
    
    params['module_setup_command'] = (
        '' if args.module_file is None else f'source {args.module_file}')

    if args.num_nodes == 1:
        params['slurm_mpi_command'] = ''
        swift_exec = 'swift'
        params['threads_per_task'] = local['cpus_per_node']
    else:
        params['slurm_mpi_command'] = f'mpirun -np $$SLURM_NTASKS'
        swift_exec = 'swift_mpi'
        params['threads_per_task'] = int(local['cpus_per_node'] / 2)
    params['swift_exec'] = args.exec_dir + '/' + swift_exec

    if args.sim_type in ['dmo', 'sibelius']:
        params['swift_flags'] = '--self-gravity'
    elif args.sim_type == 'eagle':
        params['swift_flags'] = '--eagle'
    elif args.sim_type == 'colibre':
        params['swift_flags'] = '--colibre --dust'
        
    if args.sim_type == 'sibelius':
        params['swift_extra_flags'] = '--fof params.yml'
    else:
       params['swift_extra_flags'] = ''

    params['ics_have_gas'] = ic_data['is_hydro']
       
    return params
        
def set_up_rundir(args):
    """Set up the base directory for the SWIFT run."""
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    copy(args.output_time_file, f"{run_dir}/output_times.dat")
    copy('./swift_templates/vrconfig.cfg', run_dir)
    if not os.path.isdir(run_dir + '/logs'):
        os.makedirs(run_dir + '/logs')

    
def generate_param_file(data, args):
    """Adapt and write the SWIFT parameter file."""
    base_file = args.param_template
    params = yaml.safe_load(open(base_file))
    
    # Now adjust all the required parameters...
    params['MetaData']['run_name'] = data['sim_name']

    cosmo = params['Cosmology']
    cosmo['h'] = float(data['HubbleParam'])
    cosmo['a_begin'] = float(data['AExp_ICs'])
    cosmo['Omega_cdm'] = float(data['OmegaDM'])
    cosmo['Omega_lambda'] = float(data['OmegaLambda'])
    cosmo['Omega_b'] = float(data['OmegaBaryon'])

    params['Scheduler']['max_top_level_cells'] = compute_top_level_cells(data)

    params['Snapshots']['output_list'] = 'output_times.dat'
    if args.vr:
        params['Snapshots']['invoke_stf'] = 1
        data['swift_flags'] += ' --velociraptor'
    else:
        params['Snapshots']['invoke_stf'] = 0
        
    params['Restarts']['max_run_time'] = args.run_time - 0.5

    gravity = params['Gravity']
    mean_ips = float(data['BoxSize'] / data['N_Part_Equiv'])
    print(f"Box size = {data['BoxSize']:.2f} Mpc, "
          f"N_Part_Equiv = {data['N_Part_Equiv']}, "
          f"Mean IPS = {mean_ips:2e} Mpc."
    )

    # Base line: simplest case (only DM)
    gravity['comoving_DM_softening'] = mean_ips * (1/20)
    gravity['max_physical_DM_softening'] = mean_ips * (1/50)

    # If we have gas in the ICs, need to adjust softenings
    if data['ics_have_gas']:
        m_gas = float(data['m_gas'])
        m_av = data['m_gas'] * data['num_gas'] + data['m_dm'] * data['num_dm']
        m_av /= (data['num_gas'] + data['num_dm'])
        f_dm = float(np.cbrt(data['m_dm'] / m_av))
        f_gas = float(np.cbrt(data['m_gas'] / m_av))
        gravity['comoving_baryon_softening'] = mean_ips * f_gas * (1/20)
        gravity['max_physical_baryon_softening'] = mean_ips * f_gas * (1/50)
        gravity['comoving_DM_softening'] = mean_ips * f_dm * (1/20)
        gravity['max_physical_DM_softening'] = mean_ips * f_dm * (1/50)

    # Finally, if we will generate gas later within SWIFT:
    if args.swift_gas:
        m_gas = float(data['m_pt1'] * data['OmegaBaryon'] / data['Omega0'])
        f_dm = float(np.cbrt(data['OmegaDM'] / data['Omega0']))
        f_gas = float(np.cbrt(data['OmegaBaryon'] / data['Omega0']))
        gravity['comoving_baryon_softening'] = mean_ips * f_gas * (1/20)
        gravity['max_physical_baryon_softening'] = mean_ips * f_gas * (1/50)
        gravity['comoving_DM_softening'] = mean_ips * f_dm * (1/20)
        gravity['max_physical_DM_softening'] = mean_ips * f_dm * (1/50)
        
    gravity['mesh_side_length'] = compute_mesh_side_length(data)
    if gravity['mesh_side_length'] > 1290:
        gravity['distributed_mesh'] = 1
    
    ics = params['InitialConditions']
    ics['file_name'] = args.ics_file
    ics['cleanup_h_factors'] = 1 if data['ics_contain_h_factors'] else 0
    ics['generate_gas_in_ics'] = 1 if args.swift_gas else 0

    if args.sim_type in ['eagle', 'colibre']:
        params['SPH']['particle_splitting_mass_threshold'] = float(m_gas * 4)
        print(f"Set splitting threshold to {m_gas * 4}")
        
        params['Stars']['luminosity_filename'] = (
            args.table_dir + '/photometry')

    if args.sim_type in ['eagle']:
        params['EAGLECooling']['dir_name'] = (
            args.table_dir + '/coolingtables/')
        params['EAGLEFeedback']['filename'] = (
            args.table_dir + '/yieldtables/')
        params['EAGLEAGN']['min_gas_mass_for_nibbling'] = float(m_gas / 2)
        params['COLIBRECooling']['dir_name'] = (
            args.table_dir + '/UV_dust1_CR1_G1_shield1.hdf5')
        
    if args.sim_type in ['colibre']:
        params['DustEvolution']['dust_yields_path'] = (
            args.table_dir + '/dust_yields')
        params['COLIBREFeedback']['filename'] = (
            args.table_dir + '/yieldtables/')
        params['COLIBREFeedback']['earlyfb_filename'] = (
            args.table_dir + '/Early_stellar_feedback.hdf5')
        params['COLIBREAGN']['min_gas_mass_for_nibbling'] = float(m_gas / 2)
        params['COLIBRECooling']['dir_name'] = (
            args.table_dir + '/cooling_files_new')
        
        
    # Write the modified param file to the run dir.
    out_file = f"{args.run_dir}/params.yml"
    with open(out_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)


def generate_submit_scripts(data, args):
    """Adapt and write the SLURM submit scripts."""
    
    submit_file = f"{args.run_dir}/submit.sh"
    resubmit_file = f"{args.run_dir}/resubmit.sh"
    
    make_custom_copy(args.slurm_template, submit_file, data, executable=True)
    data['swift_flags'] += ' --restart'
    make_custom_copy(args.slurm_template, resubmit_file, data, executable=True)
    copy('./swift_templates/auto_resubmit', args.run_dir)
        

def compute_top_level_cells(data):
    """
    Compute the optimal number of top-level cells per dimension.

    For now, this is scaled with the number of particles per dimension.
    Might need to refine this for e.g. zooms...
    
    """
    n_eq = data['N_Part_Equiv']
    ncells = int(np.rint(n_eq / 376 * 16))
    return max(3, ncells)


def compute_mesh_side_length(data):
    """Compute the optimal side length of the Gravity FFT mesh."""

    n_eq = data['N_Part_Equiv']
    nmesh_targ = n_eq / 376 * 256
    nmesh_pow = int(np.rint(np.log(nmesh_targ) / np.log(2)))
    return 2**nmesh_pow


def set_default(dictionary, key, value):
    """Set a key in a dict to a given value if it does not exist yet."""
    if key not in dictionary:
        dictionary[key] = value


def get_time_string(time):
    int_hrs = int(np.floor(time))

    mins = (time - int_hrs) * 60
    int_mins = int(np.floor(mins))

    secs = (mins - int_mins) * 60
    int_secs = int(np.floor(secs))

    return f'{int_hrs:02d}:{int_mins:02d}:{int_secs:02d}'
    

if __name__ == "__main__":
    main()
    print("Done!")
