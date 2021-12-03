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
    ic_metadata = get_ic_metadata(args.ics_file)

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
        'ics_file',
        help="The name of the final, SWIFT-compatible ICs file."
    )
    parser.add_argument(
        '-d', '--run_dir',
        help="Directory in which to run the simulation."
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
        help="Type of simulation to run: DMO [default] or EAGLE."
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

    args = parser.parse_args()
    if args.run_dir is None:
        raise ValueError("Must specify the run directory!")
    if args.sim_type is None:
        raise ValueError("Must specify the simulation type!")
    
    args.sim_type = args.sim_type.lower()
    if args.sim_type not in ['dmo', 'eagle']:
        raise ValueError(
            "Simulation type '{args.sim_type}' is not (yet) supported.")
    
    if args.param_template is None:
        if args.sim_type == 'dmo':
            args.param_template = './swift_templates/params_dmo.yml'
        elif args.sim_type == 'eagle':
            args.param_template = './swift_templates/params_eagle.yml'
    if args.sim_name is None:
        args.sim_name = args.ics_file.split('/')[-1].replace('.hdf5', '')
    if args.run_time < 0.5:
        print(f"Overriding input run time to 0.6 hours.")
        args.run_time = 0.6
        
    return args


def get_ic_metadata(ics_file):
    """Extract the relevant metadata from the ICs file."""
    data = {}
    with h5.File(ics_file, 'r') as f:
        g = f['Metadata']
        for key in g.attrs.keys():
            data[key] = g.attrs[key]

        h = f['Header']
        data['HubbleParam'] = h.attrs['HubbleParam']
        data['AExp_ICs'] = h.attrs['Time']
        data['BoxSize'] = h.attrs['BoxSize']

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

    if args.sim_type == 'sibelius':
        params['swift_extra_flags'] = '--fof params.yml'
    else:
       params['swift_extra_flags'] = ''

    return params
        
def set_up_rundir(args):
    """Set up the base directory for the SWIFT run."""
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    copy(args.output_time_file, f"{run_dir}/output_times.dat")
    if not os.path.isdir(run_dir + '/logs'):
        os.makedirs(run_dir + '/logs')

    
def generate_param_file(data, args):
    """Adapt and write the SWIFT parameter file."""
    base_file = args.param_template
    params = yaml.safe_load(open(base_file))
    is_hydro = args.sim_type.lower() in ['eagle', 'hydro']

    # Now adjust all the required parameters...
    cosmo = params['Cosmology']
    cosmo['h'] = float(data['HubbleParam'])
    cosmo['a_begin'] = float(data['AExp_ICs'])
    cosmo['Omega_cdm'] = float(data['OmegaDM'])
    cosmo['Omega_lambda'] = float(data['OmegaLambda'])
    cosmo['Omega_b'] = float(data['OmegaBaryon'])

    params['Scheduler']['max_top_level_cells'] = compute_top_level_cells()

    params['Snapshots']['output_list'] = 'output_times.dat'
    if args.vr:
        params['Snapshots']['invoke_stf'] = 1
        params['swift_flags'] += ' --velociraptor'
        
    params['Restarts']['max_run_time'] = args.run_time - 0.5

    gravity = params['Gravity']
    mean_ips = data['BoxSize'] / data['N_Part_Equiv']

    gravity['comoving_DM_softening'] = float(mean_ips) * (1/20)
    gravity['max_physical_DM_softening'] = float(mean_ips) * (1/50)
    gravity['mesh_side_length'] = compute_mesh_side_length()
    if is_hydro:
        fac = data['dm_to_baryon_mass_ratio']**(-1/3)
        gravity['comoving_baryon_softening'] = mean_ips * (1/20) * fac
        gravity['max_physical_baryon_softening'] = mean_ips * (1/50) * fac

    ics = params['InitialConditions']
    ics['file_name'] = args.ics_file
    ics['cleanup_h_factors'] = 1 if data['ics_contain_h_factors'] else 0
    if is_hydro:
        ics['generate_gas_in_ics'] = 0 if data['ics_include_gas'] else 0

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
        

def compute_top_level_cells():
    """Compute the optimal number of top-level cells per dimension."""
    # Place holder for now...
    return 10


def compute_mesh_side_length():
    """Compute the optimal side length of the Gravity FFT mesh."""
    return 256


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
