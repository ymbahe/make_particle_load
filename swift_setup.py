"""Script to set up a SWIFT simulation for specified ICs."""

import numpy as np
import argparse
import os
import h5py
import yaml
from shutil import copy
from tools.param_file_routines import make_custom_copy

def main():
    """Set up SWIFT run."""
    args = parse_arguments()

    # Extract required data from ICs file.
    ic_metadata = get_ic_metadata(args.ics_file)

    # Set up run directory
    set_up_rundir(args)

    # Adapt and write simulation parameter file
    generate_params(ic_metadata, args)

    # Adapt (re-/)submit scripts
    generate_submit_scripts(ic_metadata, args)


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
        '-v', '--vr_dir',
        help='Directory in which Velociraptor library is stored. If not '
             'given, on-the-fly VR will not be run.'
    )
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
        '-x', '--exec_dir', default='../../builds/std_vr'
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

    if args.param_template is None:
        if args.sim_type == 'dmo':
            args.param_template = './swift_templates/params_dmo.yml'
        elif args.sim_type == 'eagle':
            args.param_template = './swift_templates/params_eagle.yml'

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


def set_up_rundir(args):
    """Set up the base directory for the SWIFT run."""
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    copy(args.output_time_file, f"{run_dir}/snapshot_times.txt")


def generate_params(data, args):
    """Adapt and write the SWIFT parameter file."""
    base_file = args.param_template
    params = yaml.safe_load(open(base_file))
    is_hydro = args.sim_type.lower() in ['eagle', 'hydro']

    # Now adjust all the required parameters...
    cosmo = params['Cosmology']
    cosmo['h'] = data['HubbleParam']
    cosmo['a_begin'] = data['AExp_ICs']
    cosmo['Omega_cdm'] = data['OmegaDM']
    cosmo['Omega_lambda'] = 1.0 - data['Omega0']
    cosmo['Omega_b'] = data['OmegaBaryon']

    params['Scheduler']['max_top_level_cells'] = compute_top_level_cells()

    params['Snapshots']['output_list'] = 'snapshot_times.dat'
    if args.vr_dir is not None:
        params['Snapshots']['invoke_stf'] = 1

    params['Restarts']['max_run_time'] = max(args.max_run_time - 0.5)

    gravity = params['Gravity']
    mean_ips = data['BoxSize'] / data['N_DM_equiv']

    gravity['comoving_DM_softening'] = mean_ips * (1/20)
    gravity['max_physical_DM_softening'] = mean_ips * (1/50)
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
    submit_file = f"{args.run_dir}/submit.sh"

    make_custom_copy(args.slurm_template, data, submit_file, executable=True)
    data['swift_flags'].append(' --restart')
    make_custom_copy(args.slurm_template, data, resubmit_file, executable=True)


def compute_top_level_cells():
    """Compute the optimal number of top-level cells per dimension."""
    # Place holder for now...
    return 200


def compute_mesh_side_length():
    """Compute the optimal side length of the Gravity FFT mesh."""
    return 256


def set_default(dictionary, key, value):
    """Set a key in a dict to a given value if it does not exist yet."""
    if key not in dictionary:
        dictionary[key] = value


if __name__ == "__main__":
    main()
    print("Done!")
