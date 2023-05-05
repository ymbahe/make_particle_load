"""
Collection of functions to create parameter and submit files consistent with
the generated particle load, for IC-Gen, SWIFT, and/or Gadget3/4.
"""

import re
import os
import subprocess
from string import Template
from shutil import copy

template_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    'particle_load',
    'templates'
)

# -------------------- Main public functions -------------------------------

def make_all_param_files(params, codes):
    """Wrapper to generate all required param files."""
    if 'ic_gen_6p9' in codes:
        make_param_file_for_icgen_6p9(params)
    if 'ic_gen_8p4' in codes:
        make_param_file_for_icgen_8p4(params)
    if 'swift' in codes:
        make_param_file_for_swift(params)
    if 'gadget3' in codes:
        make_param_file_for_gadget3(params)
    if 'gadget4' in codes:
        make_param_file_for_gadget4(params)


def make_all_submit_files(params, codes):
    """Wrapper to generate all required submit files."""
    if 'ic_gen_6p9' in codes:
        make_submit_file_for_icgen_6p9(params)
    if 'ic_gen_8p4' in codes:
        make_submit_file_for_icgen_8p4(params)
    if 'swift' in codes:
        make_submit_file_for_swift(params)
    if 'gadget3' in codes:
        make_submit_file_for_gadget3(params)
    if 'gadget4' in codes:
        make_submit_file_for_gadget4(params)


# ---------------- Functions for individual target codes ---------------------

# .............................. IC-GEN 6.9/8.4 ...............................

def make_submit_file_for_icgen_6p9(params):
    """Make slurm submission script for IC-Gen."""

    # Make folder if it doesn't exist.
    icgen_work_dir = f"{params['icgen_work_dir']}"
    create_dir_if_needed(icgen_work_dir)

    make_custom_copy(
        f"{template_dir}/ic_gen/submit", f"{icgen_work_dir}/submit.sh",
        params, executable=True
    )
    print("Generated submit file for IC-Gen 6.9.")

def make_param_file_for_icgen_6p9(params):
    """Make a parameter file for IC-Gen."""

    icgen_work_dir = f"{params['icgen_work_dir']}"

    # Make output folder for the ICs (should already be done in main code!)
    icgen_output_dir = f"{icgen_work_dir}/ICs/"
    create_dir_if_needed(icgen_output_dir)

    # What are the constraint phase descriptors?
    if params['icgen_constraint_phase_descriptor'] != '%dummy':
        if params['icgen_constraint_phase_descriptor2'] != '%dummy':
            params['icgen_is_constraint'] = 2
        else:
            params['icgen_is_constraint'] = 1
    else:
        params['icgen_is_constraint'] = 0

    if params['icgen_num_constraints'] < 1:
        params['icgen_constraint_phase_descriptor'] = '%dummy'
        params['icgen_constraint_phase_descriptor_path'] = '%dummy'
        params['icgen_constraint_phase_descriptor_levels'] = '%dummy'
    if params['icgen_num_constraints'] < 2:
        params['icgen_constraint_phase_descriptor2'] = '%dummy'
        params['icgen_constraint_phase_descriptor2_path'] = '%dummy'
        params['icgen_constraint_phase_descriptor2_levels'] = '%dummy'

    # Is this a zoom simulation? Then we cannot use 2LPT
    if params['is_zoom']:
        params['icgen_2lpt_type'] = 0 if params['icgen_multigrid'] else 1
        params['icgen_is_multigrid'] = 1 if params['icgen_multigrid'] else 0
    else:
        params['icgen_highres_l_mpchi'] = 0.0
        params['icgen_highres_n_eff'] = 0
        params['icgen_2lpt_type'] = 1
        params['icgen_is_multigrid'] = 0

    # Use Peano-Hilbert indexing?
    params['icgen_indexing'] = 2 if params['icgen_use_PH_ids'] else 1

    make_custom_copy(
        f"{template_dir}/ic_gen/params.inp", f"{icgen_work_dir}/params.inp",
        params
    )

    print("Generated parameter file for IC-Gen 6.9.")

def make_submit_file_for_icgen_8p4(params):
    """Make slurm submission script for IC-Gen."""

    # Make folder if it doesn't exist.
    icgen_work_dir = f"{params['icgen_work_dir']}"
    create_dir_if_needed(icgen_work_dir)

    make_custom_copy(
        f"{template_dir}/ic_gen/submit", f"{icgen_work_dir}/submit.sh",
        params, executable=True
    )
    print("Generated submit file for IC-Gen 8.4.")

def make_param_file_for_icgen_8p4(params):
    """Make a parameter file for IC-Gen."""

    icgen_work_dir = f"{params['icgen_work_dir']}"

    # Make output folder for the ICs (should already be done in main code!)
    icgen_output_dir = f"{icgen_work_dir}/ICs/"
    create_dir_if_needed(icgen_output_dir)

    # What are the constraint phase descriptors?
    if params['icgen_constraint_phase_descriptor'] != '%dummy':
        if params['icgen_constraint_phase_descriptor2'] != '%dummy':
            params['icgen_is_constraint'] = 2
        else:
            params['icgen_is_constraint'] = 1
    else:
        params['icgen_is_constraint'] = 0

    if params['icgen_num_constraints'] < 1:
        params['icgen_constraint_phase_descriptor'] = '%dummy'
        params['icgen_constraint_phase_descriptor_path'] = '%dummy'
        params['icgen_constraint_phase_descriptor_levels'] = '%dummy'
    if params['icgen_num_constraints'] < 2:
        params['icgen_constraint_phase_descriptor2'] = '%dummy'
        params['icgen_constraint_phase_descriptor2_path'] = '%dummy'
        params['icgen_constraint_phase_descriptor2_levels'] = '%dummy'

    # Is this a zoom simulation? Then we cannot use 2LPT
    if params['is_zoom']:
        params['icgen_2lpt_type'] = 0 if params['icgen_multigrid'] else 1
        params['icgen_is_multigrid'] = 1 if params['icgen_multigrid'] else 0
    else:
        params['icgen_highres_l_mpchi'] = 0.0
        params['icgen_highres_n_eff'] = 0
        params['icgen_2lpt_type'] = 1
        params['icgen_is_multigrid'] = 0

    # Use Peano-Hilbert indexing?
    params['icgen_indexing'] = 2 if params['icgen_use_PH_ids'] else 1

    make_custom_copy(
        f"{template_dir}/ic_gen/params_8p4.inp", f"{icgen_work_dir}/params.inp",
        params
    )

    print("Generated parameter file for IC-Gen 8.4.")

# .............................. SWIFT ......................................

def make_submit_file_for_swift(params):
    """Make a SLURM submission script file for SWIFT."""

    run_dir = f"{params['swift_run_dir']}"
    create_dir_if_needed(run_dir)

    submit_template = f"{template_dir}/swift/submit"
    resub_template = f"{template_dir}/swift/resubmit"

    if params['sim_type'] == 'eaglexl':
        params['swift_options'] = '--eagle'
    elif params['sim_type'] in ['dmo', 'sibelius']:
        params['swift_options'] = '--self-gravity'
    else:
        raise ValueError(f"Illegal sim_type {params['sim_type']}!")

    if params['sim_type'] == 'sibelius':
        params['extra_swift_options'] = '--fof params.yml'
    else:
        params['extra_swift_options'] = ''

    make_custom_copy(
        submit_template, f"{run_dir}/submit", params, executable=True)
    make_custom_copy(
        resub_template, f"{run_dir}/resubmit", params, executable=True)

    # Also create (executable) auto-resubmit script
    with open(f"{run_dir}/auto_resubmit", 'w') as f:
        f.write('sbatch resubmit')
    os.chmod(f"{run_dir}/auto_resubmit", 0o744)

    print("Generated submit file for SWIFT.")


def make_param_file_for_swift(params):
    """Make a parameter file for SWIFT."""

    sim_type = params['sim_type'].lower()

    # Make the run and output directories
    run_dir = f"{params['swift_run_dir']}"
    create_dir_if_needed(run_dir + '/out_files')

    # Replace values.
    if 'tabula_' in sim_type:
        raise Exception("Fix this one")
        #r = ['%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.3f'%(softening_ratio_background),
        #'%.8f'%(eps_baryon_physical/h), '%.8f'%(eps_dm_physical/h), fname]

        #subprocess.call(f"cp {template_dir}/templates/swift/%s/select_output.yml %s"%\
        #        (template_set, data_dir), shell=True)
    elif sim_type == 'sibelius':
        copy(f"{template_dir}/swift/{sim_type}/select_output.yml", run_dir)
        copy(f"{template_dir}/swift/{sim_type}/stf_times_a.txt",
             f"{run_dir}/snapshot_times.txt")

    elif sim_type == 'eaglexl':
        copy(f"{template_dir}/swift/{sim_type}/select_output.yml", run_dir)
        copy(f"{template_dir}/swift/{sim_type}/output_times_a.txt",
             f"{run_dir}/snapshot_times.txt")

    elif sim_type == 'colibre':
        copy(f"{template_dir}/swift/{sim_type}/select_output.yml", run_dir)
        copy(f"{template_dir}/swift/{sim_type}/output_times_a.txt",
             f"{run_dir}/snapshot_times.txt")

    elif sim_type == 'dmo':
        copy(f"{template_dir}/swift/{sim_type}/select_output.yml", run_dir)
        copy(f"{template_dir}/swift/{sim_type}/output_times.txt",
             f"{run_dir}/snapshot_times.txt")

        #split_mass = gas_particle_mass / 10**10. * 4.
        #r = [fname, '%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.8f'%(eps_dm_physical/h),
        #'%.8f'%(eps_baryon_physical/h), '%.3f'%(softening_ratio_background), 
        #'%.8f'%split_mass, ic_dir, fname]
    else:
        set_trace()
        raise ValueError("Invalid template set")

    make_custom_copy(f"{template_dir}/swift/{sim_type}/params.yml",
                     f"{run_dir}/params.yml", params)
    print("Generated parameter file for SWIFT.")


# ........................... GADGET-3 .......................................

def make_submit_file_for_gadget3(params):
    """Make a SLURM submission script file for GADGET-3."""
    raise Exception("GADGET-3 submission files not yet implemented.")

def make_param_file_for_gadget3(params):
    """Make a SLURM submission script file for GADGET-3."""
    raise Exception("GADGET-3 parameter files not yet implemented.")


# ............................. GADGET-4 ....................................

def make_submit_file_for_gadget4(params):
    """Make a SLURM submission script file for GADGET-4."""
    raise Exception("GADGET-4 submission files not yet implemented.")

def make_param_file_for_gadget4(params):
    """Make a SLURM submission script file for GADGET-4."""
    raise Exception("GADGET-4 parameter files not yet implemented.")


# ===================== Low-level utility functions =========================

def make_custom_copy(source, target, params, executable=False):
    """
    Create a customized copy of a template file with the provided values.

    Parameters
    ----------
    source : str
        The path to the source (template) file. Customizable parameters must
        be preceded with a '$', and must have a corresponding key in params.
    target : str
        The path to the to-be-created customized copy of source.
    params : dict
        Dict containing the values for customizable parameters (keys that
        do not correspond to an entry in the source file are ignored).
    executable : bool
        Switch to make the output file executable (default: False).

    Returns
    -------
    None

    Notes
    -----
    See https://docs.python.org/3/library/string.html#template-strings for a
    detailed description of the substitution rules.

    """
    # Read and adjust input file, using values from `params`.
    with open(source, 'r') as f:
        src = Template(f.read())
        customized_copy = src.substitute(params)

    # Write the customized file.
    with open(target, 'w') as f:
        f.write(customized_copy)

    if executable:
        os.chmod(target, 0o744)

def create_dir_if_needed(path):
    """Create a specified directory and its parents if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
