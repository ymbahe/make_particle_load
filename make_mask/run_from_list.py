r"""
Script to batch process multiple haloes from one parent simulation.

Run as:
```
python3 run_from_list.py \
   -t [name of parameter template file] \
   -l [name of file containing the group numbers to process]
```
"""
import argparse
import os
import copy
from numpy import loadtxt, ndarray
from yaml import safe_load as load
from typing import Tuple
from make_mask import MakeMask


def make_masks_from_list() -> None:
    """Main wrapper function to generate the masks."""
    parser = argparse.ArgumentParser(
        description="Script to batch process multiple haloes from the same "
                    "parent simulation."
    )
    parser.add_argument(
        '-t', '--template_file',
        help="The name of the template parameter file to generate the mask, "
             "relative to the directory containing this script."
    )
    parser.add_argument(
        '-l', '--list_file',
        help="The name of the file containing the group numbers to process, "
             "relative to the directory containing this script."
    )
    args = parser.parse_args()

    if args.template_file is None:
        raise AttributeError("You must specify a template file!")
    if args.list_file is None:
        raise AttributeError("You must specify a group list file!")

    # Parse the parameter (template), and check the expected placeholders
    params = load(open(args.template_file))
    params['select_from_vr'] = True
    if 'GROUPNUMBER' not in params['fname']:
        raise ValueError(
            "'fname' parameter in the template file must contain the "
            f"placeholder 'GROUPNUMBER', but is '{params['fname']}'!"
        )

    group_numbers, sorter = get_group_numbers_list(args.list_file)
    if sorter is not None and 'sort_rule' in params:
        if sorter.lower() != params['sort_rule'].lower():
            print(
                "\n***********************************************\n"
                f"WARNING: list file appears to refer to sort_rule "
                f"'{sorter}', but parameter file specifies sort_type "
                f"'{params['sort_rule']}'. Overriding parameter file..."
                "\n**************************************************\n"
            )
            params['sort_rule'] = sorter.lower()

    n_halo = len(group_numbers)
    for ii, group_number in enumerate(group_numbers):
        params_grp = copy.deepcopy(params)
        params_grp['fname'] = (
            params_grp['fname'].replace('GROUPNUMBER', str(group_number)))
        params_grp['group_number'] = group_number

        print("\n--------------------------------------------")
        print(f"Starting to generate mask for halo {group_number} "
              f"({ii+1}/{n_halo})")
        print("--------------------------------------------\n")

        MakeMask(params=params_grp)


def get_output_dir(params: dict) -> str:
    """Find output directory and create it if it does not yet exist."""
    output_dir = params['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_group_numbers_list(list_file: str) -> Tuple[ndarray, str]:
    """Load the list of group numbers and, if present, sorting rule."""
    # Check whether the first line contains a header indicating the type
    # of sorter (M200c, M500c)
    with open(list_file) as f:
        header = f.readline()
        if header[0] == '#':
            if 'm200crit' in header.lower():
                sorter = 'm200crit'
            elif 'm500crit' in header.lower():
                sorter = 'm500crit'
            elif 'none' in header.lower():
                sorter = 'none'
            else:
                sorter = None

    return loadtxt(list_file).astype(int), sorter


if __name__ == '__main__':
    make_masks_from_list()
