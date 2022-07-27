"""Test suite for make_mask script with different settings."""

import subprocess
import pytest

import glob

param_files = glob.glob('make_mask_tests/*.yml')

@pytest.mark.parametrize("param_file", param_files)
def test_mm(param_file):
    succ = subprocess.call(
        ['python3', '../make_mask/make_mask.py', param_file])
    print(succ)
    assert succ == 0  


# -----------------------------------------------
# Now test parameter variations for base model...
# -----------------------------------------------

param_variations = [
    "",
    "padding_snaps: 5",
    "padding_snaps: 1 5",
    "padding_snaps: 1 5 3",
    "pad_ics_as_particles: True",
    "highres_padding_width: 2.5",
    "cell_padding_width: 5.0",
    "min_num_per_cell: 5",
]

@pytest.mark.parametrize("params", param_variations)
def test_mm_variations(params):
    succ = subprocess.call(
        ['python3',
         '../make_mask/make_mask.py',
         'make_mask_tests/test_l100_base.yml',
         f'-p {params}']
    )
    print(succ)
    assert succ == 0  


if __name__ == '__main__':
    test_mm(param_files[0])
