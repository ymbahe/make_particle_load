"""Test suite for generate_particle_load script with different settings."""

import subprocess
import pytest

import glob

param_files = glob.glob('particle_load_tests/*.yml')

@pytest.mark.parametrize("param_file", param_files)
def test_gpl(param_file):
    succ = subprocess.call(
        ['python3', '../particle_load/generate_particle_load.py', param_file])
    print(succ)
    assert succ == 0  

# -----------------------------------------------
# Now test parameter variations for base model...
# -----------------------------------------------

param_variations = [
    "",
    "target_mass: 1e8",
    "target_mass_type: gas",
    "identify_gas: True",
    "identify_gas: True, generate_extra_dm_particles: True, dm_to_gas_number_ratio: 4",
    "identify_gas: True, generate_extra_dm_particles: True, dm_to_gas_number_ratio: 7, extra_dm_particle_scheme: diamond",    
]

@pytest.mark.parametrize("params", param_variations)
def test_gpl_variations(params):
    succ = subprocess.call(
        ['python3',
         '../particle_load/generate_particle_load.py',
         'particle_load_tests/test_l100_base.yml',
         f'-p {params}']
    )
    print(succ)
    assert succ == 0  
    

if __name__ == '__main__':
    test_gpl(param_files[0])
