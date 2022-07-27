"""Test suite for make_mask.py module"""

import numpy as np
import pytest

from make_mask.make_mask import find_vertices, process_param_string


def test_vertices():
    r = np.array([
        [0.1, 4, 2.1],
        [-3.2, 5.5, 1.0],
        [9.0, 2.1, -3.3],
        [0.5, 4.1, 3.0]
    ])
    box = find_vertices(r, serial_only=True)

    np.testing.assert_allclose(
        box, np.array([[-3.2, 2.1, -3.3], [9.0, 5.5, 3.0]]))
    
    return

test_input = ['name1: value1',
              'name1:value1, name2:value2',
              'name1:value1,name2: value2 , name3:value3',
              'name1: value1, name2: 10, name3: -7.3',
              None,
              '',
]
expected = [{'name1': 'value1'},
            {'name1': 'value1', 'name2': 'value2'},
            {'name1': 'value1', 'name2': 'value2', 'name3': 'value3'},
            {'name1': 'value1', 'name2': 10, 'name3': -7.3},
            {},
            {},
]
@pytest.mark.parametrize(
    "test_input,expected",
    [(_str, _dict) for _str, _dict in zip(test_input, expected)]
)
def test_string_parse(test_input, expected):
    param_dict = process_param_string(test_input)
    assert param_dict == expected
