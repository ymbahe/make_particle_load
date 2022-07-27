import numpy as np
import pytest

from tools.utils import process_param_string

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
