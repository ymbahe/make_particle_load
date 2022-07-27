"""Test suite for make_mask.py module"""

import numpy as np
import pytest

from make_mask.make_mask import find_vertices

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
