"""Computation of synthesized sound fields."""

import numpy as np
from .source import point


def generic(omega, x0, d, x, y, z, c=None, source=point):
    """Compute sound field for a generic driving function"""
    d = np.squeeze(np.asarray(d))
    if len(d) != len(x0):
        raise ValueError("length mismatch")
    p = 0
    for weight, position in zip(d, x0):
        if weight != 0:
            p += weight * source(omega, position, x, y, z, c)
    return p