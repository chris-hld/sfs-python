"""Compute the sound field generated by a sound source"""

import numpy as np
from scipy import special
from .. import util


def point(omega, x0, grid, c=None):
    """Point source.

    Eq.(#ynr) in [Wierstorf et al, 2015]::

                    1  e^(-j w/c |x-x0|)
      G(x-xs, w) = --- -----------------
                   4pi      |x-x0|

    """
    k = util.wavenumber(omega, c)
    x0 = util.asarray_1d(x0)
    grid = util.asarray_of_arrays(grid)

    r = np.linalg.norm(grid - x0)
    return np.squeeze(1/(4*np.pi) * np.exp(-1j * k * r) / r)


def line(omega, x0, grid, c=None):
    """Line source parallel to the z-axis.

    Note: third component of x0 is ignored.

    Eq.(#54d) in [Wierstorf et al, 2015]::

                         (2)
      G(x-xs, w) = -j/4 H0  (w/c |x-x0|)

    """
    k = util.wavenumber(omega, c)
    x0 = util.asarray_1d(x0)
    x0 = x0[:2]  # ignore z-component
    grid = util.asarray_of_arrays(grid)

    r = np.linalg.norm(grid[:2] - x0)
    p = -1j/4 * special.hankel2(0, k * r)
    # If necessary, duplicate in z-direction:
    gridshape = np.broadcast(*grid).shape
    if len(gridshape) > 2:
        p = np.tile(p, [1, 1, gridshape[2]])
    return np.squeeze(p)


def plane(omega, x0, n0, grid, c=None):
    """Plane wave.

    Eq.(#vmd) in [Wierstorf et al, 2015]::

      G(x, w) = e^(-j w/c n x)

    """
    k = util.wavenumber(omega, c)
    x0 = util.asarray_1d(x0)
    n0 = util.asarray_1d(n0)
    grid = util.asarray_of_arrays(grid)

    return np.squeeze(np.exp(-1j * k * np.inner(grid - x0, n0)))
