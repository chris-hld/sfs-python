"""Weights (tapering) for the driving function.

.. plot::
    :context: reset

    import sfs
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['figure.figsize'] = 8, 3  # inch
    plt.rcParams['axes.grid'] = True

    active1 = np.zeros(101, dtype=bool)
    active1[5:-5] = True

    # The active part can wrap around from the end to the beginning:
    active2 = np.ones(101, dtype=bool)
    active2[30:-10] = False

"""
import numpy as np


def none(active):
    """No tapering window.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.none(active1))
        plt.axis([-3, 103, -0.1, 1.1])

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.none(active2))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    return active


def kaiser(active, beta):
    """Kaiser tapering window.

    This uses :func:`numpy.kaiser`.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.kaiser(active1, 0), label='β = 0')
        plt.plot(sfs.tapering.kaiser(active1, 2), label='β = 2')
        plt.plot(sfs.tapering.kaiser(active1, 6), label='β = 6')
        plt.plot(sfs.tapering.kaiser(active1, 8.6), label='β = 8.6')
        plt.plot(sfs.tapering.kaiser(active1, 14), label='β = 14')
        plt.axis([-3, 103, -0.1, 1.1])
        plt.legend(loc='lower center')

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.kaiser(active2, 7))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    idx = _windowidx(active)
    window = np.zeros(len(active))
    window[idx] = np.kaiser(len(idx), beta)
    return window


def tukey(active, alpha):
    """Tukey tapering window.

    This uses a function similar to :func:`scipy.signal.tukey` except
    that the first and last value are not zero.

    Examples
    --------
    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.tukey(active1, 0), label='α = 0')
        plt.plot(sfs.tapering.tukey(active1, 0.25), label='α = 0.25')
        plt.plot(sfs.tapering.tukey(active1, 0.5), label='α = 0.5')
        plt.plot(sfs.tapering.tukey(active1, 0.75), label='α = 0.75')
        plt.plot(sfs.tapering.tukey(active1, 1), label='α = 1')
        plt.axis([-3, 103, -0.1, 1.1])
        plt.legend(loc='lower center')

    .. plot::
        :context: close-figs

        plt.plot(sfs.tapering.tukey(active2, 0.3))
        plt.axis([-3, 103, -0.1, 1.1])

    """
    idx = _windowidx(active)
    alpha = np.clip(alpha, 0, 1)
    if alpha == 0:
        return none(active)
    # design Tukey window
    x = np.linspace(0, 1, len(idx) + 2)
    tukey = np.ones_like(x)
    first_part = x < alpha / 2
    tukey[first_part] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[first_part] - alpha / 2)))
    third_part = x >= (1 - alpha / 2)
    tukey[third_part] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[third_part] - 1 + alpha / 2)))
    # fit window into tapering function
    result = np.zeros(len(active))
    result[idx] = tukey[1:-1]
    return result


def _windowidx(active):
    """Return list of connected indices for window function.

    Note: Gaps within the active part are not allowed.

    """
    # find index where active loudspeakers begin (works for connected contours)
    if (active[0] and not active[-1]) or np.all(active):
        first_idx = 0
    else:
        first_idx = np.argmax(np.diff(active.astype(int))) + 1
    # shift generic index vector to get a connected list of indices
    idx = np.roll(np.arange(len(active)), -first_idx)
    # remove indices of inactive secondary sources
    return idx[:np.count_nonzero(active)]
