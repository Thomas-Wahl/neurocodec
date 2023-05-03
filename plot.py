import numpy
import control
from scipy import stats
from matplotlib.axes._axes import Axes
from numpy.typing import ArrayLike


def set_labels(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    xunit: str | None,
    yunit: str | None
):
    ax.set_xlabel(xlabel + ('' if xunit is None else f' [{xunit}]'))
    ax.set_ylabel(ylabel + ('' if yunit is None else f' [{yunit}]'))


def time_series(
    ax: Axes,
    resp: control.TimeResponseData,
    tunit: str | None = None,
    yunit: str | None = None,
    tbounds: tuple[float, float] | None = None,
    **kwargs
):
    set_labels(ax, 'time', 'signal', tunit, yunit)
    if tbounds is None:
        return ax.plot(resp.t, resp.y.T, **kwargs)
    tmin, tmax = tbounds
    mask = numpy.logical_and(tmin <= resp.t, resp.t < tmax)
    return ax.plot(resp.t[mask], resp.y[..., mask].T, **kwargs)


def bode(
    axmag: Axes,
    axphase: Axes,
    resp: control.FrequencyResponseData,
    funit: str | None = None,
    munit: str | None = None,
    logscale=True,
    degree=True,
    **kwargs
):
    if resp.ninputs > 1:
        raise ValueError(
            f'resp must have no more than 1 input: {resp.ninputs = }'
        )
    mag = resp.magnitude.squeeze()
    if logscale:
        mag = control.mag2db(mag)
        munit = 'dB'
    if degree:
        phase = numpy.rad2deg(numpy.unwrap(resp.phase.squeeze()))
        punit = '°'
    else:
        phase = numpy.unwrap(resp.phase.squeeze())
        punit = 'rad'
    if funit == 'Hz':
        freq = resp.frequency / (2. * numpy.pi)
    else:
        freq = resp.frequency
    set_labels(axmag, 'frequency', 'magnitude', funit, munit)
    set_labels(axphase, 'frequency', 'phase', funit, punit)
    return (
        axmag.plot(freq, mag.T, **kwargs),
        axphase.plot(freq, phase.T, **kwargs)
    )


def psd(
    ax: Axes,
    freq: ArrayLike,
    power: ArrayLike,
    funit: str | None = None,
    punit: str | None = None,
    logscale=True,
    **kwargs
):
    freq = numpy.atleast_1d(numpy.squeeze(freq))
    power = numpy.atleast_1d(power)
    # check dimensions
    if freq.ndim > 1:
        raise ValueError(f'freq must be 1-dimensional: {freq.ndim = }')
    if power.ndim > 2:
        raise ValueError(
            f'power must be at most 2-dimensional: {power.ndim = }'
        )
    if logscale:
        power = 10. * numpy.log10(power)
        punit = 'dB'
    set_labels(ax, 'frequency', 'power', funit, punit)
    return ax.plot(freq, power.T, **kwargs)


def activity(
    ax: Axes,
    activ: ArrayLike,
    compare: tuple[tuple[int, int]] = (),
    **kwargs
):
    activ = numpy.atleast_2d(activ)
    if activ.ndim > 2:
        raise ValueError(
            f'activ must be at most 2-dimensional: {activ.ndim = }'
        )
    mean = activ.mean(-1)
    sdev = numpy.sqrt(activ.var(-1))
    bars = ax.bar(range(len(mean)), mean, yerr=sdev, **kwargs)
    ymax = ax.get_ylim()[1] * 1.1
    ax.set_ylim(ymax=ymax)
    ax.set_xticks(())
    
    dy = ymax / 20.

    def annotate(x1: int, x2: int, text: str):
        y = max(mean[x1], mean[x2]) + dy
        x = (x1 + .1,) * 2 + (x2 - .1,) * 2
        ax.plot(x, (y, y + dy, y + dy, y), 'k')
        ax.text((x1 + x2) / 2., y + dy, text, ha='center', va='bottom')
    for x1, x2 in compare:
        res = stats.ttest_ind(
            activ[x1],
            activ[x2],
            equal_var=False
        )
        if res.pvalue < .05:
            text = '*' * min(int(1. + numpy.log10(.05 / res.pvalue)), 3)
        else:
            text = 'n.s.'
        annotate(x1, x2, text)
    return bars
