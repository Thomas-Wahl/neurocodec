import numpy
import control
from scipy import signal
from numpy.typing import NDArray


def alpha(resp: control.TimeResponseData | control.FrequencyResponseData):
    '''Measure EEG alpha activity.'''
    return activity(resp, (8, 12))


def gamma(resp: control.TimeResponseData | control.FrequencyResponseData):
    '''Measure EEG gamma activity.'''
    return activity(resp, (25, 55))


def activity(
    resp: control.TimeResponseData | control.FrequencyResponseData,
    band: tuple[int, int] | None = None
) -> NDArray[numpy.floating]:
    '''Measure EEG activity in a given frequency band.'''
    match resp:
        case control.TimeResponseData():
            if band is None:
                fnyq = .5 / (resp.t[1] - resp.t[0])
                frange = 1., fnyq + .1, 1.
            else:
                frange = band[0], band[1] + .1, 1.
            f, Pxx = welch(resp, frange)
            return numpy.sqrt(Pxx.sum(-1))
        case control.FrequencyResponseData():
            f, Pxx = psd(resp)
            if band is not None:
                mask = numpy.logical_and(band[0] <= f, f <= band[1])
                f, Pxx = f[mask], Pxx[..., mask]
            return numpy.sqrt(Pxx.mean(-1) * (f[-1] - f[0]))


def welch(
    resp: control.TimeResponseData,
    frange: tuple[float, float, float] | None = None,
    **kwargs
) -> tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    if len(resp.t) < 2:
        raise ValueError('resp must be longer than 1')
    fs = 1. / (resp.t[1] - resp.t[0])
    if frange is None:
        return signal.welch(resp.y, fs, **kwargs)
    fstart, fstop, fstep = frange
    nperseg = round(fs / fstep)
    f, Pxx = signal.welch(resp.y, fs, nperseg=nperseg, **kwargs)
    mask = numpy.logical_and(fstart <= f, f < fstop)
    return f[mask], Pxx[..., mask]


def psd(resp: control.FrequencyResponseData):
    f = resp.frequency / (2. * numpy.pi)
    Pxx: NDArray[numpy.floating] = (2. * resp.magnitude**2).sum(1)
    return f, (Pxx.squeeze() if Pxx.shape[-1] > 1 else Pxx)


def standard(resp: control.TimeResponseData | control.FrequencyResponseData):
    fmin, fmax = 1., 80.
    match resp:
        case control.TimeResponseData():
            return welch(resp, (fmin, fmax + .1, 1.))
        case control.FrequencyResponseData():
            f, Pxx = psd(resp)
            mask = numpy.logical_and(fmin <= f, f <= fmax)
            return f[mask], Pxx[..., mask]


def spectrogram(
    resp: control.TimeResponseData,
    frange: tuple[float, float, float]
):
    widths = (resp.t[1] - resp.t[0]) * numpy.arange(*frange)
    return signal.cwt(resp.y, signal.morlet2, widths, w=2.*numpy.pi)


def standard_spectrogram(resp: control.TimeResponseData):
    return spectrogram(resp, (1., 80.1, 1.))
