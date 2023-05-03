import numpy
import control
from numpy.typing import ArrayLike


def freqresp(
    sys: control.lti.LTI,
    freq: ArrayLike,
    squeeze: bool | None = None
) -> control.FrequencyResponseData:
    return sys.frequency_response(2. * numpy.pi * freq, squeeze)


def lsim(
    sys: control.lti.LTI | control.InputOutputSystem,
    T: ArrayLike | None = None,
    U: ArrayLike | float = 0.,
    transpose=False,
    interpolate=False,
    return_x: bool | None = None,
    squeeze: bool | None = None
) -> control.TimeResponseData:
    X0 = 0.
    match sys:
        case control.lti.LTI():
            return control.forced_response(
                sys,
                T,
                U,
                X0,
                transpose,
                interpolate,
                return_x,
                squeeze
            )
        case control.InputOutputSystem():
            X0: numpy.ndarray | None = sys.params.get('xeq', None)
            if X0 is None:
                if sys.isdtime():
                    X0 = find_discrete_eqpt(sys, 0., 0.)
                else:
                    X0 = control.find_eqpt(sys, 0., 0.)
                if X0 is None:
                    raise RuntimeError('could not find sys equilibrium point')
            return control.input_output_response(
                sys,
                T,
                U,
                X0,
                {},
                transpose,
                return_x,
                squeeze
            )


def find_discrete_eqpt(
    sys: control.StateSpace | control.InputOutputSystem,
    x0: ArrayLike | float = 0.,
    u0: ArrayLike | float = 0.
):
    x = numpy.zeros(sys.nstates) + x0
    u = numpy.zeros(sys.ninputs) + u0
    for _ in range(1000):
        xp = sys.dynamics(0., x, u)
        if numpy.allclose(xp, x, 5e-2, .1):
            return xp
        x = xp


def next_seed(seed: int | None) -> int | None:
    if seed is not None:
        return numpy.random.RandomState(seed).randint(0x80000000)


def white_noise(size: int, length: int, dt: float, seed: int | None = None):
    return numpy.random.RandomState(seed).normal(0., dt**-.5, (size, length))


def get_params(
    sys: control.lti.LTI | control.InputOutputSystem,
    dt: float | None,
    nu: int | None
) -> tuple[float | bool, int, int]:
    if sys.isdtime():
        dt = control.common_timebase(sys, dt)
    elif not dt:
        raise ValueError(
            'dt > 0 must be provided for continuous time systems'
        )
    if isinstance(sys, control.InputOutputSystem):
        nu = sys.params.get('nu', nu)
    if nu is None:
        raise ValueError('nu must be provided for this system')
    if nu > sys.ninputs:
        raise ValueError(f'{nu = } is greater than {sys.ninputs = }')
    return dt, nu, sys.ninputs - nu


def rest(
    sys: control.lti.LTI | control.InputOutputSystem,
    tmax: float,
    dt: float | None = None,
    nu: int | None = None,
    seed: int | None = None
):
    '''Simulate the rest state activity of a stochastic system.'''
    dt, nu, nw = get_params(sys, dt, nu)
    T = numpy.arange(0., tmax, dt)
    U = numpy.zeros((sys.ninputs, len(T)))
    U[nu:] = white_noise(nw, len(T), dt, seed)
    return lsim(sys, T, U)


def stim(
    sys: control.lti.LTI | control.InputOutputSystem,
    u: ArrayLike,
    dt: float | None = None,
    seed: int | None = None
):
    '''Simulate the stimulation response of a stochastic system.'''
    u = numpy.atleast_2d(u)
    if u.ndim > 2:
        raise ValueError(f'U must be at most 2-dimensional: {u.ndim = }')
    dt, nu, nw = get_params(sys, dt, u.shape[0])
    T = numpy.arange(0., u.shape[1] * dt, dt)
    U = numpy.vstack((u, white_noise(nw, u.shape[1], dt, seed)))
    return lsim(sys, T, U)


def stimwn(
    sys: control.lti.LTI | control.InputOutputSystem,
    uamp: ArrayLike,
    tmax: float,
    dt: float | None = None,
    seed: int | None = None
):
    '''Simulate the response of a stochastic system to white noise.'''
    uamp = numpy.atleast_1d(numpy.squeeze(uamp))
    if uamp.ndim > 1:
        raise ValueError(f'uamp must be 1-dimensional: {uamp.ndim = }')
    dt, nu, nw = get_params(sys, dt, len(uamp))
    T = numpy.arange(0., tmax, dt)
    U = numpy.zeros((sys.ninputs, len(T)))
    U[:nu] = white_noise(
        nu,
        len(T),
        dt,
        next_seed(seed)
    ) * uamp[numpy.newaxis].T
    U[nu:] = white_noise(nw, len(T), dt, seed)
    return lsim(sys, T, U)
