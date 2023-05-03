import model
import numpy
import control
from numpy.typing import ArrayLike


def is_strictly_proper(sys: control.StateSpace | control.TransferFunction):
    match sys:
        case control.StateSpace():
            return not sys.D.any()
        case control.TransferFunction():
            return not any(
                [[len(num) for num in col] for col in sys.num] >=
                [[len(den) for den in col] for col in sys.den]
            )


def is_stable(sys: control.lti.LTI):
    if sys.isdtime():
        return (numpy.abs(sys.poles()) < 1.).all()
    return (sys.poles().real < 0.).all()


def inv(sys: control.lti.LTI):
    match sys:
        case control.StateSpace():
            D = numpy.linalg.inv(sys.D)
            A = sys.A - sys.B @ D @ sys.C
            B = sys.B @ D
            C = -D @ sys.C
            return control.StateSpace(A, B, C, D, sys.dt)
        case _:
            return 1. / sys


def div(
    sys: control.lti.LTI,
    other: control.lti.LTI
) -> control.StateSpace | control.lti.LTI:
    match sys:
        case control.StateSpace():
            if not is_strictly_proper(other):
                return sys * inv(other)
            sys = control.xferfcn._convert_to_transfer_function(sys)
            return control.statesp._convert_to_statespace(sys / other)
        case _:
            return sys / other


def pred(a: float, step=1):
    if not step:
        return control.StateSpace((), (), (), 1., True)
    z = control.TransferFunction.z
    Phi = ((2. - a) * z - 1.)**int(step) / (z - a)**int(step)
    return control.statesp._convert_to_statespace(Phi)


def pred_from_gain(maxgain: float, step=1):
    gain = maxgain**(1./step)
    a = (3. - gain) / (1. + gain)
    return pred(a, step)


def alpha_gamma_controller(
    G: control.StateSpace | control.lti.LTI,
    delay=0,
    dt: float | None = None,
    maxgain=10.,
    **kwargs: float
) -> control.lti.LTI:
    if not G.issiso():
        raise ValueError('G must be a SISO system')
    F = model.alpha_gamma_filter(**kwargs)
    if G.isdtime():
        dt = control.common_timebase(G.dt, dt)
    elif delay and not dt:
        raise ValueError(
            'dt > 0 must be provided for delayed continuous time plant'
        )
    if not delay:
        if G.isdtime():
            F = F.sample(dt)
        return div(F.feedback(), G)
    Phi = pred_from_gain(maxgain, delay)
    c1 = kwargs.pop('c1', 1.)
    c2 = kwargs.pop('c2', -.5)
    f1 = kwargs.get('f1', 10.)
    f2 = kwargs.get('f2', 40.)
    Fp = model.alpha_gamma_filter(
        c1=c1/numpy.abs(Phi(numpy.exp(2j * numpy.pi * f1 * dt))),
        c2=c2/numpy.abs(Phi(numpy.exp(2j * numpy.pi * f2 * dt))),
        **kwargs
    )
    if G.isdtime():
        return Phi * div(div(Fp.sample(dt), G), 1. + F.sample(dt))
    return Phi * div(div(Fp, G), 1. + F).sample(dt)


def kalman(
    P: control.StateSpace | control.InputOutputSystem,
    QN: ArrayLike,
    RN: ArrayLike,
    nu: int | None = None,
    G: control.StateSpace | None = None
):
    nu = get_nu(P, nu)
    if G is None:
        G = extract_G(P, nu)
    Bw = P.B[:, nu:]
    if P.isdtime():
        L, S, E = control.dlqe(G.A, Bw, G.C, QN, RN)
    else:
        L, S, E = control.lqe(G.A, Bw, G.C, QN, RN)
    A = G.A - L @ G.C
    B = numpy.hstack((L, G.B - L @ G.D))
    C = numpy.identity(G.nstates)
    return control.StateSpace(A, B, C, 0., P.dt)


def lqr(G: control.StateSpace, Q: ArrayLike, R: ArrayLike):
    K, S, E = control.lqr(G, Q, R)
    a = numpy.vstack((
        numpy.hstack((G.A, G.B)),
        numpy.hstack((G.C, G.D))
    ))
    b = numpy.vstack((
        numpy.zeros((G.nstates, 1)),
        numpy.ones((G.ninputs, 1))
    ))
    x = numpy.linalg.solve(a, b)
    Nx = x[:G.nstates]
    Nu = x[G.nstates:]
    N = Nu + K @ Nx
    return K, N


def get_nu(
    P: control.lti.LTI | control.InputOutputSystem,
    nu: int | None = None
) -> int:
    if isinstance(P, control.InputOutputSystem):
        nu = P.params.get('nu', nu)
    if nu is None:
        raise ValueError('nu must be provided for this plant')
    if nu > P.ninputs:
        raise ValueError(f'{nu = } is greater than {P.ninputs = }')
    return nu


def extract_G(P: control.lti.LTI | control.InputOutputSystem, nu: int):
    if isinstance(P, control.lti.LTI):
        return P[:, :nu]
    if 'lin' in P.params:
        return P.params['lin'][:, :nu]
    if (xeq := control.find_eqpt(P, numpy.zeros(P.nstates))) is None:
        raise ValueError('could not find an equilibrium point for P')
    return P.linearize(xeq, numpy.zeros(P.ninputs))[:, :nu]


def summing_junction(n: int, sign=1):
    I = numpy.identity(n)
    return control.StateSpace((), (), (), numpy.hstack((I, sign * I)))


def get_labels(signal: str, n: int):
    return [f'{signal}[{i}]' for i in range(n)]


def plant(
    P: control.lti.LTI | control.InputOutputSystem,
    nu: int | None = None
):
    nu = get_nu(P, nu)
    nw = P.ninputs - nu
    wlabels = get_labels('w', nw)
    ulabels = get_labels('u', nu)
    ylabels = get_labels('y', P.noutputs)
    match P:
        case control.lti.LTI():
            return control.LinearIOSystem(
                P,
                inputs=ulabels+wlabels,
                outputs=ylabels,
                name='P'
            )
        case control.InputOutputSystem():
            P = P.copy('P')
            P.input_index = {
                label: index for label, index in zip(
                    ulabels + wlabels,
                    P.input_index.values()
                )
            }
            P.output_index = {
                label: index for label, index in zip(
                    ylabels,
                    P.output_index.values()
                )
            }
            return P


def closed_loop(
    P: control.lti.LTI | control.InputOutputSystem,
    K: control.lti.LTI
) -> control.LinearICSystem | control.InterconnectedSystem:
    nu = K.noutputs
    P = plant(P, nu)
    wlabels = P.input_labels[nu:]
    ulabels = P.input_labels[:nu]
    ylabels = P.output_labels
    K = control.LinearIOSystem(K, inputs=ylabels, outputs=ulabels, name='K')
    return control.interconnect(
        [P, K],
        None,
        wlabels,
        ylabels + ulabels,
        dict(ny=P.noutputs)
    )


def reference_tracking(
    P: control.lti.LTI | control.InputOutputSystem,
    K: control.lti.LTI
) -> control.LinearICSystem | control.InterconnectedSystem:
    nu = K.noutputs
    P = plant(P, nu)
    sub = summing_junction(P.noutputs, -1)
    wlabels = P.input_labels[nu:]
    ulabels = P.input_labels[:nu]
    ylabels = P.output_labels
    rlabels = get_labels('r', P.noutputs)
    elabels = get_labels('e', P.noutputs)
    K = control.LinearIOSystem(K, inputs=elabels, outputs=ulabels, name='K')
    sub = control.LinearIOSystem(sub, inputs=rlabels+ylabels, outputs=elabels)
    return control.interconnect(
        [P, K, sub],
        None,
        rlabels + wlabels,
        ylabels + ulabels,
        dict(ny=P.noutputs)
    )


def lqg(
    P: control.lti.LTI | control.InputOutputSystem,
    E: control.lti.LTI,
    K: ArrayLike,
    N: ArrayLike
) -> control.LinearICSystem | control.InterconnectedSystem:
    K = numpy.atleast_2d(K)
    if K.ndim > 2:
        raise ValueError(f'K must be at most 2-dimensional: {K.ndim = }')
    if K.shape[1] != E.noutputs:
        raise ValueError(f'{K.shape[1] = } is different from {E.noutputs = }')
    nu = K.shape[0]
    if E.ninputs != P.noutputs + nu:
        raise ValueError(
            f'{E.ninputs = } is different from {P.noutputs + nu = }'
        )
    P = plant(P, nu)
    sub = summing_junction(nu, -1)
    wlabels = P.input_labels[nu:]
    ulabels = P.input_labels[:nu]
    ylabels = P.output_labels
    rlabels = get_labels('r', P.noutputs)
    xlabels = get_labels('x', E.noutputs)
    Nrlabels = get_labels('Nr', nu)
    Kxlabels = get_labels('Kx', nu)
    E = control.LinearIOSystem(
        E,
        inputs=ylabels+ulabels,
        outputs=xlabels,
        name='E'
    )
    K = control.LinearIOSystem(
        control.StateSpace((), (), (), K),
        inputs=xlabels,
        outputs=Kxlabels,
        name='K'
    )
    N = control.LinearIOSystem(
        control.StateSpace((), (), (), N),
        inputs=rlabels,
        outputs=Nrlabels,
        name='N'
    )
    sub = control.LinearIOSystem(
        sub,
        inputs=Nrlabels+Kxlabels,
        outputs=ulabels
    )
    return control.interconnect(
        [P, E, K, N, sub],
        None,
        rlabels + wlabels,
        ylabels + ulabels,
        dict(ny=P.noutputs)
    )


def smith(
    G: control.StateSpace,
    K: control.StateSpace,
    delay: ArrayLike,
    dt: float | None = None
):
    '''Build a Smith predictor from a plant G, a controller K and a delay.'''
    common_dt = control.common_timebase(G, K)
    if common_dt == 0.:
        if dt is None:
            raise ValueError('dt must be provided for continuous time systems')
        G = G.sample(dt)
        K = K.sample(dt)
    else:
        dt = control.common_timebase(common_dt, dt)
    Θ = model.delay(delay, dt)
    I = control.StateSpace((), (), (), numpy.identity(Θ.ninputs))
    return K.feedback(G * (I - Θ))


def smith_plant(
    P: control.lti.LTI | control.InputOutputSystem,
    delay: ArrayLike,
    dt: float | None = None,
    nu: int | None = None,
    G: control.lti.LTI | None = None
):
    nu = get_nu(P, nu)
    nw = P.ninputs - nu
    if G is None:
        G = extract_G(P, nu)
    Θ = model.delay(delay, dt)
    I = control.StateSpace((), (), (), numpy.identity(Θ.ninputs))
    if not P.isdtime():
        P = control.sample_system(P, dt)
        G = G.sample(dt)
    Phi = control.statesp._convert_to_statespace(G * (I - Θ))
    Phi = control.StateSpace(
        Phi.A,
        numpy.pad(Phi.B, ((0, 0), (0, nw))),
        Phi.C,
        numpy.pad(Phi.D, ((0, 0), (0, nw))),
        Phi.dt
    )
    match P:
        case control.StateSpace():
            return P.lft(Θ, ny=0) + Phi
        case _:
            P = control.interconnect(
                [P, Θ],
                [((0, i), (1, i)) for i in range(Θ.noutputs)],
                params = vars(P).get('params', {})
            )
            return control.parallel(P, Phi)


def alpha_gamma_cl(
    P: control.lti.LTI | control.InputOutputSystem,
    nu: int | None = None,
    delay=0,
    dt: float | None = None,
    G: control.lti.LTI | None = None,
    *,
    maxgain=10.,
    **kwargs: float
):
    if G is None:
        G = extract_G(P, get_nu(P, nu))
    if dt is None:
        dt = control.common_timebase(P, dt)
    if dt and not G.isdtime():
        G = G.sample(dt)
    K = alpha_gamma_controller(G, delay, dt, maxgain, **kwargs)
    if delay:
        K *= model.delay(delay, dt)
        if not P.isdtime():
            P = control.sample_system(P, dt)
    return closed_loop(P, K)


def pi_cl(
    P: control.lti.LTI | control.InputOutputSystem,
    nu: int | None = None,
    delay=0,
    dt: float | None = None,
    G: control.lti.LTI | None = None,
    *,
    kp=10.,
    ki=500.
):
    if P.isdtime():
        dt = control.common_timebase(P, dt)
    K = model.pid(kp, ki)
    if delay:
        if G is None:
            G = extract_G(P, get_nu(P, nu))
        K = smith(G, K, delay, dt) * model.delay(delay, dt)
        if not P.isdtime():
            P = control.sample_system(P, dt)
    return reference_tracking(P, K)


def lqg_cl(
    P: control.StateSpace | control.InputOutputSystem,
    nu: int | None = None,
    delay=0,
    dt: float | None = None,
    G: control.lti.LTI | None = None,
    *,
    Q: ArrayLike | None = None,
    R: ArrayLike | None = None
):
    if P.isdtime():
        dt = control.common_timebase(P, dt)
    nu = get_nu(P, nu)
    nw = P.ninputs - nu
    if G is None:
        G = extract_G(P, nu)
    QN = numpy.identity(nw)
    RN = numpy.identity(P.noutputs) * 1e-6
    E = kalman(P, QN, RN, nu, G)
    if Q is None:
        Q = numpy.identity(E.noutputs)
    if R is None:
        R = numpy.identity(nu) * 1e-3
    K, N = lqr(G, Q, R)
    if delay:
        P = smith_plant(P, delay, dt, nu, G)
        if not E.isdtime():
            E = E.sample(dt)
    return lqg(P, E, K, N)
