import numpy
import control
from scipy import linalg
from scipy import special
from scipy import optimize
from scipy import integrate
from numpy.typing import ArrayLike


def bandpass(f0: float, B: float):
    ω0 = 2. * numpy.pi * f0
    Bp = 2. * numpy.pi * B
    A = (-Bp, -ω0**2), (1., 0.)
    B = Bp, 0.
    C = 1., 0.
    return control.StateSpace(A, B, C, 0., 0.)


def alpha_gamma_filter(
    *,
    f1=10.,
    f2=40.,
    B1=4.,
    B2=30.,
    c1=1.,
    c2=-.5
):
    F1 = bandpass(f1, B1)
    F2 = bandpass(f2, B2)
    return c1 * F1 + c2 * F2


def pid(kp: float, ki=0., kd=0., fc: float | None = None):
    D = kp
    if kd:
        if fc is None:
            raise ValueError('fc must be specified for non-zero kd')
        ωc = 2. * numpy.pi * fc
        if ki:
            A = (0., 0.), (0., -ωc)
            B = 1., 1.
            C = ki, kd
        else:
            A = -ωc
            B = 1.
            C = kd
    else:
        if ki:
            A = 0.
            B = 1.
            C = ki
        else:
            A = B = C = ()
    return control.StateSpace(A, B, C, D, 0.)


def delay(delay: ArrayLike, dt: float | bool | None = None):
    delay = numpy.atleast_1d(numpy.squeeze(delay))
    if delay.ndim > 1:
        raise ValueError(f'delay must be 1-dimensional: {delay.ndim = }')
    if not issubclass(delay.dtype.type, numpy.integer):
        raise TypeError('delay must contain integer values')
    if dt == 0.:
        raise ValueError('dt must correspond to a discrete time system')
    A = linalg.block_diag(*(numpy.eye(d, k=-1) for d in delay))
    B = linalg.block_diag(*(numpy.eye(d, 1) for d in delay))
    C = linalg.block_diag(*(numpy.eye(1, d, k=d-1) for d in delay))
    return control.StateSpace(A, B, C, 0., dt)


def neural_oscillator(
    *,
    # coefficients
    τe1=5e-3,  # s
    τi1=20e-3,  # s
    τe2=5e-3,  # s
    τi2=20e-3,  # s
    N11=1.15,
    N21=.63,
    N12=2.52,
    N22=6.6,
    # input coupling strengths
    b1=.18,
    b2=.18,
    b3=.14,
    b4=.14,
    # observation
    ce=1.,
    ci=0.,
    # noise
    Q1=1e-4,
    Q2=1e-4,
    # neurons
    N=1000
):
    # matrices
    Tauinv = numpy.diag(numpy.array((τe1, τi1, τe2, τi2))**-1)
    M1 = numpy.array(((-1. + N11,     - N11),
                      (N21, -1. - N21)))
    M2 = numpy.array(((-1. + N12,     - N12),
                      (N22, -1. - N22)))
    Bu = numpy.array((b1, b2, b3, b4))[numpy.newaxis].T
    Bw = numpy.sqrt(numpy.array(((Q1, 0.), (0., 0.), (0., Q2), (0., 0.))) / N)
    # variables
    A = Tauinv @ linalg.block_diag(M1, M2)
    B = Tauinv @ numpy.hstack((Bu, Bw))
    C = numpy.array((ce, ci, ce, ci))
    return control.StateSpace(A, B, C, 0., 0.)


def cortico_thalamic(
    Ts: float,
    *,
    # synaptic scales
    τe=10e-3,  # s
    τi=50e-3,  # s
    τthe=5e-3,  # s
    τthi=30e-3,  # s
    τret=8e-3,  # s
    τce=5e-3,  # s
    τci=20e-3,  # s
    # cortico-subcortical delay
    τ=40e-3,  # s
    # coupling strengths
    Fe=1.,
    Fi=2.,
    Fccx=.05,
    Fct=1.2,
    Ftc=1.,
    Ftr=1.,
    Frt=.3,
    Frc=.6,
    # intra-populations coupling strengths
    Fcx=2.18,
    Mcx=3.88,
    # coupling strength
    Fcxth=.1,
    # coupling strength ratio
    S1=1.7,
    # mean input noise
    µe=.1,
    µi=0.,
    µthe=1.3,
    µthi=1.,
    µret=0.,
    µce=.05,
    µci=.05,
    # constant input
    Ie=2.7,
    Ii=1.7,
    Ice=1.1,
    Ici=.4,
    # input coupling strengths
    b1=1.,
    b2=1.,
    b3=1.,
    b4=1.,
    bret=0.,
    # input noise variances
    Qe_τe=.3,
    Qi_τi=.2,
    Qthe_τthe=.05,
    Qthi_τthi=.014,
    Qret_τret=.17,
    Qce=.005,
    Qci=.004,
    # observation
    w=.3,
    # neurons
    N=1000
):
    Qe = Qe_τe * τe
    Qi = Qi_τi * τi
    Qthe = Qthe_τthe * τthe
    Qthi = Qthi_τthi * τthi
    Qret = Qret_τret * τret
    Tauinv = numpy.diag(1. / numpy.array((τe, τi, τthe, τthi, τret, τce, τci)))
    Bw = numpy.sqrt(numpy.diag((Qe, Qi, Qthe, Qthi, Qret, Qce, Qci)) / N)
    Bu = numpy.array((b1, b2,  0.,  0.,  bret, b3, b4))[numpy.newaxis].T
    # finite size fluctuations
    # input noise variances
    # mean noise input
    s2c = Qe_τe + Qi_τi
    s2ret = Qret_τret
    s2th = Qthe_τthe + Qthi_τthi
    s2ce = Qce / τce
    s2ci = Qci / τci
    # transfer functions

    def erf_transfer(x: numpy.ndarray, s2: float):
        return (1. - special.erf(-x * (2. * s2)**-.5)) / 2.

    def Tc(x: numpy.ndarray): return erf_transfer(x, s2c)
    def Tret(x: numpy.ndarray): return erf_transfer(x, s2ret)
    def Tth(x: numpy.ndarray): return erf_transfer(x, s2th)
    def Se(x: numpy.ndarray): return S1 * erf_transfer(x, s2ce)
    def Si(x: numpy.ndarray): return erf_transfer(x, s2ci)

    def F(V: numpy.ndarray):
        '''Delay free part of the transfer function'''
        # vector unpacking
        Ve, Vi, Vthe, Vthi, Vret, u, v = V
        # transfer functions computations
        TcV = Tc(Ve - Vi)
        TretV = Tret(Vret)
        TthV = Tth(Vthe - Vthi)
        SeV = Se(u)
        SiV = Si(v)
        # dynamics
        FVe = -Ve + Fe * TcV + Fccx * SeV + µe + Ie
        FVi = -Vi + Fi * TcV + µi + Ii
        FVthe = -Vthe + µthe
        FVthi = -Vthi + Ftr * TretV + µthi
        FVret = -Vret + Frt * TthV + µret
        Fu = -u + Fcx * SeV - Mcx * SiV + µce + Ice
        Fv = -v - Fcx * SiV + Mcx * SeV + µci + Ici
        # vector
        return Tauinv @ numpy.array((FVe, FVi, FVthe, FVthi, FVret, Fu, Fv))

    def F2(W: numpy.ndarray):
        '''Delayed part of the transfer function'''
        # vector unpacking
        Wc, Wth = W
        # transfer function computation
        TcW = Tc(Wc)
        TthW = Tth(Wth)
        # dynamics
        F2Ve = Fct * TthW
        F2Vthe = Ftc * TcW
        F2Vret = Frc * TcW
        F2u = Fcxth * TthW
        # vector
        return Tauinv @ numpy.array((F2Ve, 0., F2Vthe, 0., F2Vret, F2u, 0.))

    # transfer functions derivatives
    def erf_transfer_prime(x: numpy.ndarray, s2: float):
        return numpy.exp(-x**2 / (2. * s2)) * (2. * numpy.pi * s2)**-.5

    def Tcp(x: numpy.ndarray): return erf_transfer_prime(x, s2c)
    def Tretp(x: numpy.ndarray): return erf_transfer_prime(x, s2ret)
    def Tthp(x: numpy.ndarray): return erf_transfer_prime(x, s2th)
    def Sep(x: numpy.ndarray): return S1 * erf_transfer_prime(x, s2ce)
    def Sip(x: numpy.ndarray): return erf_transfer_prime(x, s2ci)

    def J(V: numpy.ndarray):
        '''Jacobian matrix of the delay free part of the transfer function'''
        # vector unpacking
        Ve, Vi, Vthe, Vthi, Vret, u, v = V
        # transfer functions computations
        TcpVe = Tcp(Ve - Vi)
        TcpVi = -TcpVe
        TretpVret = Tretp(Vret)
        TthpVthe = Tthp(Vthe - Vthi)
        TthpVthi = -TthpVthe
        Sepu = Sep(u)
        Sipv = Sip(v)
        # dynamics
        dFVe_dVe = -1. + Fe * TcpVe
        dFVi_dVe = Fi * TcpVe
        dFVe_dVi = Fe * TcpVi
        dFVi_dVi = -1. + Fi * TcpVi
        dFVthe_dVthe = -1.
        dFVret_dVthe = Frt * TthpVthe
        dFVthi_dVthi = -1.
        dFVret_dVthi = Frt * TthpVthi
        dFVthi_dVret = Ftr * TretpVret
        dFVret_dVret = -1.
        dFVe_du = Fccx * Sepu
        dFu_du = -1. + Fcx * Sepu
        dFv_du = Mcx * Sepu
        dFu_dv = -Mcx * Sipv
        dFv_dv = -1. - Fcx * Sipv
        # vector
        return Tauinv @ numpy.array((
            (dFVe_dVe, dFVe_dVi, 0., 0., 0., dFVe_du, 0.),
            (dFVi_dVe, dFVi_dVi, 0., 0., 0.,      0., 0.),
            (0., 0., dFVthe_dVthe, 0., 0., 0., 0.),
            (0., 0.,           0., dFVthi_dVthi, dFVthi_dVret, 0., 0.),
            (0., 0., dFVret_dVthe, dFVret_dVthi, dFVret_dVret, 0., 0.),
            (0., 0., 0., 0., 0., dFu_du, dFu_dv),
            (0., 0., 0., 0., 0., dFv_du, dFv_dv)
        ))

    def J2(W: numpy.ndarray):
        '''Jacobian matrix of the delayed part of the transfer function'''
        # vector unpacking
        Wc, Wth = W
        # transfer function computation
        TcpWc = Tcp(Wc)
        TthpWth = Tthp(Wth)
        # dynamics
        dFVthe_dWc = Ftc * TcpWc
        dFVret_dWc = Frc * TcpWc
        dFVe_dWth = Fct * TthpWth
        dFu_dWth = Fcxth * TthpWth
        # matrix
        return Tauinv @ numpy.array(((0.,  dFVe_dWth),
                                     (0.,         0.),
                                     (dFVthe_dWc, 0.),
                                     (0.,         0.),
                                     (dFVret_dWc, 0.),
                                     (0.,   dFu_dWth),
                                     (0.,         0.)))

    B = Tauinv @ numpy.hstack((Bu, Bw))
    # constant input
    C = numpy.array((1.,  0.,  0.,  0.,  0.,   w,  0.))
    # state delay
    C2 = numpy.array(((1., -1.,  0.,  0., 0., 0., 0.),
                      (0.,  0.,  1., -1., 0., 0., 0.)))
    nx, nw = Bw.shape
    nx, nu = Bu.shape
    # find equilibrium point
    x0 = numpy.zeros(nx)
    def fun(x: numpy.ndarray): return F(x) + F2(C2 @ x)
    def jac(x: numpy.ndarray): return J(x) + J2(C2 @ x) @ C2
    root = optimize.root(fun, x0, jac=jac)
    if not root.success:
        raise ValueError('could not find an equilibrium point for the system')
    xeq: numpy.ndarray = root.x
    # linearize
    A = J(xeq)
    B2 = J2(C2 @ xeq)
    # sample linear
    d = τ / Ts
    if not d.is_integer():
        raise ValueError(f'{Ts = } is not a submultiple of {τ = }')
    delays = (int(d),) * 2
    Θ = delay(delays, Ts)
    lin = control.StateSpace(
        A,
        numpy.hstack((B, B2)),
        numpy.vstack((C, C2)),
        0.,
        0.
    ).sample(Ts).lft(Θ)

    # extend states
    sumdelay = sum(delays)
    xeq = numpy.hstack((xeq, (C2 @ xeq).repeat(Θ.nstates // C2.shape[0])))

    # sample non-linear
    def updfcn(
        t: float,
        x: numpy.ndarray,
        u: numpy.ndarray,
        params={}
    ):
        x1, x2 = x[:nx], x[nx:]
        inp = B @ u + F2(Θ.C @ x2)
        def fun(t: float, x: numpy.ndarray): return F(x) + inp
        def jac(t: float, x: numpy.ndarray): return J(x)
        sol = integrate.solve_ivp(fun, (t, t + Ts), x1)
        xp = numpy.empty_like(x)
        xp[:nx] = sol.y[..., -1]
        xp[nx:] = Θ.B @ C2 @ x1 + Θ.A @ x2
        return xp

    def outfcn(
        t: float,
        x: numpy.ndarray,
        u: numpy.ndarray | float = 0.,
        params={}
    ):
        return C @ x[:nx]

    return control.NonlinearIOSystem(
        updfcn,
        outfcn,
        dict(nu=nu, xeq=xeq, lin=lin),
        inputs=B.shape[1],
        outputs=1,
        states=len(xeq),
        dt=Ts
    )


def healthy_neural_oscillator():
    return neural_oscillator(Q1=3.6e-4, Q2=2.5e-5)


def healthy_cortico_thalamic(Ts: float):
    return cortico_thalamic(Ts, Qe_τe=.5, Qthe_τthe=.24, Qci=1e-6)
