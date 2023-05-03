import numpy
import control
from scipy import signal
from numpy.typing import ArrayLike


'''
NOTE: The use of this program is limited to NON-COMMERCIAL usage only.
All credit goes to Bjorn Gustavsen for his MATLAB implementation,
and the following papers:
 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.
 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.
 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
'''


def complex_index(a: numpy.ndarray):
    cindex = numpy.zeros(len(a), int)
    cindex[0] = bool(a[0].imag)
    for i in range(1, len(a) - 1):
        if a[i].imag:
            if cindex[i - 1] == 1:
                cindex[i] = 2
            else:
                cindex[i] = 1
                cindex[i + 1] = 2
    return cindex


def starting_poles(
    s: ArrayLike,
    n: int | None = None,
    dt: float | bool | None = None,
    logspace=False
):
    nreal = 0 if n is None else n % 2
    npair = (len(s) if n is None else n) // 2
    n = nreal + npair
    omega = numpy.abs(s.imag)
    omega[omega == 0.] += 10.**-n
    if logspace:
        beta = numpy.logspace(
            numpy.log10(min(omega)),
            numpy.log10(max(omega)),
            n
        )
    else:
        beta = numpy.linspace(min(omega), max(omega), n)
    alpha = beta / 100.
    poles = numpy.zeros(nreal + 2 * npair, complex)
    if nreal:
        poles[0] = alpha[0]
    for i in range(nreal, n):
        poles[2 * i - nreal] = alpha[i] - 1j * beta[i]
        poles[2 * i + 1 - nreal] = alpha[i] + 1j * beta[i]
    return numpy.exp(poles * dt) if dt else poles


def real_ss(a: numpy.ndarray, C: numpy.ndarray):
    N = len(a)
    A = numpy.diag(a)
    B = numpy.ones(N)
    C = numpy.atleast_2d(C)
    for n in range(0, N - 1, 2):
        if a[n].imag:
            A[n, n] = a[n].real
            A[n + 1, n] = a[n + 1].imag
            A[n, n + 1] = a[n].imag
            A[n + 1, n + 1] = a[n + 1].real
            B[n] = 2.
            B[n + 1] = 0.
            c1 = C[:, n].real.copy()
            c2 = C[:, n].imag.copy()
            C[:, n] = c1
            C[:, n + 1] = c2
    return A.real, B, numpy.squeeze(C.real)


def real_lti(
    a: numpy.ndarray,
    C: numpy.ndarray,
    D: numpy.ndarray,
    E: numpy.ndarray,
    improper=False,
    dt: float | bool | None = None
) -> control.StateSpace | control.TransferFunction:
    A, B, C = real_ss(a, C)

    B = B[numpy.newaxis].T
    D = D[numpy.newaxis].T
    if improper:
        if len(E) > 1:
            num = tuple((numpy.array((Em, 0.)),) for Em in E)
            den = tuple((numpy.array((1.,)),) for _ in E)
            E = E[numpy.newaxis].T
            sE = control.tf(num, den, dt)
        else:
            sE = (
                control.TransferFunction.z if dt
                else control.TransferFunction.s
            ) * E
        sys = control.xferfcn._convert_to_transfer_function(
            control.StateSpace(A, B, C, D, dt)
        ) + sE
    else:
        sys = control.StateSpace(A, B, C, D, dt)
    return sys


def sort_eigvals(eigvals: numpy.ndarray):
    '''Sorts eigenvalues so that the real ones come first.'''
    N = len(eigvals)
    for n in range(N):
        if eigvals[n].imag:
            for m in range(n + 1, N):
                if not eigvals[m].imag:
                    eigvals[n], eigvals[m] = eigvals[m], eigvals[n]
    return -numpy.sort(-eigvals)


def sort_residues(
    a: numpy.ndarray,
    p: numpy.ndarray,
    r: numpy.ndarray
) -> numpy.ndarray:
    # find true index of each pole p
    indices = -numpy.ones(a.shape, int)
    for i in range(len(indices)):
        delta = numpy.abs(a - p[i])
        index = -1
        min_delta = numpy.inf
        for j in range(len(delta)):
            if delta[j] < min_delta:
                min_delta = delta[j]
                index = j
        indices[i] = index

    # if not numpy.allclose(a, p[indices], 1e-1):
    #    raise ValueError("Unmatching poles 'a' and 'p'")

    return r[indices]


def sigma_zeros(
    a: numpy.ndarray,
    C: numpy.ndarray,
    D: float,
    discrete=False,
    stable=True
):
    A, B, C = real_ss(a, C)
    eigvals = numpy.linalg.eigvals(A - numpy.outer(B, C) / D)
    # Forcing unstable poles to be stable
    if stable:
        if discrete:
            eigvals /= eigvals**(2 * (numpy.abs(eigvals) > 1.))
        else:
            eigvals -= 2. * eigvals.real * (eigvals.real > 0.)
    # sort poles so that the real ones come first
    return sort_eigvals(eigvals)


def sigma_symmetric_zeros(a: numpy.ndarray, C: numpy.ndarray, D: float):
    A, B, C = real_ss(a**2, 2. * a * C)
    eigvals = numpy.linalg.eigvals(A - numpy.outer(B, C) / D)
    # flip signs of negative real eigvals
    eigvals -= 2. * eigvals.real * \
        numpy.logical_and(eigvals.imag == 0., eigvals.real < 0.)
    # take the square root of the eigvals
    # and only select the ones with a negative real part
    eigvals **= .5
    eigvals -= 2. * eigvals.real * (eigvals.real > 0.)
    # sort poles so that the real ones come first
    return sort_eigvals(eigvals)


def single_pole(s, a):
    return 1. / (s - a)


def symmetric_pole(s, a):
    return 1. / (s - a) - 1. / (s + a)


def solve_poles(
    s: numpy.ndarray,
    f: numpy.ndarray,
    a: numpy.ndarray,
    w: numpy.ndarray,
    strictly_proper=True,
    improper=False,
    discrete=False,
    stable=True,
    relax=True,
    symmetric=False
):
    '''
    s: C^Ns       (frequency points)
    f: C->C^NcxNs (function evaluated at s)
    a: C^N        (poles)
    w: C^NcxNs    (weights)
    '''
    if discrete and symmetric:
        raise ValueError('Cannot solve discrete symmetric poles.')

    Ns = len(s)
    Nc = f.shape[0]
    N = len(a)
    offs = (not strictly_proper) + improper
    offset = N + offs
    Phi = symmetric_pole if symmetric else single_pole

    Dt = 1.
    # tolerances for Dt
    tol_low = 1e-18
    tol_high = 1e18

    cindex = complex_index(a)

    Dk = numpy.zeros((Ns, N + 1 + improper), complex)
    for n in range(N):
        if not cindex[n]:    # real pole
            Dk[:, n] = Phi(s, a[n])
        elif cindex[n] == 1:  # complex pole, first part
            Dk[:, n] = Phi(s, a[n]) + Phi(s, a[n + 1])
            Dk[:, n + 1] = 1j * (Phi(s, a[n]) - Phi(s, a[n + 1]))
    Dk[:, N] = 1.
    if improper:
        Dk[:, N + 1] = s

    if relax:
        # Scaling for last row of LS-problem (pole identification)
        scale = 0.
        for m in range(Nc):
            scale += numpy.linalg.norm(w[m] * f[m])**2
        scale = scale**.5 / Ns

        A = numpy.zeros((Nc * (N + 1), N + 1))
        b = numpy.zeros(Nc * (N + 1))
        for m in range(Nc):
            Am = numpy.zeros((Ns, offset + N + 1), complex)

            for n in range(offset):  # left  block
                Am[:, n] = w[m] * Dk[:, n]
            for n in range(N + 1):    # right block
                Am[:, offset + n] = -w[m] * Dk[:, n] * f[m]

            Am = numpy.vstack((Am.real, Am.imag))

            # Integral criterion for sigma
            if m + 1 == Nc:
                Am = numpy.pad(Am, ((0, 1), (0, 0)))
                for n in range(N + 1):
                    Am[2 * Ns, offset + n] = scale * numpy.sum(Dk[:, n]).real

            Q, R = numpy.linalg.qr(Am)
            start = offset
            stop = offset + N + 1
            R22 = R[start: stop, start: stop]
            A[m * (N + 1): (m + 1) * (N + 1)] = R22
            if m + 1 == Nc:
                b[m * (N + 1): (m + 1) * (N + 1)] = scale * \
                    Q[-1, offset:].conj() * Ns

        Escale = numpy.zeros(A.shape[1])
        for n in range(A.shape[1]):
            Escale[n] = 1. / numpy.linalg.norm(A[:, n])
            A[:, n] *= Escale[n]

        x, residuals, rank, s = numpy.linalg.lstsq(A, b, rcond=-1)
        x *= Escale

        if not (tol_low < numpy.abs(x[-1]) < tol_high):
            if not x[-1]:
                Dt = 1.
            elif numpy.abs(x[-1]) < tol_low:
                Dt = numpy.sign(x[-1]) * tol_low
            elif numpy.abs(x[-1]) > tol_high:
                Dt = numpy.sign(x[-1]) * tol_high
            relax = False

    # Situation: No relaxation, or produced D~ extremely small or large.
    # Solve again, without relaxation
    if not relax:
        A = numpy.zeros((Nc * N, N))
        b = numpy.zeros(Nc * N)
        for m in range(Nc):
            Am = numpy.zeros((Ns, offset + N), complex)

            for n in range(offset):  # left  block
                Am[:, n] = w[m] * Dk[:, n]
            for n in range(N):      # right block
                Am[:, offset + n] = -w[m] * Dk[:, n] * f[m]
            bm = w[m] * f[m] * Dt

            Am = numpy.vstack((Am.real, Am.imag))
            bm = numpy.concatenate((bm.real, bm.imag))
            Q, R = numpy.linalg.qr(Am)
            start = offset
            stop = offset + N
            R22 = R[start: stop, start: stop]
            A[m * N: (m + 1) * N] = R22
            b[m * N: (m + 1) * N] = Q[:, start: stop].T @ bm

        Escale = numpy.zeros(A.shape[1])
        for n in range(A.shape[1]):
            Escale[n] = 1. / numpy.linalg.norm(A[:, n])
            A[:, n] *= Escale[n]

        x, residuals, rank, s = numpy.linalg.lstsq(A, b, rcond=-1)
        x *= Escale

        x = numpy.append(x, Dt)

    Ct = x[:-1]
    Dt = x[-1]

    # We now change back to make C~ complex again
    Ct = Ct.astype(complex)
    for n in range(len(cindex) - 1):
        if cindex[n] == 1:
            c1 = Ct[n]
            c2 = Ct[n + 1]
            Ct[n] = c1 + 1j * c2
            Ct[n + 1] = c1 - 1j * c2

    return sigma_symmetric_zeros(a, Ct, Dt) if symmetric \
        else sigma_zeros(a, Ct, Dt, discrete, stable)


def solve_residues(
    s: numpy.ndarray,
    f: numpy.ndarray,
    a: numpy.ndarray,
    w: numpy.ndarray,
    strictly_proper=True,
    improper=False,
    symmetric=False
):
    '''
    s: C^Ns       (frequency points)
    f: C->C^NcxNs (function evaluated at s)
    a: C^N        (poles)
    w: C^NcxNs    (weights)
    '''

    Ns = len(s)
    Nc = f.shape[0]
    N = len(a)
    offs = (not strictly_proper) + improper
    offset = N + offs
    Phi = symmetric_pole if symmetric else single_pole

    cindex = complex_index(a)

    # ========================================================================
    # We now calculate the SER for f (new fitting), using the above calculated
    # zeros as known poles :
    # ========================================================================
    A = numpy.zeros((2 * Ns, offset), complex)
    B = numpy.zeros((2 * Ns, Nc), complex)

    Dk = numpy.zeros((Ns, N), complex)
    for n in range(N):
        if not cindex[n]:    # real pole
            Dk[:, n] = Phi(s, a[n])
        elif cindex[n] == 1:  # complex pole, 1st part
            Dk[:, n] = Phi(s, a[n]) + Phi(s, a[n + 1])
            Dk[:, n + 1] = 1j * (Phi(s, a[n]) - Phi(s, a[n + 1]))

    A[:Ns, :N] = Dk
    if not strictly_proper:
        A[:Ns, N] = 1.
    if improper:
        A[:Ns, N + 1] = s
    # TODO: multiply column Nc and Nc + 1 by weights when common weights
    for m in range(Nc):
        A[:Ns, m] *= w[m]
        B[:Ns, m] = w[m] * f[m]

    A[Ns: 2 * Ns] = A[:Ns].imag
    A[:     Ns] = A[:Ns].real
    B[Ns: 2 * Ns] = B[:Ns].imag
    B[:     Ns] = B[:Ns].real
    A = A.real
    B = B.real

    Escale = numpy.zeros(A.shape[1])
    for n in range(A.shape[1]):
        Escale[n] = numpy.linalg.norm(A[:, n])
        A[:, n] /= Escale[n]
    X, residuals, rank, s = numpy.linalg.lstsq(A, B, rcond=-1)
    for n in range(Nc):
        X[:, n] /= Escale

    C = X[:N].T
    D = numpy.zeros(Nc) if strictly_proper else X[N]
    E = numpy.zeros(Nc) if not improper else X[N + 1]

    # We now change back to make C complex again
    C = C.astype(complex)
    for n in range(len(cindex) - 1):
        if cindex[n] == 1:
            C1 = C[:, n].copy()
            C2 = C[:, n + 1].copy()
            C[:, n] = C1 + 1j * C2
            C[:, n + 1] = C1 - 1j * C2

    return (passive_residues(a, C) if symmetric else numpy.squeeze(C)), D, E


def passive_residues(a: numpy.ndarray, C: numpy.ndarray):
    A, B, C = real_ss(a**2, 2. * a * C)
    B = B[numpy.newaxis].T
    C = numpy.atleast_2d(C)
    R = numpy.zeros(C.shape, complex)
    for m in range(C.shape[0]):
        tf = control.ss2tf(A, B, C[m], 0.)
        eigvals = tf.zeros()
        # take the square root of the eigvals
        # and only select the ones with a negative real part
        eigvals **= .5
        eigvals -= 2. * eigvals.real * (eigvals.real > 0.)
        gain = numpy.abs(tf.num[0][0][0])**.5
        num, den = signal.zpk2tf(eigvals, a, gain)
        r, p, _ = signal.residue(num, den)
        R[m] = sort_residues(a, p, r)
    return numpy.squeeze(R)


def check_params(
    s: ArrayLike,
    f: ArrayLike,
    poles: ArrayLike | int | None = None,
    weights: ArrayLike | None = None,
    strictly_proper: bool | None = None,
    improper=False,
    relax=True,
    dt: float | bool | None = None
):
    # check params
    if strictly_proper and improper:
        raise ValueError(
            'LTI system cannot be both strictly proper and improper'
        )
    s = numpy.atleast_1d(numpy.squeeze(s))
    f = numpy.atleast_2d(f)
    # check dimensions
    if s.ndim > 1:
        raise ValueError(f's must be 1-dimensional: {s.ndim = }')
    if f.ndim > 2:
        raise ValueError(f'f must be at most 2-dimensional: {f.ndim = }')
    if len(s) not in f.shape:
        raise ValueError(f'f must be the same length as s')
    # force f to be an Nc×Ns array
    if f.shape[1] != len(s):
        f = f.T
    # set default values
    if strictly_proper is None:
        strictly_proper = not improper
    if poles is None:
        n = (
            len(s) -
            ((not strictly_proper) + improper + (f.shape[0] > 1 or relax)) / 2
        ).__floor__()
        poles = starting_poles(s, n, dt)
    elif isinstance(poles, int):
        poles = starting_poles(s, poles, dt)
    else:
        poles = numpy.atleast_1d(numpy.squeeze(poles))
        if poles.ndim > 1:
            raise ValueError(f'poles must be 1-dimensional: {poles.ndim = }')
    # check number of sample points
    minNs = (
        len(poles) +
        ((not strictly_proper) + improper + (f.shape[0] > 1 or relax)) / 2
    ).__ceil__()
    if len(s) < minNs:
        raise ValueError(f's length must be at least {minNs}: {len(s) = }')
    # set default values
    weights = numpy.ones(f.shape) if weights is None \
        else numpy.atleast_2d(weights)
    # check dimensions
    if weights.ndim > 2:
        raise ValueError(
            f'weights must be at most 2-dimensional: {weights.ndim = }'
        )
    # force weights to be an Nc×Ns array
    if weights.ndim == 1:
        if len(weights) not in f.shape:
            raise ValueError(
                'mismatch between weights length and f dimensions: '
                f'{len(weights) = }, {f.shape = }'
            )
        # match weights to s in priority
        if len(weights) == f.shape[1]:
            weights = numpy.tile(weights, (f.shape[0], 1))
        else:
            weights = numpy.tile(weights[numpy.newaxis].T, (1, f.shape[1]))
    else:
        if sorted(weights.shape) != sorted(f.shape):
            raise ValueError(
                'mismatch between weights dimensions and f dimensions: '
                f'{weights.shape = }, {f.shape = }'
            )
        elif weights.shape[0] != f.shape[0]:
            weights = weights.T
    return s, f, poles, weights, strictly_proper, improper


def vf(
    s: ArrayLike,
    f: ArrayLike,
    poles: ArrayLike | int | None = None,
    weights: ArrayLike | None = None,
    strictly_proper=None,
    improper=False,
    dt=None,
    stable=True,
    relax=True,
    niter=1
):
    s, f, a, w, strictly_proper, improper = check_params(
        s,
        f,
        poles,
        weights,
        strictly_proper,
        improper,
        relax,
        dt
    )

    for _ in range(niter):
        a = solve_poles(
            s,
            f,
            a,
            w,
            strictly_proper,
            improper,
            bool(dt),
            stable,
            relax
        )

    C, D, E = solve_residues(s, f, a, w, strictly_proper, improper)

    return real_lti(a, C, D, E, improper, dt)


def magvf(
    s: ArrayLike,
    f: ArrayLike,
    poles: ArrayLike | int | None = None,
    weights: ArrayLike | None = None,
    strictly_proper: bool | None = None,
    improper=False,
    relax=True,
    niter=1
):
    s, f, a, w, strictly_proper, improper = check_params(
        s,
        f,
        poles,
        weights,
        strictly_proper,
        improper,
        relax,
        0.
    )
    # magvf specific checks
    if f.imag.any():
        raise ValueError('f must contain real values')
    f = f.astype(float)

    for _ in range(niter):
        a = solve_poles(
            s,
            f,
            a,
            w,
            strictly_proper,
            improper,
            False,
            False,
            relax,
            symmetric=True
        )
    C, D, E = solve_residues(
        s,
        f,
        a,
        w,
        strictly_proper,
        improper,
        symmetric=True
    )
    return real_lti(a, C, D, E, improper, 0.)
