import os
import eeg
import sim
import build
import model
import numpy
import control
import fitting
from scipy import interpolate
from numpy.typing import NDArray
from numpy.typing import ArrayLike


def ct_poles():
    return numpy.array((
        -22.06313568 + 7.00734018j, -22.06313568 - 7.00734018j,
        -22.41899703 + 64.81817606j, -22.41899703 - 64.81817606j,
        -31.00457731 + 134.74912842j, -31.00457731 - 134.74912842j,
        -71.85435801 + 219.99008376j, -71.85435801 - 219.99008376j
    ))


def psdfit(
    G: control.StateSpace | control.TransferFunction,
    f: NDArray[numpy.floating],
    Pgg: NDArray[numpy.floating]
):
    s = 2j * numpy.pi * f
    if not G.isdtime():
        Gs = G(s)
        Gfit = fitting.magvf(s, Pgg, G.poles())
        rmse = (numpy.abs((Gfit(s) - Gs) / Gs)**2).mean()**.5
        return Gfit, rmse
    poles = ct_poles()
    Gz = G(numpy.exp(G.dt * s))
    rs = numpy.random.RandomState(0)
    rmse = numpy.inf
    while rmse > .2:
        a = poles * (rs.rand(len(poles)) + .5)
        Gfit = fitting.magvf(s, Pgg, a, niter=8)
        rmse = (numpy.abs((Gfit(s) - Gz) / Gz)**2).mean()**.5
    return Gfit, rmse


def single_fit(
    G: control.StateSpace | control.TransferFunction,
    rest: control.TimeResponseData,
    stim: control.TimeResponseData
):
    ar = eeg.activity(stim) / eeg.activity(rest)
    ursp = control.TimeResponseData(
        stim.t,
        stim.u[:G.ninputs],
        issiso=rest.issiso
    )
    f, Py0 = eeg.standard(rest)
    f, Pyy = eeg.standard(stim)
    f, Puu = eeg.standard(ursp)
    Pgg = (Pyy - Py0) / Puu
    Gfit, rmse = psdfit(G, f, Pgg)
    return Gfit, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar


def fit_model(
    G: control.StateSpace | control.TransferFunction,
    rest: dict[int, control.TimeResponseData],
    stim: dict[int, control.TimeResponseData]
):
    rmse = numpy.empty(len(rest))
    ar = numpy.empty(len(rest))
    for i in sorted(rest):
        y0rp = rest[i]
        yrsp = stim[i]
        Gfit, ursp, f, Py0, Pyy, Puu, Pgg, rmse[i], ar[i] = single_fit(
            G,
            y0rp,
            yrsp
        )
    return Gfit, y0rp, yrsp, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar


def writing(filepath: str):
    print('writing', filepath, end=' '*(42-len(filepath))+'\r', flush=True)


def shift_seed(seed: int | None, offset: int):
    if seed is not None:
        return sim.next_seed((seed + offset) % 0x80000000)


def get_seed(*args: int | str | None):
    if None in args:
        return
    seed = 0
    for arg in args:
        if isinstance(arg, str):
            arg = sim.next_seed(
                int.from_bytes(arg.encode(), 'little') % 0x80000000
            )
        seed = shift_seed(arg, seed)
    return seed


def writefile(filepath: str, resp: control.TimeResponseData):
    '''Write time series to specified file.'''
    if resp.ntraces > 1:
        raise ValueError('multiple traces in resp are not allowed')
    data = bytearray((resp.ninputs, resp.noutputs))
    data.extend(len(resp.t).to_bytes(4, 'little'))
    data.extend(resp.t.tobytes())
    data.extend(resp.u.tobytes())
    data.extend(resp.y.tobytes())
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as file:
        file.write(data)


def readfile(filepath: str):
    '''Read time series from specified file.'''
    with open(filepath, 'rb') as file:
        ninputs, noutputs = file.read(2)
        length = int.from_bytes(file.read(4), 'little')
        data = file.read()
    time = numpy.frombuffer(data, count=length)
    inputs = numpy.frombuffer(
        data,
        time.dtype,
        ninputs*length,
        time.dtype.itemsize*time.size
    ).reshape(ninputs, length)
    outputs = numpy.frombuffer(
        data,
        time.dtype,
        noutputs*length,
        time.dtype.itemsize*(time.size+inputs.size)
    ).reshape(noutputs, length)
    return control.TimeResponseData(time, outputs, None, inputs)


def readdir(dirpath: str):
    data: dict[str] = {}
    for dpath, dnames, fnames in os.walk(dirpath):
        pos = data
        dirnames = dpath.split(os.sep)
        for dirname in dirnames:
            if dirname in dirpath:
                continue
            if dirname in pos:
                pos = pos[dirname]
                continue
            new_pos: dict[str] = {}
            pos[dirname] = new_pos
            pos = new_pos
        for fname in fnames:
            pos[int(fname)] = readfile(os.path.join(dpath, fname))
    return data


def request(*args: str | int):
    path = os.path.join('data', *(str(arg) for arg in args))
    return readdir(path) if os.path.isdir(path) else readfile(path)


def rest(*args: str | int):
    name = rest.__name__
    # sim
    tmax = 30.
    dt = 1e-3
    n = 50
    plants = {
        'lin': {
            'healthy': model.healthy_neural_oscillator(),
            'pathological': model.neural_oscillator()
        },
        'ct': {
            'healthy': model.healthy_cortico_thalamic(dt),
            'pathological': model.cortico_thalamic(dt)
        }
    }
    kwargs = {'lin': dict(dt=dt, nu=1), 'ct': dict()}
    for mdl in args[:1] if args else plants:
        for variant in args[1:2] if len(args) >= 2 else plants[mdl]:
            for i in args[2:3] if len(args) >= 3 else range(n):
                if not 0 <= int(i) < n:
                    raise ValueError(f'i must be in [0, {n}): i = {i}')
                filepath = os.path.join('data', name, mdl, variant, str(i))
                if os.path.exists(filepath):
                    continue
                writing(filepath)
                writefile(filepath, sim.rest(
                    plants[mdl][variant],
                    tmax,
                    seed=get_seed(name, mdl, variant, int(i)),
                    **kwargs[mdl]
                ))
    return request(name, *args)


def stim(*args: str | int):
    name = stim.__name__
    # sim
    uamp = 5e-3
    tmax = 30.
    dt = 1e-3
    n = 50
    plants = {
        'lin': model.neural_oscillator(),
        'ct': model.cortico_thalamic(dt)
    }
    for mdl in args[:1] if args else plants:
        for i in args[1:2] if len(args) >= 2 else range(n):
            if not 0 <= int(i) < n:
                raise ValueError(f'i must be in [0, {n}): i = {i}')
            filepath = os.path.join('data', name, mdl, str(i))
            if os.path.exists(filepath):
                continue
            writing(filepath)
            writefile(filepath, sim.stimwn(
                plants[mdl],
                uamp,
                tmax,
                dt,
                get_seed(name, mdl, int(i))
            ))
    return request(name, *args)


def stims(*args: str | int):
    name = stims.__name__
    # sim
    uamps = {
        'low': 2.5e-3,
        'high': 1e-2
    }
    tmax = 30.
    dt = 1e-3
    n = 50
    P = model.neural_oscillator()
    for uamp in args[:1] if args else uamps:
        for i in args[1:2] if len(args) >= 2 else range(n):
            if not 0 <= int(i) < n:
                raise ValueError(f'i must be in [0, {n}: i = {i}]')
            filepath = os.path.join('data', name, uamp, str(i))
            if os.path.exists(filepath):
                continue
            writing(filepath)
            writefile(filepath, sim.stimwn(
                P,
                uamps[uamp],
                tmax,
                dt,
                get_seed(name, uamp, int(i))
            ))
    return request(name, *args)


def cl(*args: str | int):
    name = cl.__name__
    # sim
    delay = 5
    tmax = 30.
    dt = 1e-3
    n = 50
    P = model.neural_oscillator()
    nu = 1
    F = model.alpha_gamma_filter()
    S = 1. + F
    loops = {
        'cl': build.alpha_gamma_cl(P, nu, delay, dt),
        'pi': build.pi_cl(P, nu, delay, dt),
        'lq': build.lqg_cl(P, nu, delay, dt)
    }
    for loop in args[:1] if args else loops:
        for i in args[1:2] if len(args) >= 2 else range(n):
            if not 0 <= int(i) < n:
                raise ValueError(f'i must be in [0, {n}): i = {i}')
            filepath = os.path.join('data', name, loop, str(i))
            if os.path.exists(filepath):
                continue
            writing(filepath)
            if loop == 'cl':
                writefile(filepath, sim.rest(
                    loops[loop],
                    tmax,
                    dt,
                    0,
                    get_seed(name, loop, int(i))
                ))
                continue
            y0rp = rest('lin', 'pathological', i)
            rrsp = sim.lsim(S, y0rp.t, y0rp.y)
            writefile(filepath, sim.stim(
                loops[loop],
                rrsp.y,
                dt,
                get_seed(name, loop, int(i))
            ))
    return request(name, *args)


def ctcl(*args: str | int):
    name = ctcl.__name__
    # sim
    delay = 5
    tmax = 30.
    dt = 1e-3
    n = 50
    P = model.cortico_thalamic(dt)
    G: control.StateSpace = P.params['lin'][:, :P.params['nu']]
    for i in args[:1] if args else range(n):
        if not 0 <= int(i) < n:
            raise ValueError(f'i must be in [0, {n}): i = {i}')
        filepath = os.path.join('data', name, str(i))
        if os.path.exists(filepath):
            continue
        y0rp = rest('ct', 'pathological', i)
        yrsp = stim('ct', i)
        Gfit, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar = single_fit(
            G,
            y0rp,
            yrsp
        )
        writing(filepath)
        writefile(filepath, sim.rest(
            build.alpha_gamma_cl(P, delay=delay, G=Gfit),
            tmax,
            dt,
            0,
            get_seed(name, int(i))
        ))
    return request(name, *args)


def tf(*args: str | int):
    name = tf.__name__
    # sim
    delay = 5
    tmax = 120.
    dt = 1e-3
    n = 50
    P = model.cortico_thalamic(dt)
    G: control.StateSpace = P.params['lin'][:, :P.params['nu']]
    for i in args[:1] if args else range(n):
        if not 0 <= int(i) < n:
            raise ValueError(f'i must be in [0, {n}): i = {i}')
        filepath = os.path.join('data', name, str(i))
        if os.path.exists(filepath):
            continue
        y0rp = rest('ct', 'pathological', i)
        yrsp = stim('ct', i)
        Gfit, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar = single_fit(
            G,
            y0rp,
            yrsp
        )
        loop = build.alpha_gamma_cl(P, delay=delay, G=Gfit)
        tdata = numpy.arange(0., tmax, dt)

        def interp(data: ArrayLike) -> NDArray:
            interp = interpolate.interp1d(
                numpy.linspace(0., tmax, 2 * len(data)),
                numpy.repeat(data, 2)
            )
            return interp(tdata)
        c1dat = interp((1e-12, 1e-12, 1., 1.))
        c2dat = interp((-1e-12, -.5, -1e-12, -.5))
        wdata = sim.white_noise(
            loop.ninputs,
            len(tdata),
            dt,
            get_seed(name, i)
        )
        ydata = numpy.empty((loop.noutputs, len(tdata)))
        if (x := sim.find_discrete_eqpt(loop)) is None:
            raise RuntimeError('could not find loop equilibrium point')
        for i, (t, w, c1, c2) in enumerate(
            zip(tdata, wdata.T, c1dat.T, c2dat.T)
        ):
            ydata[..., i] = loop.output(t, x, w)
            if c1dat[i+1] != c1 or c2dat[i+1] != c2:
                print('changed loop')
                loop = build.alpha_gamma_cl(
                    P,
                    delay=delay,
                    G=Gfit,
                    c1=c1,
                    c2=c2
                )
            y = ydata[..., i]
            newy = loop.output(t, x, w)
            if not numpy.allclose(y, newy, 5e-2, 1.):
                control.timeresp.warnings.warn(
                    'youtput error is too large after changing loop: '
                    f'{t = :.3f}, yerr = {(newy - y) / y}'
                )
                x[:-25] = 0. # K.x
            x = loop.dynamics(t, x, w)
        resp = control.TimeResponseData(tdata, ydata, None, wdata)
        from matplotlib import pyplot
        pyplot.imshow(control.mag2db(numpy.abs(eeg.standard_spectrogram(
            resp
        )[::-1])))
        raise SystemExit
        writefile(filepath, resp)
    return request(name, *args)
