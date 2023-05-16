import os
import eeg
import sim
import plot
import build
import numpy
import model
import series
import control
from matplotlib import pyplot
from matplotlib import patches


pyplot.style.use('FrontiersinHarvard.mplstyle')


def save(name: str):
    if not os.path.exists('figure'):
        os.mkdir('figure')
    pyplot.figure(name).savefig(
        os.path.join('figure', f'{name}.pdf'),
        transparent=True
    )


def measure(activity, data: dict[int, control.TimeResponseData]):
    return numpy.fromiter((activity(data[i]) for i in sorted(data)), float)


def rest():
    name = rest.__name__
    data = series.rest()
    resps = {
        'Aa': data['lin']['healthy'][0],
        'Ab': data['lin']['pathological'][0],
        'Ba': data['ct']['healthy'][0],
        'Bb': data['ct']['pathological'][0]
    }
    activs = {
        'A3': numpy.vstack((
            measure(eeg.alpha, data['lin']['healthy']),
            measure(eeg.alpha, data['lin']['pathological'])
        )),
        'A4': numpy.vstack((
            measure(eeg.gamma, data['lin']['healthy']),
            measure(eeg.gamma, data['lin']['pathological'])
        )),
        'B3': numpy.vstack((
            measure(eeg.alpha, data['ct']['healthy']),
            measure(eeg.alpha, data['ct']['pathological'])
        )),
        'B4': numpy.vstack((
            measure(eeg.gamma, data['ct']['healthy']),
            measure(eeg.gamma, data['ct']['pathological'])
        ))
    }
    # plot
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight((22./7.) * fig.get_figheight())
    shape = 14, 4
    axes = {
        'A1': pyplot.subplot2grid(shape, (2, 0), 4, 2, title='A)'),
        'B1': pyplot.subplot2grid(shape, (2, 2), 4, 2, title='B)'),
        'A2': pyplot.subplot2grid(shape, (6, 0), 4, 2),
        'B2': pyplot.subplot2grid(shape, (6, 2), 4, 2),
        'A3': pyplot.subplot2grid(shape, (10, 0), 4, 1),
        'A4': pyplot.subplot2grid(shape, (10, 1), 4, 1),
        'B3': pyplot.subplot2grid(shape, (10, 2), 4, 1),
        'B4': pyplot.subplot2grid(shape, (10, 3), 4, 1)
    }
    axes['A1'].set_title('linear brain model', loc='center')
    axes['B1'].set_title('cortico-thalamic model', loc='center')
    axes['A3'].set_ylabel('$\\alpha$-activity')
    axes['B3'].set_ylabel('$\\alpha$-activity')
    axes['A4'].set_ylabel('$\\gamma$-activity')
    axes['B4'].set_ylabel('$\\gamma$-activity')
    tbounds = 29.5, 30.
    akw = dict(color='C7', label='healthy')
    bkw = dict(color='C0', label='pathological')
    compare = (0, 1),
    color = 'C7', 'C0'
    plot.time_series(axes['A1'], resps['Aa'], 's', tbounds=tbounds, **akw)
    plot.time_series(axes['B1'], resps['Ba'], 's', tbounds=tbounds, **akw)
    plot.time_series(axes['A1'], resps['Ab'], 's', tbounds=tbounds, **bkw)
    plot.time_series(axes['B1'], resps['Bb'], 's', tbounds=tbounds, **bkw)
    plot.psd(axes['A2'], *eeg.standard(resps['Aa']), 'Hz', **akw)
    plot.psd(axes['B2'], *eeg.standard(resps['Ba']), 'Hz', **akw)
    plot.psd(axes['A2'], *eeg.standard(resps['Ab']), 'Hz', **bkw)
    plot.psd(axes['B2'], *eeg.standard(resps['Bb']), 'Hz', **bkw)
    plot.activity(axes['A3'], activs['A3'], compare, color=color)
    plot.activity(axes['A4'], activs['A4'], compare, color=color)
    plot.activity(axes['B3'], activs['B3'], compare, color=color)
    plot.activity(axes['B4'], activs['B4'], compare, color=color)
    fig.legend(*axes['A1'].get_legend_handles_labels(), ncol=2)
    save(name)
    pyplot.show()


def fit():
    name = fit.__name__
    rest = series.rest('lin', 'pathological')
    stim = series.stim('lin')
    G = model.neural_oscillator()[:, 0]
    Gfit, y0rp, yrsp, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar = series.fit_model(
        G,
        rest,
        stim
    )
    freq = numpy.linspace(1., f[-1], 1000)
    Gresp = sim.freqresp(G, freq)
    Gfitr = sim.freqresp(Gfit, freq)
    # plot
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight(2.5 * fig.get_figheight())
    shape = 15, 2
    span = 6, 1
    axes = {
        'A': pyplot.subplot2grid(shape, (3, 0), *span, title='A)'),
        'B': pyplot.subplot2grid(shape, (3, 1), *span, title='B)'),
        'C': pyplot.subplot2grid(shape, (9, 0), *span, title='C)'),
        'D': pyplot.subplot2grid(shape, (9, 1), *span, title='D)')
    }
    tbounds = 29.5, 30.
    kwargs = {
        'u': dict(color='C2', label='$u$'),
        'y': dict(color='C3', label='$y$'),
        'y0': dict(color='C0', label='$y_0$'),
        'G': dict(color='k', label='$G$'),
        'Gfit': dict(color='C9', linestyle='--', label='$\\tilde{G}$'),
        'Pgg': dict(color='C6', label='$|\hat{g}|^2$')
    }
    axes['C'].set_title(
        f'RMSE = {rmse.mean():.1%}$\pm${rmse.var()**.5:.1%}'
        .replace('%', '\%'),
        loc='right'
    )
    plot.time_series(axes['A'], ursp, 's', tbounds=tbounds, **kwargs['u'])
    plot.time_series(axes['A'], yrsp, 's', tbounds=tbounds, **kwargs['y'])
    plot.time_series(axes['A'], y0rp, 's', tbounds=tbounds, **kwargs['y0'])
    plot.psd(axes['B'], f, Puu, 'Hz', **kwargs['u'])
    plot.psd(axes['B'], f, Pyy, 'Hz', **kwargs['y'])
    plot.psd(axes['B'], f, Py0, 'Hz', **kwargs['y0'])
    plot.bode(axes['C'], axes['D'], Gresp, 'Hz', **kwargs['G'])
    plot.psd(axes['C'], f, Pgg, 'Hz', **kwargs['Pgg'])
    plot.bode(axes['C'], axes['D'], Gfitr, 'Hz', **kwargs['Gfit'])
    h1, l1 = axes['A'].get_legend_handles_labels()
    h2, l2 = axes['C'].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=6)
    save(name)
    pyplot.show()


def fits():
    name = fits.__name__
    rest = series.rest('lin', 'pathological')
    stims = series.stims() | {'mid': series.stim('lin')}
    G = model.neural_oscillator()[:, 0]
    # figure
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight(3.5 * fig.get_figheight())
    shape = 21, 2
    span = 6, 1
    axes = {
        'A1': pyplot.subplot2grid(shape, (3, 0), *span, title='A)'),
        'B1': pyplot.subplot2grid(shape, (9, 0), *span, title='B)'),
        'C1': pyplot.subplot2grid(shape, (15, 0), *span, title='C)'),
        'A2': pyplot.subplot2grid(shape, (3, 1), *span),
        'B2': pyplot.subplot2grid(shape, (9, 1), *span),
        'C2': pyplot.subplot2grid(shape, (15, 1), *span)
    }
    uaxs = {
        'high': 'A',
        'mid': 'B',
        'low': 'C'
    }
    kwargs = {
        'G': dict(color='k', label='$G$'),
        'Gfit': dict(color='C9', linestyle='--', label='$\\tilde{G}$'),
        'Pgg': dict(color='C6', label='$|\hat{g}|^2$')
    }
    for uamp in stims:
        (
            Gfit,
            y0rp,
            yrsp,
            ursp,
            f,
            Py0,
            Pyy,
            Puu,
            Pgg,
            rmse,
            ar
        ) = series.fit_model(G, rest, stims[uamp])
        freq = numpy.linspace(1., f[-1], 1000)
        Gresp = sim.freqresp(G, freq)
        Gfitr = sim.freqresp(Gfit, freq)
        # plot
        axmag, axphase = (axes[f'{uaxs[uamp]}{i}'] for i in (1, 2))
        plot.bode(axmag, axphase, Gresp, 'Hz', **kwargs['G'])
        plot.psd(axmag, f, Pgg, 'Hz', **kwargs['Pgg'])
        plot.bode(axmag, axphase, Gfitr, 'Hz', **kwargs['Gfit'])
        axmag.set_title(
            f'AR = {ar.mean():.3f}$\pm${ar.var()**.5:.3f}',
            loc='right'
        )
        axphase.set_title(
            f'RMSE = {rmse.mean():.1%}$\pm${rmse.var()**.5:.1%}'
            .replace('%', '\%'),
            loc='right'
        )
    fig.legend(*axes['A1'].get_legend_handles_labels(), ncol=3)
    save(name)
    pyplot.show()


def cl():
    name = cl.__name__
    rest = series.rest('lin', 'pathological')
    data = {
        'cl': series.cl('cl'),
        'pi': series.cl('pi'),
        'lq': series.cl('lq')
    }
    yrsps = {loop: {i: control.TimeResponseData(
        resp.t,
        resp.y[:rest[0].noutputs],
        issiso=rest[0].issiso
    ) for i, resp in resps.items()} for loop, resps in data.items()}
    ursps = {loop: {i: control.TimeResponseData(
        resp.t,
        resp.y[rest[0].noutputs:],
        issiso=rest[0].issiso
    ) for i, resp in resps.items()} for loop, resps in data.items()}
    refs = {i: control.TimeResponseData(
        resp.t,
        resp.u[:rest[0].noutputs],
        issiso=rest[0].issiso
    ) for i, resp in data['pi'].items()}
    activs = {
        'alpha': numpy.vstack((
            measure(eeg.alpha, rest),
            measure(eeg.alpha, yrsps['pi']),
            measure(eeg.alpha, yrsps['lq']),
            measure(eeg.alpha, yrsps['cl']),
            measure(eeg.alpha, refs)
        )),
        'gamma': numpy.vstack((
            measure(eeg.gamma, rest),
            measure(eeg.gamma, yrsps['pi']),
            measure(eeg.gamma, yrsps['lq']),
            measure(eeg.gamma, yrsps['cl']),
            measure(eeg.gamma, refs)
        )),
        'uampl': numpy.vstack((
            measure(eeg.activity, yrsps['pi']),
            measure(eeg.activity, yrsps['lq']),
            measure(eeg.activity, yrsps['cl'])
        ))
    }
    # plot
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight(4.5 * fig.get_figheight())
    shape = 18, 6
    axes = {
        'A1': pyplot.subplot2grid(shape, (1, 0), 4, 3, title='A)'),
        'B1': pyplot.subplot2grid(shape, (5, 0), 4, 3, title='B)'),
        'C1': pyplot.subplot2grid(shape, (9, 0), 4, 3, title='C)'),
        'A2': pyplot.subplot2grid(shape, (1, 3), 4, 3),
        'B2': pyplot.subplot2grid(shape, (5, 3), 4, 3),
        'C2': pyplot.subplot2grid(shape, (9, 3), 4, 3),
        'D1': pyplot.subplot2grid(shape, (14, 0), 4, 2, title='D)'),
        'D2': pyplot.subplot2grid(shape, (14, 2), 4, 2),
        'D3': pyplot.subplot2grid(shape, (14, 4), 4, 2)
    }
    axes['A1'].set_title('PI control', loc='center')
    axes['B1'].set_title('LQG control', loc='center')
    axes['C1'].set_title('Our method', loc='center')
    axes['D1'].set_ylabel('$\\alpha$-activity')
    axes['D2'].set_ylabel('$\\gamma$-activity')
    axes['D3'].set_ylabel('$u$-amplitude')
    win = 29.5, 30.
    kw = {
        'u': dict(color='C2', label='$u$'),
        'y': dict(color='C3', label='$y$'),
        'y0': dict(color='C0', label='$y_0$'),
        'uamp': dict(color=('C1', 'C4', 'C3')),
        'actv': dict(color=('C0', 'C1', 'C4', 'C3', 'C7')),
    }
    cmp = {
        'alpha': ((1, 4), (2, 4), (3, 4)),
        'gamma': ((1, 4), (2, 4), (3, 4)),
        'uampl': ((0, 1), (1, 2), (0, 2))
    }
    labels = 'resting state', 'PI', 'LQG', 'ours', 'target'
    plot.time_series(axes['A1'], ursps['pi'][0], 's', tbounds=win, **kw['u'])
    plot.time_series(axes['A1'], yrsps['pi'][0], 's', tbounds=win, **kw['y'])
    plot.time_series(axes['B1'], ursps['lq'][0], 's', tbounds=win, **kw['u'])
    plot.time_series(axes['B1'], yrsps['lq'][0], 's', tbounds=win, **kw['y'])
    plot.time_series(axes['C1'], ursps['cl'][0], 's', tbounds=win, **kw['u'])
    plot.time_series(axes['C1'], yrsps['cl'][0], 's', tbounds=win, **kw['y'])
    plot.time_series(axes['A1'], rest[0], 's', tbounds=win, **kw['y0'])
    plot.time_series(axes['B1'], rest[0], 's', tbounds=win, **kw['y0'])
    plot.time_series(axes['C1'], rest[0], 's', tbounds=win, **kw['y0'])
    plot.psd(axes['A2'], *eeg.standard(ursps['pi'][0]), 'Hz', **kw['u'])
    plot.psd(axes['A2'], *eeg.standard(yrsps['pi'][0]), 'Hz', **kw['y'])
    plot.psd(axes['B2'], *eeg.standard(ursps['lq'][0]), 'Hz', **kw['u'])
    plot.psd(axes['B2'], *eeg.standard(yrsps['lq'][0]), 'Hz', **kw['y'])
    plot.psd(axes['C2'], *eeg.standard(ursps['cl'][0]), 'Hz', **kw['u'])
    plot.psd(axes['C2'], *eeg.standard(yrsps['cl'][0]), 'Hz', **kw['y'])
    plot.psd(axes['A2'], *eeg.standard(rest[0]), 'Hz', **kw['y0'])
    plot.psd(axes['B2'], *eeg.standard(rest[0]), 'Hz', **kw['y0'])
    plot.psd(axes['C2'], *eeg.standard(rest[0]), 'Hz', **kw['y0'])
    plot.activity(axes['D1'], activs['alpha'], cmp['alpha'], **kw['actv'])
    plot.activity(axes['D2'], activs['gamma'], cmp['gamma'], **kw['actv'])
    plot.activity(axes['D3'], activs['uampl'], cmp['uampl'], **kw['uamp'])
    for i, (color, label) in enumerate(zip(kw['actv']['color'], labels)):
        axes['D1'].bar(i, 0, color=color, label=label)
    fig.legend(*axes['A1'].get_legend_handles_labels(), ncol=3)
    fig.legend(
        *axes['D1'].get_legend_handles_labels(),
        loc=(.17, .19),
        ncol=5
    )
    save(name)
    pyplot.show()


def rob():
    name = rob.__name__
    rest = series.rest('lin', 'pathological')
    stims = series.stims() | {'mid': series.stim('lin')}
    nu = 1
    dt = 1e-3
    delays = numpy.arange(16)
    P = model.neural_oscillator()
    G = P[:, :nu]
    S = 1. + model.alpha_gamma_filter()
    freq = numpy.arange(1., 80.1, 1.)
    P0rsp = sim.freqresp(P[:, nu:], freq)
    Sresp = sim.freqresp(S, freq) * P0rsp
    Salpha = eeg.alpha(Sresp)
    Sgamma = eeg.gamma(Sresp)

    def loop_rob(loop: control.LinearIOSystem, ref=False):
        if not build.is_stable(loop):
            return 0., numpy.NaN, numpy.NaN
        resp = sim.freqresp(loop, freq)
        if key in ('pi', 'lq'):
            rrsp = control.FRD(
                resp.fresp[:P.noutputs, :P.noutputs].reshape(
                    resp.noutputs,
                    resp.noutputs,
                    -1
                ),
                resp.frequency,
            ) * Sresp
            resp = control.FRD(
                numpy.hstack((rrsp.fresp, resp.fresp[:, P.noutputs:])),
                resp.frequency
            )
        loop = control.StateSpace(
            loop.A,
            loop.B.mean(-1),
            loop.C[:P.noutputs],
            loop.D[:P.noutputs].mean(-1),
            loop.dt
        )
        margin = control.stability_margins(loop)[2]
        erralpha = 100. * (eeg.alpha(resp) - Salpha) / Salpha
        errgamma = 100. * (eeg.gamma(resp) - Sgamma) / Sgamma
        return margin, erralpha, errgamma
    builders = {
        'pi': build.pi_cl,
        'lq': build.lqg_cl,
        'cl': build.alpha_gamma_cl
    }
    margins = {key: numpy.empty(len(delays)) for key in builders}
    eralpha = {key: numpy.empty(len(delays)) for key in builders}
    ergamma = {key: numpy.empty(len(delays)) for key in builders}
    for i, delay in enumerate(delays):
        for key, builder in builders.items():
            loop: control.LinearIOSystem = builder(
                P,
                nu,
                delay,
                dt if delay else None
            )[:P.noutputs, :]
            (
                margins[key][i],
                eralpha[key][i],
                ergamma[key][i]
            ) = loop_rob(loop, key in ('pi', 'lq'))
    # figure
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight(4.5 * fig.get_figheight())
    shape = 17, 3
    span = 4, 1
    axes = {
        'A1': pyplot.subplot2grid(shape, (1, 0), *span, title='A)'),
        'B1': pyplot.subplot2grid(shape, (5, 0), *span, title='B)'),
        'C1': pyplot.subplot2grid(shape, (9, 0), *span, title='C)'),
        'D1': pyplot.subplot2grid(shape, (13, 0), *span, title='D)'),
        'A2': pyplot.subplot2grid(shape, (1, 1), *span),
        'B2': pyplot.subplot2grid(shape, (5, 1), *span),
        'C2': pyplot.subplot2grid(shape, (9, 1), *span),
        'D2': pyplot.subplot2grid(shape, (13, 1), *span),
        'A3': pyplot.subplot2grid(shape, (1, 2), *span),
        'B3': pyplot.subplot2grid(shape, (5, 2), *span),
        'C3': pyplot.subplot2grid(shape, (9, 2), *span),
        'D3': pyplot.subplot2grid(shape, (13, 2), *span),
    }
    kwargs = {
        'pi': dict(color='C1', label='PI'),
        'lq': dict(color='C4', label='LQG'),
        'cl': dict(color='C3', label='ours')
    }
    for i, ylabel in enumerate((
            'stability margin',
            '$\\alpha$-error [\%]',
            '$\gamma$-error [\%]'
    )):
        axes[f'D{i+1}'].set_xlabel('delay [ms]')
        for row in 'A', 'B', 'C', 'D':
            axes[f'{row}{i+1}'].set_ylabel(ylabel)
    for row in 'A', 'B', 'C', 'D':
        axes[f'{row}1'].set_ylim(0., 1.5)
        axes[f'{row}2'].set_ylim(-50., 250.)
        axes[f'{row}3'].set_ylim(-50., 250.)
        axes[f'{row}2'].grid(axis='y')
        axes[f'{row}3'].grid(axis='y')
    for method in ('lq', 'pi', 'cl'):
        mrg = margins[method]
        mask = numpy.isfinite(mrg)
        kw = kwargs[method]
        axes['A1'].plot(delays[mask], mrg[mask], **kw)
        axes['A2'].plot(delays, eralpha[method], **kw)
        axes['A3'].plot(delays, ergamma[method], **kw)
    uaxs = {
        'high': 'B',
        'mid': 'C',
        'low': 'D'
    }
    for uamp in stims:
        Gfits = numpy.empty(len(rest), dtype=object)
        rmse = numpy.empty(len(rest))
        for i in sorted(rest):
            y0rp = rest[i]
            yrsp = stims[uamp][i]
            (
                Gfits[i],
                ursp,
                f,
                Py0,
                Pyy,
                Puu,
                Pgg,
                rmse[i],
                ar
            ) = series.single_fit(G, y0rp, yrsp)
        axes[f'{uaxs[uamp]}2'].set_title(
            f'RMSE = {rmse.mean():.1%}$\pm${rmse.var()**.5:.1%}'
            .replace('%', '\%'),
            loc='center'
        )
        for key, builder in builders.items():
            margins = numpy.empty((len(Gfits), len(delays)))
            eralpha = numpy.empty((len(Gfits), len(delays)))
            ergamma = numpy.empty((len(Gfits), len(delays)))
            for i, Gfit in enumerate(Gfits):
                for j, delay in enumerate(delays):
                    loop: control.LinearIOSystem = builder(
                        P,
                        nu,
                        delay,
                        dt if delay else None,
                        Gfit
                    )[:P.noutputs, :]
                    (
                        margins[i, j],
                        eralpha[i, j],
                        ergamma[i, j]
                    ) = loop_rob(loop, key in ('pi', 'lq'))
            for i, ydata in enumerate((margins, eralpha, ergamma)):
                ycols = tuple(col[numpy.isfinite(col)] for col in ydata.T)
                ymean = numpy.fromiter(
                    (col.mean() for col in ycols),
                    ydata.dtype
                )
                ysdev = numpy.sqrt(
                    numpy.fromiter((col.var() for col in ycols), ydata.dtype)
                )
                axes[f'{uaxs[uamp]}{i+1}'].errorbar(
                    delays,
                    ymean,
                    yerr=ysdev,
                    **kwargs[key]
                )
    fig.legend(*axes['A1'].get_legend_handles_labels(), ncol=3)
    save(name)
    pyplot.show()


def ctcl():
    name = ctcl.__name__
    rest = series.rest('ct', 'pathological')
    stim = series.stim('ct')
    data = series.ctcl()
    P = model.cortico_thalamic(1e-3)
    G = P.params['lin'][:, :P.params['nu']]
    yrsps = {i: control.TimeResponseData(
        resp.t,
        resp.y[:P.noutputs],
        issiso=P.issiso()
    ) for i, resp in data.items()}
    ursps = {i: control.TimeResponseData(
        resp.t,
        resp.y[P.noutputs:],
        issiso=P.issiso()
    ) for i, resp in data.items()}
    S = 1. + model.alpha_gamma_filter()
    refs = {i: sim.lsim(S, y0rp.t, y0rp.y) for i, y0rp in rest.items()}
    activs = {
        'alpha': numpy.vstack((
            measure(eeg.alpha, rest),
            measure(eeg.alpha, yrsps),
            measure(eeg.alpha, refs)
        )),
        'gamma': numpy.vstack((
            measure(eeg.gamma, rest),
            measure(eeg.gamma, yrsps),
            measure(eeg.gamma, refs)
        ))
    }
    activs['alpha'] = activs['alpha'][..., activs['alpha'][1] < .015]
    activs['gamma'] = activs['gamma'][..., activs['gamma'][1] < .010]
    Gfit, y0rp, yrsp, ursp, f, Py0, Pyy, Puu, Pgg, rmse, ar = series.fit_model(
        G,
        rest,
        stim
    )
    freq = numpy.linspace(1., f[-1], 1000)
    Gresp = sim.freqresp(G, freq)
    Gfitr = sim.freqresp(Gfit, freq)
    # plot
    fig = pyplot.figure(name)
    fig.set_figwidth(2. * fig.get_figwidth())
    fig.set_figheight(3. * fig.get_figheight())
    shape = 18, 4
    span = 6, 2
    axes = {
        'A': pyplot.subplot2grid(shape, (3, 0), *span, title='A)'),
        'B': pyplot.subplot2grid(shape, (3, 2), *span, title='B)'),
        'C': pyplot.subplot2grid(shape, (9, 0), *span, title='C)'),
        'D1': pyplot.subplot2grid(shape, (9, 2), 6, 1, title='D)'),
        'D2': pyplot.subplot2grid(shape, (9, 3), 6, 1)
    }
    kwargs = {
        'G': dict(color='k', label='$G$'),
        'Gfit': dict(color='C9', linestyle='--', label='$\\tilde{G}$'),
        'Pgg': dict(color='C6', label='$|\hat{g}|^2$'),
        'u': dict(color='C2', label='$u$'),
        'y': dict(color='C3', label='$y$'),
        'y0': dict(color='C0', label='$y_0$'),
        'actv': dict(color=('C0', 'C3', 'C7'))
    }
    axes['A'].set_title(
        f'RMSE = {rmse.mean():.1%}$\pm${rmse.var()**.5:.1%}'
        .replace('%', '\%'),
        loc='right'
    )
    axes['D1'].set_ylabel('$\\alpha$-activity')
    axes['D2'].set_ylabel('$\\gamma$-activity')
    compare = ((0, 1), (1, 2))
    labels = 'resting state', 'stimulated state', 'target'
    plot.bode(axes['A'], axes['B'], Gresp, 'Hz', **kwargs['G'])
    plot.psd(axes['A'], f, Pgg, 'Hz', **kwargs['Pgg'])
    plot.bode(axes['A'], axes['B'], Gfitr, 'Hz', **kwargs['Gfit'])
    plot.psd(axes['C'], *eeg.standard(ursps[0]), 'Hz', **kwargs['u'])
    plot.psd(axes['C'], *eeg.standard(yrsps[0]), 'Hz', **kwargs['y'])
    plot.psd(axes['C'], *eeg.standard(rest[0]), 'Hz', **kwargs['y0'])
    plot.activity(axes['D1'], activs['alpha'], compare, **kwargs['actv'])
    plot.activity(axes['D2'], activs['gamma'], compare, **kwargs['actv'])
    for i, (color, label) in enumerate(zip(kwargs['actv']['color'], labels)):
        axes['D1'].bar(i, 0, color=color, label=label)
    h1, l1 = axes['A'].get_legend_handles_labels()
    h2, l2 = axes['C'].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=6)
    fig.legend(
        *axes['D1'].get_legend_handles_labels(),
        loc='lower right',
        ncol=3
    )
    save(name)
    pyplot.show()


def stab():
    name = stab.__name__
    with open('stab.csv') as file:
        file.readline()
        s = ((float(x) for x in line.split(',')) for line in file)
        a, tau3, tau5, tau10 = zip(*s)
    x = a[0]
    y = 0
    width = a[-1] - x
    height = max(tau3[-1], tau5[-1], tau10[-1]) - y
    rect = patches.Rectangle((x, y), width, height, color='C7')
    fig = pyplot.figure(name)
    fig.set_figwidth(fig.get_figwidth() * 1.5)
    #fig.set_figheight(fig.get_figheight() * 2)
    ax = pyplot.subplot2grid((1, 3), (0, 0), 1, 2)
    ax.set_xlabel('a')
    ax.set_ylabel('pole mag. [dB]')
    ax.add_patch(rect)
    ax.plot(a, control.mag2db(tau3), 'k', label='$\\tau$ = 3ms')
    ax.plot(a, control.mag2db(tau5), 'k--', label='$\\tau$ = 5ms')
    ax.plot(a, control.mag2db(tau10), 'k:', label='$\\tau$ = 10ms')
    ax.text(.6, .075, 'unstable')
    ax.text(.6, -.2, 'stable')
    ax.set_ylim(-.25, .25)
    fig.legend(*ax.get_legend_handles_labels())
    save(name)
    pyplot.show()


def tf():
    name = tf.__name__
    series.tf(0)
