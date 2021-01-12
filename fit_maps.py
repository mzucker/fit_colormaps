from collections import namedtuple

import sys
import argparse
import os

import quadprog

import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon

import numpy as np

FitOptions = namedtuple('FitOptions',
                        'fit_type, numer_degree, denom_degree, clip_reconstruction, loss')

OutputOptions = namedtuple('OutputOptions',
                           'console_file, glsl_file, '
                           'plot_title, image_filename, '
                           'domain, range, plot_shape, min_samples')

DEFAULT_FIT_OPTIONS = FitOptions(fit_type='poly',
                                 numer_degree=4,
                                 denom_degree=0,
                                 clip_reconstruction=None,
                                 loss='minimax')

DEFAULT_OUTPUT_OPTIONS = OutputOptions(console_file=None,
                                       glsl_file=None,
                                       plot_title=None,
                                       image_filename=None,
                                       domain=(0., 1.),
                                       range=None,
                                       plot_shape=False,
                                       min_samples=256)

LOSS_NAMES = dict(rmse='RMSE', minimax='max error')

######################################################################

def coeffs_per_degree(fit_type, degree):

    assert fit_type in ['poly', 'fourier']
    assert degree >= 0
    
    if fit_type == 'poly':
        return degree + 1
    else:
        return 2 * degree + 1

def count_coeffs(fit_opts):

    numer_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.numer_degree)
    denom_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.denom_degree) - 1

    return numer_coeffs, numer_coeffs + denom_coeffs
    
######################################################################

def get_fourier_basis(x, degree):

    n = 2*degree + 1

    A = np.zeros( (len(x), n) )
    A[:, n-1] = 1

    for i in range(1, degree+1):
        theta = x*2*np.pi*i
        A[:, n-2*i] = np.sin(theta)
        A[:, n-2*i-1] = np.cos(theta)

    return A

######################################################################

def get_polynomial_basis(x, degree):

    n = np.arange(degree, -1, -1)

    A = x.reshape(-1, 1) ** n.reshape(1, -1)

    return A
    
######################################################################

def get_bases(x, fit_opts, force_matrix=False):

    if fit_opts.fit_type == 'poly' and not force_matrix:
        return (x, x)

    max_degree = max(fit_opts.numer_degree, fit_opts.denom_degree)

    if fit_opts.fit_type == 'fourier':

        basis = get_fourier_basis(x, max_degree)

    else:

        basis = get_polynomial_basis(x, max_degree)

    numer_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.numer_degree)
    denom_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.denom_degree)

    return (basis[:, -numer_coeffs:], basis[:, -denom_coeffs:])

######################################################################

def reconstruct_base(p, basis, fit_type):

    assert fit_type in ['poly', 'fourier']

    if fit_type == 'poly':
        return np.polyval(p, basis)
    else:
        return np.dot(basis, p)

######################################################################

def split_params(p, fit_opts):
    
    numer_coeffs, total_coeffs = count_coeffs(fit_opts)
    assert len(p) == total_coeffs

    pnum = p[:numer_coeffs]
    pdenom = np.concatenate((p[numer_coeffs:], [ np.ones_like(p[0]) ] ), axis=0)

    return pnum, pdenom

######################################################################

def reconstruct(p, bases, fit_opts):

    numer_basis, denom_basis = bases

    if fit_opts.denom_degree:
        p_num, p_denom = split_params(p, fit_opts)
        rnum = reconstruct_base(p_num, numer_basis, fit_opts.fit_type)
        rdenom = reconstruct_base(p_denom, denom_basis, fit_opts.fit_type)
        y = rnum / rdenom
    else:
        y = reconstruct_base(p, numer_basis, fit_opts.fit_type)

    if fit_opts.clip_reconstruction is not None:
        return np.clip(y, *fit_opts.clip_reconstruction)
    else:
        return y

######################################################################

def residual(p, bases, y, fit_opts):

    return reconstruct(p, bases, fit_opts) - y

######################################################################

def loss(p, bases, y, fit_opts):

    e = residual(p, bases, y, fit_opts)
    
    if fit_opts.loss == 'rmse':
        return np.sqrt(np.dot(e, e)/len(e))
    else:
        return np.abs(e).max()
    
######################################################################    
    
def fit_single_channel(cidx, bases, y, fit_opts, output_opts):

    loss_name = LOSS_NAMES[fit_opts.loss]
    args = (bases, y, fit_opts)

    all_p0 = []

    numer_basis, denom_basis = bases

    # step 1/3: global initial fit

    if fit_opts.fit_type == 'poly':
        p0 = np.polyfit(numer_basis, y, fit_opts.numer_degree)
    else:
        p0, _, _, _ = np.linalg.lstsq(numer_basis, y, rcond=None)

    if not fit_opts.denom_degree:
        
        all_p0 = [p0]
        
    else:

        numer_coeffs, total_coeffs = count_coeffs(fit_opts)
        denom_coeffs = total_coeffs - numer_coeffs
        
        p0 = np.hstack([p0, np.zeros(denom_coeffs, dtype=p0.dtype)])
        all_p0 = [p0]

        if fit_opts.fit_type == 'poly':
            assert len(numer_basis.shape) == 1
            assert np.all(numer_basis == denom_basis)
            force_numer_basis, force_denom_basis = get_bases(numer_basis, fit_opts, True)
        else:
            force_numer_basis, force_denom_basis = numer_basis, denom_basis

        assert len(force_numer_basis.shape) == 2
        assert len(force_denom_basis.shape) == 2
        assert len(y.shape) == 1

        npoints, numer_coeffs2 = force_numer_basis.shape
        npoints2, denom_coeffs_plus_one = force_denom_basis.shape

        assert npoints == npoints2
        assert numer_coeffs2 == numer_coeffs
        assert denom_coeffs_plus_one == denom_coeffs + 1

        A = np.hstack( (force_numer_basis, -force_denom_basis[:, :-1] * y.reshape(-1, 1)) )
        assert A.shape == (npoints, numer_coeffs + denom_coeffs)

        assert np.all(force_numer_basis[:, -1] == 1)
        assert np.all(force_denom_basis[:, -1] == 1)
        b = y

        C = -A
        C[:, :numer_coeffs] = 0

        CT = C.transpose()

        lambda_I = 1e-8 * np.eye(numer_coeffs + denom_coeffs)

        G = np.dot(A.T, A) + lambda_I
        a = np.dot(A.T, b)

        step = 0.025
        epsrng = np.arange(step, 1, step)
        #epsrng = 1 - np.exp(-np.arange(1, 10))
        all_eps = []

        for eps in epsrng:

            lb = np.ones_like(C[:,0])*eps - 1

            p0, _, _, _, _, _ = quadprog.solve_qp(G, a, CT, lb)

            denom = np.dot(C, p0) + 1
            #print('denominator:', denom.min(), denom.max())

            all_p0.append(p0)
            all_eps.append(eps)

    all_p0_loss = np.array([loss(p0, *args) for p0 in all_p0])

    best_idx = all_p0_loss.argmin()

    best_p0 = all_p0[best_idx]
    best_e0 = all_p0_loss[best_idx]

    #all_p0 = all_p0[best_idx:best_idx+1]

    best_p1 = None
    best_e1 = None

    for p0 in all_p0:

        e0 = loss(p0, *args)

        p1 = p0
        e1 = e0

        if (fit_opts.loss == 'minimax' or fit_opts.denom_degree) :

            p1 = p0.copy()

            if fit_opts.denom_degree:
                # step 2/3: local search to do least-squares fit for rational
                # polynomial or Fourier series, using output of step 1 as
                # initial guess.
                res = scipy.optimize.least_squares(residual, p1,
                                                   ftol=1e-5, xtol=1e-5,
                                                   method='dogbox', args=args)
                p1 = res.x

            if fit_opts.loss == 'minimax':
                # step 3/3: local search to do minimax optimization, starting from
                # output of step 1 or 2. 
                res = scipy.optimize.minimize(loss, p1,
                                              method='Nelder-Mead',
                                              args=args)
                p1 = res.x

            e1 = loss(p1, *args)

        updated = (best_e1 is None or e1 < best_e1)

        if updated:
            best_e1 = e1
            best_p1 = p1
            updated = True
            
        print('  - channel {} {}: {:.7f} -> {:.7f} ({:6.2f}%){}'.format(
            cidx, loss_name, best_e0, e1, 100.0*e1/best_e0, ' *** new min ***' if updated else ''),
              file=output_opts.console_file)

    print(file=output_opts.console_file)
    
    return best_p1

######################################################################

def fit(key, data, fit_opts, output_opts):

    npoints, nchannels = data.shape
    
    assert nchannels in range(1, 5)
    
    x = np.linspace(0, 1, npoints, endpoint=False)

    _, total_coeffs = count_coeffs(fit_opts)

    print('Fitting {}-parameter model to {} points in {}...\n'.format(
        total_coeffs, npoints, key), file=output_opts.console_file)

    bases = get_bases(x, fit_opts)
    
    coeffs = []

    for cidx in range(nchannels):
        channel = data[:, cidx]
        p = fit_single_channel(cidx, bases,
                               channel, fit_opts, output_opts)
        coeffs.append(p)

    return np.array(coeffs)

######################################################################

def get_rtype(nchannels):
    
    if nchannels == 1:
        return 'float'
    else:
        return 'vec' + str(nchannels)

######################################################################

def vecstr(v):
    fmt = '{:.16g}'
    if len(v) == 1:
        return fmt.format(v[0])
    else:
        interior = ', '.join(fmt.format(vi) for vi in v)
        return get_rtype(len(v)) + '(' + interior + ')'
    
######################################################################

def glslify_base(p, fit_opts, prefix):

    assert fit_opts.fit_type in ['poly', 'fourier']

    nchannels = p.shape[1]
    rtype = get_rtype(nchannels)

    result = ''
    rval = ''

    if fit_opts.fit_type == 'poly':

        degree = len(p) - 1
        
        degree0_is_1 = False
        
        for i, v in enumerate(reversed(p)):
            if i == 0 and np.all(v == 1):
                degree0_is_1 = True
            else:
                result += '    const {} {}{} = {};\n'.format(
                    rtype, prefix, i, vecstr(v))
                
        result += '\n'
        
        for i in range(degree + 1):
            if i == 0 and degree0_is_1:
                rval += '1.0'
            else:
                rval += '{}{}'.format(prefix, i)
            if i < degree:
                rval += '+t*'
            if i < degree-1:
                rval += '('
                
        rval += ')'*(degree-1)
        
    else:

        n = len(p)
        degree = (n - 1) // 2
        assert n == 2*degree + 1

        if np.all(p[-1] == 1):
            rval += '1.0'
        else:
            rval += vecstr(p[0])

        rval += prefix

        if np.all(p[-1] == 1):
            if rtype == 'float':
                c0 = '1.0'
            else:
                c0 = '{}(1.0)'.format(rtype)
        else:
            c0 = vecstr(p[-1])

        result += '    {} {} = {};\n'.format(rtype, prefix, c0)

        for i in range(1, degree+1):
            
            s = p[n-2*i]
            c = p[n-2*i-1]
            
            result += '    {} += {}*cs{}.x;\n'.format(prefix, vecstr(c), i)
            result += '    {} += {}*cs{}.y;\n'.format(prefix, vecstr(s), i)

        result += '\n'

    if fit_opts.denom_degree:
        if fit_opts.fit_type == 'poly':
            vname = dict(n='num', d='denom')[prefix]
            result += '    {} {} = {};\n\n'.format(rtype, vname, rval)
    else:
        if fit_opts.fit_type == 'poly':
            result += '    return ' + rval + ';\n\n'
        else:
            result += '    return ' + prefix + ';\n\n'

    return result

######################################################################

def glslify_single(key, coeffs, fit_opts, output_opts):

    nchannels = coeffs.shape[1]

    function = '{} {}(float t) {{\n\n'.format(get_rtype(nchannels), key)

    max_degree = max(fit_opts.numer_degree, fit_opts.denom_degree)

    if fit_opts.fit_type == 'fourier':
        function += '    t *= 6.283185307179586;\n\n'
        for i in range(1, max_degree+1):
            if i == 1:
                timest = 't'
            else:
                timest = '{}.0*t'.format(i)
            function += '    vec2 cs{} = vec2(cos({}), sin({}));\n'.format(
                i, timest, timest)
        function += '\n'

    if fit_opts.denom_degree:
        p_num, p_denom = split_params(coeffs.T, fit_opts)
        function += glslify_base(p_num, fit_opts, 'n')
        function += glslify_base(p_denom, fit_opts, 'd')
        if fit_opts.fit_type == 'fourier':
            function += '    return n/d;\n\n'
        else:
            function += '    return num/denom;\n\n'
    else:
        function += glslify_base(coeffs.T, fit_opts, 'p')
            
    function += '}\n'

    print(function, file=output_opts.glsl_file)

######################################################################

def glslify(results, fit_opts, output_opts):

    if output_opts.glsl_file is None:
        return
    
    for key, _, coeffs in results:
        glslify_single(key, coeffs, fit_opts, output_opts)
    

######################################################################

def plot_single(key, data, coeffs, fit_opts, output_opts):

    npoints, nchannels = data.shape

    assert len(coeffs) == nchannels
        
    chan_colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [.75, .75, .75]], dtype=float)

    x0, x1 = output_opts.domain
    xm = 0.05 * (x1 - x0)

    if fit_opts.fit_type == 'fourier':
        x = np.linspace(x0, x1, npoints+1, endpoint=True)
        data = np.vstack((data, [data[0]]))
    else:
        x = np.linspace(x0, x1, npoints, endpoint=False)

    if output_opts.range is None:
        y0 = np.floor(data.min() + 1e-7)
        y1 = np.ceil(data.max() - 1e-7 )
    else:
        y0, y1 = output_opts.range

    x_fine = x
    
    if npoints < output_opts.min_samples:
        factor = int(np.ceil(output_opts.min_samples / npoints))
        x_fine = np.linspace(x0, x1, npoints*factor+1, endpoint=True)

    bases = get_bases(x, fit_opts)
    bases_fine = get_bases(x_fine, fit_opts)
        
    ym = 0.05 * (y1 - y0)

    max_err = 0

    reconstructions = []
    losses = []

    regular_data = (nchannels != 2 or not output_opts.plot_shape)

    for cidx, p in enumerate(coeffs):

        color = chan_colors[cidx] * 0.75
        
        channel = data[:, cidx]

        px = reconstruct(p, bases, fit_opts)
        losses.append(loss(p, bases, channel, fit_opts))

        if x_fine is not x:
            px_fine = reconstruct(p, bases_fine, fit_opts)
        else:
            px_fine = px

        reconstructions.append(px_fine)

        if regular_data:
            plt.plot(x, channel, color=color)
            plt.plot(x_fine, px_fine, ':', color=0.5*color, linewidth=2)
        
    if regular_data:
        plt.xlim(x0-xm, x1+xm)
        plt.ylim(y0-ym, y1+ym)
        tx, ty = x1, y0
        ha = 'right'
        va = 'bottom'
        effects=[
            path_effects.Stroke(linewidth=8, foreground=[1, 1, 1, 0.8]),
            path_effects.Normal()
        ]
    else:
        xshift = 0#0.5*(y1 - y0)
        axes = plt.gca()
        axes.add_patch(Polygon(data, ec='none', fc=[0.8]*3))
        #plt.plot(data[:, 0]-xshift, data[:, 1], 'k-', linewidth=1)
        plt.plot(reconstructions[0]+xshift, reconstructions[1], 'b-', linewidth=0.5)
        plt.plot([y0, y1, y1, y0], [y0, y0, y1, y1], '.', color='none')
        plt.axis('equal')
        plt.axis('off')
        tx, ty = 0.5*(y0+y1), y0-0.025*(y1-y0)
        ha = 'center'
        va = 'top'
        effects=None

    loss_name = LOSS_NAMES[fit_opts.loss]
    losses = np.array(losses)
    #with np.printoptions(formatter={'all':lambda x: '{:.3g}'.format(x)}) as opts:
    lstr = np.array2string(losses, formatter=dict(all=lambda x:'{:.3g}'.format(x)), separator=', ')
    plt.text(tx, ty, '{}: per-channel {}={}, total={:.3g}'.format(
        key, loss_name, lstr, losses.sum()),
             ha=ha, va=va, path_effects=effects)

######################################################################    
        
def plot(results, fit_opts, output_opts):

    if output_opts.image_filename == '-':
        return

    n = len(results)

    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n/rows))
    
    plt.figure(figsize=(8, 4.5))
    
    plt.suptitle(output_opts.plot_title)

    for i, (key, mapdata, coeffs) in enumerate(results):
        plt.subplot(rows, cols, i+1)
        plot_single(key, mapdata, coeffs, fit_opts, output_opts)

    if output_opts.image_filename is None:
        plt.show()
    else:
        plt.savefig(output_opts.image_filename, dpi=144)
        print('wrote', output_opts.image_filename,
              file=output_opts.console_file)

######################################################################

def fill_tuple(tclass, lookup):
    return tclass(*[lookup[field] for field in tclass._fields])

######################################################################

def domain_range(s):
    x0, x1 = map(float, s.split(','))
    return (x0, x1)

######################################################################

def parse_cmdline():

    parser = argparse.ArgumentParser('minimax polynomial fitting of colormaps')

    parser.add_argument('mapfiles', nargs='*',
                        metavar='PALETTE.txt',
                        help='map files to process')

    parser.add_argument('-t', dest='fit_type',
                        choices=['fourier', 'poly'],
                        default=DEFAULT_FIT_OPTIONS.fit_type,
                        help='type of fit')

    parser.add_argument('-n', dest='numer_degree',
                        metavar='N',
                        type=int, default=DEFAULT_FIT_OPTIONS.numer_degree,
                        help='degree of numerator')

    parser.add_argument('-d', dest='denom_degree',
                        metavar='N',
                        type=int, default=DEFAULT_FIT_OPTIONS.denom_degree,
                        help='degree of denominator')
    
    parser.add_argument('-c', dest='clip_reconstruction',
                        metavar='Y0,Y1',
                        type=domain_range,
                        default=DEFAULT_FIT_OPTIONS.clip_reconstruction,
                        help='data clipping limits (default: none)')

    parser.add_argument('-l', dest='loss',
                        choices=('minimax', 'rmse'),
                        default=DEFAULT_FIT_OPTIONS.loss,
                        help='loss function')

    parser.add_argument('-g', dest='glsl_filename',
                        metavar='FILENAME.glsl',
                        default=None,
                        help='GLSL output file name')

    parser.add_argument('-q', dest='quiet',
                        action='store_true',
                        help='no console output')
    
    parser.add_argument('-T', dest='plot_title',
                        metavar='TITLESTRING',
                        default=DEFAULT_OUTPUT_OPTIONS.plot_title,
                        help='title for plots')

    parser.add_argument('-p', dest='image_filename',
                        default=DEFAULT_OUTPUT_OPTIONS.image_filename,
                        help='image filename or - to suppress plotting')
    
    parser.add_argument('-x', dest='domain',
                        metavar='X0,X1', default=DEFAULT_OUTPUT_OPTIONS.domain,
                        type=domain_range, help='domain for plotting')

    parser.add_argument('-y', dest='range',
                        metavar='Y0,Y1', default=DEFAULT_OUTPUT_OPTIONS.range,
                        type=domain_range, help='range for plotting')
    
    parser.add_argument('-m', dest='min_samples',
                        metavar='N', default=DEFAULT_OUTPUT_OPTIONS.min_samples,
                        type=int, help='min number of points in domain for plotting')

    assert not DEFAULT_OUTPUT_OPTIONS.plot_shape

    parser.add_argument('-s', dest='plot_shape',
                        action='store_true',
                        help='plot data as 2D shape')


    opts = parser.parse_args()

    mapfiles = opts.mapfiles
    del opts.mapfiles

    if mapfiles == []:
        mapfiles = 'inferno magma plasma viridis turbo hsluv hpluv'.split()

    fit_opts = fill_tuple(FitOptions, vars(opts))

    if opts.plot_title is None:
        rlabel = '/{} rational'.format(fit_opts.denom_degree) if fit_opts.denom_degree else ''
        if fit_opts.fit_type == 'poly':
            opts.plot_title = 'Degree {}{} polynomial'.format(
                fit_opts.numer_degree, rlabel)
        else:
            opts.plot_title = 'Order {}{} Fourier series'.format(
                fit_opts.numer_degree, rlabel)
        if fit_opts.clip_reconstruction is not None:
            opts.plot_title += ' clipped to [{:g},{:g}]'.format(
                *fit_opts.clip_reconstruction)
        _, total_coeffs = count_coeffs(fit_opts)
        opts.plot_title += ' ({} coefficients per channel)'.format(total_coeffs)

    opts.console_file = sys.stdout
            
    if opts.glsl_filename is None:
        opts.glsl_file = None
    elif opts.glsl_filename == '-':
        opts.glsl_file = sys.stdout
        opts.console_file = sys.stderr
    else:
        opts.glsl_file = open(opts.glsl_filename, 'w')

    if opts.quiet:
        opts.console_file = open(os.devnull, 'w')
        
    output_opts = fill_tuple(OutputOptions, vars(opts))
            
    return mapfiles, fit_opts, output_opts

######################################################################

def main():

    mapfiles, fit_opts, output_opts = parse_cmdline()

    print('Fit options:\n', file=output_opts.console_file)
    for key, val in fit_opts._asdict().items():
        print('  {:20} {}'.format(key+':', val),
              file=output_opts.console_file)
    print(file=output_opts.console_file)

    results = []
    
    for i, filename in enumerate(mapfiles):
        datafilename = 'data/' + filename + '.txt'
        if not os.path.exists(filename) and os.path.exists(datafilename):
            filename = datafilename
        key, _ = os.path.splitext(os.path.basename(filename))
        mapdata = np.genfromtxt(filename)
        if len(mapdata.shape) == 1:
            mapdata = mapdata.reshape(-1, 1)
        coeffs = fit(key, mapdata, fit_opts, output_opts)
        results.append((key, mapdata, coeffs))

    glslify(results, fit_opts, output_opts)

    plot(results, fit_opts, output_opts)
    

######################################################################
    
if __name__ == '__main__':
    main()
