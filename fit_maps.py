from collections import namedtuple

import sys
import argparse
import os

import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon

import numpy as np

FitOptions = namedtuple('FitOptions',
                        'fit_type, degree, is_rational, clip_reconstruction')

OutputOptions = namedtuple('OutputOptions',
                           'console_file, glsl_file, '
                           'plot_title, image_filename, '
                           'domain, range, plot_shape, min_samples')

DEFAULT_FIT_OPTIONS = FitOptions(fit_type='poly',
                                 degree=4,
                                 is_rational=True,
                                 clip_reconstruction=None)

DEFAULT_OUTPUT_OPTIONS = OutputOptions(console_file=None,
                                       glsl_file=None,
                                       plot_title=None,
                                       image_filename=None,
                                       domain=(0., 1.),
                                       range=None,
                                       plot_shape=False,
                                       min_samples=256)

######################################################################

def num_coeffs(fit_opts):

    assert fit_opts.fit_type in ['poly', 'fourier']
    
    if fit_opts.fit_type == 'poly':
        nbase = fit_opts.degree + 1
    else:
        nbase = 2 * fit_opts.degree + 1
    
    if not fit_opts.is_rational:
        return nbase, nbase
    else:
        return nbase, 2*nbase - 1

######################################################################
    
def initial_fit(basis, y, fit_opts):

    assert fit_opts.fit_type in ['poly', 'fourier']
    
    if fit_opts.fit_type == 'poly':
        p = np.polyfit(basis, y, fit_opts.degree)
    else:
        p, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)

    if not fit_opts.is_rational:
        return p
    else:
        return np.hstack((p, np.zeros_like(p[1:])))

######################################################################

def get_basis(x, fit_opts):

    if fit_opts.fit_type == 'fourier':
        
        n = 2*fit_opts.degree + 1

        A = np.zeros( (len(x), n) )
        A[:, n-1] = 1

        for i in range(1, fit_opts.degree+1):
            theta = x*2*np.pi*i
            A[:, n-2*i] = np.sin(theta)
            A[:, n-2*i-1] = np.cos(theta)

        return A

    else:

        return x

######################################################################

def reconstruct_base(p, basis, fit_opts):

    assert fit_opts.fit_type in ['poly', 'fourier']

    if fit_opts.fit_type == 'poly':
        return np.polyval(p, basis)
    else:
        return np.dot(basis, p)

######################################################################

def split_params(p, fit_opts):
    
    nbase, ntotal = num_coeffs(fit_opts)
    assert len(p) == ntotal

    pnum = p[:nbase]
    pdenom = np.concatenate((p[nbase:], [ np.ones_like(p[0]) ] ), axis=0)

    return pnum, pdenom

######################################################################

def reconstruct(p, basis, fit_opts):

    if fit_opts.is_rational:
        p_num, p_denom = split_params(p, fit_opts)
        rnum = reconstruct_base(p_num, basis, fit_opts)
        rdenom = reconstruct_base(p_denom, basis, fit_opts)
        y = rnum / rdenom
    else:
        y = reconstruct_base(p, basis, fit_opts)

    if fit_opts.clip_reconstruction is not None:
        return np.clip(y, *fit_opts.clip_reconstruction)
    else:
        return y
        
######################################################################    
    
def fit_single_channel(cidx, x, basis, y, fit_opts, output_opts):

    def curve_fit_f(x, *p):
        return reconstruct(np.array(p), basis, fit_opts)

    def err(p):
        return y - reconstruct(p, basis, fit_opts)
    
    def l1err(p):
        e = err(p)
        return np.abs(e).max()

    def rmse(p):
        e = err(p)
        return np.sqrt(np.dot(e, e)/len(e))

    # step 1/3: global least-squares fit to regular (not rational)
    # polynomial or Fourier series - for rational versions just
    # initializes denominator to the constant function f(x) = 1
    p0 = initial_fit(basis, y, fit_opts)
    guess = p0

    if fit_opts.is_rational:
        # step 2/3: local search to do least-squares fit for rational
        # polynomial or Fourier series, using output of step 1 as
        # initial guess.
        npts = len(x)
        _, nparams = num_coeffs(fit_opts)
        guess, _ = scipy.optimize.curve_fit(curve_fit_f,
                                            x, y, guess,
                                            ftol=1e-5,
                                            xtol=1e-5,
                                            method='dogbox')

    # step 3/3: local search to do minimax optimization, starting from
    # output of step 1 or 2. in my experience, gradient-based optimizers
    # don't do so hot minimizing maximum error (infinity norm), so using
    # a derivative-free optimizer like nelder mead is safest.
    res = scipy.optimize.minimize(l1err, guess, method='Nelder-Mead')
    p1 = res.x

    e0 = l1err(p0)
    e1 = l1err(p1)
    
    print('  - channel {} max error: {:.7f} -> {:.7f} ({:.2f}%)'.format(
        cidx, e0, e1, 100.0*e1/e0), file=output_opts.console_file)
    
    return p1

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
        
        assert len(p) == fit_opts.degree + 1
        
        degree0_is_1 = False
        
        for i, v in enumerate(reversed(p)):
            if i == 0 and np.all(v == 1):
                degree0_is_1 = True
            else:
                result += '    const {} {}{} = {};\n'.format(
                    rtype, prefix, i, vecstr(v))
                
        result += '\n'
        
        for i in range(fit_opts.degree + 1):
            if i == 0 and degree0_is_1:
                rval += '1.0'
            else:
                rval += '{}{}'.format(prefix, i)
            if i < fit_opts.degree:
                rval += '+t*'
            if i < fit_opts.degree-1:
                rval += '('
                
        rval += ')'*(fit_opts.degree-1)
        
    else:
        
        assert len(p) == 2*fit_opts.degree + 1

        n = 2*fit_opts.degree + 1

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

        for i in range(1, fit_opts.degree+1):
            
            s = p[n-2*i]
            c = p[n-2*i-1]
            
            result += '    {} += {}*cs{}.x;\n'.format(prefix, vecstr(c), i)
            result += '    {} += {}*cs{}.y;\n'.format(prefix, vecstr(s), i)

        result += '\n'

    if fit_opts.is_rational:
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

    if fit_opts.fit_type == 'fourier':
        function += '    t *= 6.283185307179586;\n\n'
        for i in range(1, fit_opts.degree+1):
            if i == 1:
                timest = 't'
            else:
                timest = '{}.0*t'.format(i)
            function += '    vec2 cs{} = vec2(cos({}), sin({}));\n'.format(
                i, timest, timest)
        function += '\n'

    if fit_opts.is_rational:
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

def fit(key, data, fit_opts, output_opts):

    npoints, nchannels = data.shape
    
    assert nchannels in range(1, 5)
    
    x = np.linspace(0, 1, npoints, endpoint=False)

    _, nparams = num_coeffs(fit_opts)

    print('Fitting {}-parameter model to {} points in {}...'.format(
        nparams, npoints, key), file=output_opts.console_file)

    basis = get_basis(x, fit_opts)

    coeffs = []

    for cidx in range(nchannels):
        channel = data[:, cidx]
        p = fit_single_channel(cidx, x, basis, channel, fit_opts, output_opts)
        coeffs.append(p)

    print(file=output_opts.console_file)

    return np.array(coeffs)

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
        y0 = np.floor(data.min())
        y1 = np.ceil(data.max())
    else:
        y0, y1 = output_opts.range

    x_fine = x
    
    if npoints < output_opts.min_samples:
        factor = int(np.ceil(output_opts.min_samples / npoints))
        x_fine = np.linspace(x0, x1, npoints*factor+1, endpoint=True)

    basis = get_basis(x, fit_opts)
    basis_fine = get_basis(x_fine, fit_opts)
        
    ym = 0.05 * (y1 - y0)

    max_err = 0

    reconstructions = []

    regular_data = (nchannels != 2 or not output_opts.plot_shape)

    for cidx, p in enumerate(coeffs):

        color = chan_colors[cidx] * 0.75
        
        channel = data[:, cidx]

        px = reconstruct(p, basis, fit_opts)
        max_err = max(max_err, np.abs(px - channel).max())

        if x_fine is not x:
            px_fine = reconstruct(p, basis_fine, fit_opts)
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
        plt.axis('equal')
        plt.axis('off')
        tx, ty = y0-xshift, y0
        ha = 'left'
        va = 'bottom'
        effects=None

    plt.text(tx, ty, '{}: max err={:.3g}'.format(key, max_err),
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

    parser.add_argument('-d', dest='degree',
                        metavar='N',
                        type=int, default=DEFAULT_FIT_OPTIONS.degree,
                        help='degree of polynomial/fourier series')
    
    parser.add_argument('-c', dest='clip_reconstruction',
                        metavar='Y0,Y1',
                        type=domain_range,
                        default=DEFAULT_FIT_OPTIONS.clip_reconstruction,
                        help='data clipping limits (default: none)')

    assert DEFAULT_FIT_OPTIONS.is_rational
    
    parser.add_argument('-R', dest='is_rational',
                        action='store_false',
                        help='don\'t use rational approximation')
        
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
        rlabel = 'rational ' if fit_opts.is_rational else ''
        if fit_opts.fit_type == 'poly':
            opts.plot_title = 'Degree {} {}polynomial'.format(
                fit_opts.degree, rlabel)
        else:
            opts.plot_title = 'Order {} {}Fourier series'.format(
                fit_opts.degree, rlabel)
        if fit_opts.clip_reconstruction is not None:
            opts.plot_title += ' clipped to [{:g},{:g}]'.format(
                *fit_opts.clip_reconstruction)

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
