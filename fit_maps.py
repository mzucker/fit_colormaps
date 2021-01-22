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

######################################################################

FitOptions = namedtuple('FitOptions',
                        'fit_type, numer_degree, denom_degree, clip_output, loss')

FitOptions.__doc__ = """

Options to control the fit. Members:

  fit_type:     Either 'poly' or 'fourier'
  numer_degree: Degree of numerator.
  denom_degree: Degree of denominator.
  clip_output:  Either None or a (lo, hi) tuple to clip output.
  loss:         Either 'minimax' or 'rmse'

"""

######################################################################

OutputOptions = namedtuple('OutputOptions',
                           'console_file, glsl_file, '
                           'plot_title, image_filename, '
                           'domain, range, plot_shape, min_samples')

OutputOptions.__doc__ = """

Options to control program output. Members:

  console_file:   File where text output should go
  glsl_file:      File for GLSL program output
  plot_title:     Title for plot (None for auto-generated)
  image_filename: Filename to save output plot to
  domain:         X-axis limits for non-shape plots
  range:          Y-axis limits for non-shape plots
  plot_shape:     Boolean flag whether to plot 2D data as a shape
  min_samples:    Minimum # of samples for smooth plots

"""

######################################################################
# default options

_DEFAULT_FIT_OPTIONS = FitOptions(fit_type='poly',
                                  numer_degree=4,
                                  denom_degree=0,
                                  clip_output=None,
                                  loss='minimax')

_DEFAULT_OUTPUT_OPTIONS = OutputOptions(console_file=None,
                                        glsl_file=None,
                                        plot_title=None,
                                        image_filename=None,
                                        domain=(0., 1.),
                                        range=None,
                                        plot_shape=False,
                                        min_samples=256)

# labels for output
_LOSS_NAMES = dict(rmse='RMSE', minimax='max error')

######################################################################

def coeffs_per_degree(fit_type, degree):

    """Compute the number of coefficients for a given degree
    for either polynomial or fourier basis."""

    assert fit_type in ['poly', 'fourier']
    assert degree >= 0
    
    if fit_type == 'poly':
        return degree + 1
    else:
        return 2 * degree + 1

######################################################################

def count_coeffs(fit_opts):

    """Count coefficients in numerator and total coefficients
    for a given set of fit options."""

    numer_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.numer_degree)
    denom_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.denom_degree) - 1

    return numer_coeffs, numer_coeffs + denom_coeffs
    
######################################################################

def get_fourier_matrix(x, degree):

    """Return a matrix where row i is

      [ cos(2*pi*n*x_i),     sin(2*pi*n*x_i),
        cos(2*pi*(n-1)*x_i), sin(2*pi*(n-1)*x_i)),
        ...
        cos(2*pi*2*x_i),     sin(2*pi*2*x_i),
        cos(2*pi*x_i),       sin(2*pi*x_i),
        1 ]

    For degree n.
    """

    n = 2*degree + 1

    A = np.zeros( (len(x), n) )
    A[:, n-1] = 1

    for i in range(1, degree+1):
        theta = x*2*np.pi*i
        A[:, n-2*i-1] = np.sin(theta)
        A[:, n-2*i] = np.cos(theta)

    return A

######################################################################

def get_polynomial_matrix(x, degree):

    """Return a matrix where row i is

      [ x**n, x**(n-1), ..., x**2, x, 1 ]

    For degree n.
    """

    n = np.arange(degree, -1, -1)

    A = x.reshape(-1, 1) ** n.reshape(1, -1)

    return A
    
######################################################################

def get_data_matrices(x, fit_opts, full=False):

    """Get the data matrices needed for least-squares fitting, evaluated
    at the 1D points x.

    Each data matrix is an m-by-k matrix D(x) such that evaluating the
    function f(x) at each data point can be achieved by multiplying
    the matrix D(x) by a k-by-1 vector p of coefficients:

      f(x) = D(x) p

    Since least-squares polynomial fitting is built into numpy, when
    fit_opts.fit_type == 'poly' and full is False, the data matrix is
    replaced by the x array which can be passed along to np.polyfit().

    This function returns two matrices, one for the numerator and
    one for the denominator. The denominator matrix will be trivial
    if fit_opts.denom_degree == 1.

    """

    if fit_opts.fit_type == 'poly' and not full:
        return (x, x)

    max_degree = max(fit_opts.numer_degree, fit_opts.denom_degree)

    if fit_opts.fit_type == 'fourier':

        data_matrix = get_fourier_matrix(x, max_degree)
        
    else:

        data_matrix = get_polynomial_matrix(x, max_degree)

    num_numer_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.numer_degree)
    num_denom_coeffs = coeffs_per_degree(fit_opts.fit_type, fit_opts.denom_degree)

    numer_matrix = data_matrix[:, -num_numer_coeffs:]
    denom_matrix = data_matrix[:, -num_denom_coeffs:]

    return (numer_matrix, denom_matrix)

######################################################################

def evaluate_single(coeffs, data_matrix, fit_type):

    """ Evaluate a single (numerator or denominator) function f(x)
    either using numpy.polyval or using matrix multiplication of
    the data matrix:

       f(x) = D(x) * p

    Where the data matrix D(x) is m-by-k and the coefficient
    matrix p is k-by-1.

    Returns the vector f(x) of size m-by-1.
    """

    assert fit_type in ['poly', 'fourier']

    if fit_type == 'poly':
        return np.polyval(coeffs, data_matrix)
    else:
        return np.dot(data_matrix, coeffs)

######################################################################

def split_params(coeffs, fit_opts):

    """Given a combined vector of coefficients for the numerator and the
    denominator, split them into two individual vectors of
    coefficients and append the constant 1 coefficient into the
    denominator coefficient vector.
    """
    
    num_numer_coeffs, total_coeffs = count_coeffs(fit_opts)
    assert len(coeffs) == total_coeffs

    pnum = coeffs[:num_numer_coeffs]
    pdenom = np.concatenate((coeffs[num_numer_coeffs:], [ np.ones_like(coeffs[0]) ] ), axis=0)

    return pnum, pdenom

######################################################################

def evaluate(coeffs, data_matrices, fit_opts):

    """Evaluate a rational function by independently
    evaluating the numerator and the denominator separately,
    dividing them, and clipping output if necessary.
    """

    numer_data_matrix, denom_data_matrix = data_matrices

    if fit_opts.denom_degree:
        coeffs_num, coeffs_denom = split_params(coeffs, fit_opts)
        rnum = evaluate_single(coeffs_num, numer_data_matrix, fit_opts.fit_type)
        rdenom = evaluate_single(coeffs_denom, denom_data_matrix, fit_opts.fit_type)
        y = rnum / rdenom
    else:
        y = evaluate_single(coeffs, numer_data_matrix, fit_opts.fit_type)

    if fit_opts.clip_output is not None:
        return np.clip(y, *fit_opts.clip_output)
    else:
        return y

######################################################################

def residual(coeffs, data_matrices, y, fit_opts):

    """Compute the residual vector f(x) - y for least squares or minimax
    fitting."""

    return evaluate(coeffs, data_matrices, fit_opts) - y

######################################################################

def loss(coeffs, data_matrices, y, fit_opts):

    """Computes the RMSE error or maximum error of the residual."""

    e = residual(coeffs, data_matrices, y, fit_opts)
    
    if fit_opts.loss == 'rmse':
        return np.sqrt(np.dot(e, e)/len(e))
    else:
        return np.abs(e).max()
    
######################################################################    
    
def fit_single_channel(cidx, x, y, fit_opts, output_opts):

    """Fit a 1D function f(x) to given data y. Note
    that cidx, the channel index, is just used for printed output.
    """

    loss_name = _LOSS_NAMES[fit_opts.loss]

    data_matrices = get_data_matrices(x, fit_opts)

    args = (data_matrices, y, fit_opts)

    numer_data_matrix, denom_data_matrix = data_matrices

    ############################################################
    # step 1/3: global initial fit

    if fit_opts.fit_type == 'poly':
        init_coeffs = np.polyfit(numer_data_matrix, y, fit_opts.numer_degree)
    else:
        init_coeffs, _, _, _ = np.linalg.lstsq(numer_data_matrix, y, rcond=None)

    ############################################################
    # step 2/3: fit numerator and denominator coefficients
    # using quadratic program

    if not fit_opts.denom_degree:
        
        # can skip this step if not rational
        all_init_coeffs = [init_coeffs]
        all_eps = [None]
        
    else:

        num_numer_coeffs, total_coeffs = count_coeffs(fit_opts)
        num_denom_coeffs = total_coeffs - num_numer_coeffs
        
        init_coeffs = np.hstack([init_coeffs, np.zeros(num_denom_coeffs, dtype=init_coeffs.dtype)])
        all_init_coeffs = [init_coeffs]
        all_eps = [None]

        if fit_opts.fit_type == 'poly':
            assert len(numer_data_matrix.shape) == 1
            assert np.all(numer_data_matrix == denom_data_matrix)
            numer_full_matrix, denom_full_matrix = get_data_matrices(numer_data_matrix, fit_opts, full=True)
        else:
            numer_full_matrix, denom_full_matrix = numer_data_matrix, denom_data_matrix

        assert len(numer_full_matrix.shape) == 2
        assert len(denom_full_matrix.shape) == 2
        assert len(y.shape) == 1

        npoints = numer_full_matrix.shape[0]

        '''
        
        Notation:

        Let the numerator data matrix be N(x), and denote its i'th row as N_i^T    (1-by-k)
        Let the denominator data matrix be D(x), and denote its i'th row as D_i^T  (1-by-j)

        The overall coefficients are given by the numerator coefficients p (k-by-1) and the denominator coefficients q (j-by-1).
        
        The i'th function value for the rational function f(x) is given by:

                       N_i^T p
            f(x_i) = -----------
                     D_i^T q + 1

        Note we want f(x_i) = y_i for all i.

        Cross-multipliying, we get 

            N_i^T p = (D_i^T q + 1) * y_i
            N_i^T p = y_i * D_i^T q + y_i
        
        Rearranging to put the unknown coefficients p and q on the left-hand side we get

            N_i^T p - y_i * D_i^T q = y_i

        Let the overall coefficient vector w be the concatenation of p and q. 
        Also let the row vector A_i^T be the concatenation of N_i^T and -y_i * D_i^T.

        We now have the transformed system

            A_i^T w = y_i

        Furthermore suppose that we want to enforce a constraint that the denominator
        is larger than a constant, that is

           D_i^T q + 1 >= eps

        for all i. This gives us the condition

           D_i^T q >= eps - 1

        This forms the basis of our quadratic program.
        
        '''

        assert npoints == denom_full_matrix.shape[0]
        assert numer_full_matrix.shape[1] == num_numer_coeffs
        assert denom_full_matrix.shape[1] == num_denom_coeffs + 1

        A = np.hstack( (numer_full_matrix, -denom_full_matrix[:, :-1] * y.reshape(-1, 1)) )
        assert A.shape == (npoints, num_numer_coeffs + num_denom_coeffs)

        assert np.all(numer_full_matrix[:, -1] == 1)
        assert np.all(denom_full_matrix[:, -1] == 1)
        b = y

        C = np.zeros_like(A)
        C[:, num_numer_coeffs:] = denom_full_matrix[:, :-1]

        CT = C.transpose()

        # NOTE: had some issues with singularities, so was messing around with this
        # not sure if needed? -MZ 1/13/21
        lambda_I = 1e-8 * np.eye(total_coeffs)

        G = np.dot(A.T, A) + lambda_I
        a = np.dot(A.T, b)

        step = 0.025
        epsrng = np.arange(step, 1, step)
        #epsrng = 1 - np.exp(-np.arange(1, 10))

        for eps in epsrng:

            lb = np.ones_like(C[:,0])*eps - 1

            init_coeffs, _, _, _, _, _ = quadprog.solve_qp(G, a, CT, lb)

            denom = np.dot(C, init_coeffs) + 1
            #print('denominator:', denom.min(), denom.max())
            
            all_eps.append(eps)
            all_init_coeffs.append(init_coeffs)

    all_init_coeffs_loss = np.array([loss(init_coeffs, *args) for init_coeffs in all_init_coeffs])

    best_idx = all_init_coeffs_loss.argmin()

    best_init_coeffs = all_init_coeffs[best_idx]
    best_init_loss = all_init_coeffs_loss[best_idx]

    #all_init_coeffs = all_init_coeffs[best_idx:best_idx+1]

    best_final_coeffs = None
    best_final_loss = None

    for eps, init_loss, init_coeffs in zip(all_eps, all_init_coeffs_loss, all_init_coeffs):

        final_coeffs = init_coeffs
        final_loss = init_loss

        if (fit_opts.loss == 'minimax' or fit_opts.denom_degree) :

            final_coeffs = init_coeffs.copy()

            if fit_opts.denom_degree:
                # step 2/3: local search to do least-squares fit for rational
                # polynomial or Fourier series, using output of step 1 as
                # initial guess.
                res = scipy.optimize.least_squares(residual, final_coeffs,
                                                   ftol=1e-5, xtol=1e-5,
                                                   method='dogbox', args=args)
                final_coeffs = res.x

            if fit_opts.loss == 'minimax':
                # step 3/3: local search to do minimax optimization, starting from
                # output of step 1 or 2. 
                res = scipy.optimize.minimize(loss, final_coeffs,
                                              method='Nelder-Mead',
                                              args=args)
                final_coeffs = res.x

            final_loss = loss(final_coeffs, *args)

        updated = (best_final_loss is None or final_loss < best_final_loss)

        if updated:
            best_final_loss = final_loss
            best_final_coeffs = final_coeffs
            updated = True

        if eps is None:
            epstr = 'None '
        else:
            epstr = '{:5.3f}'.format(eps)
            
        print('  - channel {} {} (eps={:}): {:.7f} -> {:.7f} ({:6.2f}%){}'.format(
            cidx, loss_name, epstr, init_loss, final_loss, 100.0*final_loss/init_loss, ' *** new min ***' if updated else ''),
              file=output_opts.console_file)

    print(file=output_opts.console_file)
    
    return best_final_coeffs

######################################################################

def fit(name, data, fit_opts, output_opts):

    """Fit all channels sequentially."""

    npoints, nchannels = data.shape
    
    assert nchannels in range(1, 5)
    
    x = np.linspace(0, 1, npoints, endpoint=False)

    _, total_coeffs = count_coeffs(fit_opts)

    print('Fitting {}-parameter model to {} points in {}...\n'.format(
        total_coeffs, npoints, name), file=output_opts.console_file)

    all_coeffs = []

    for cidx in range(nchannels):
        channel = data[:, cidx]
        coeffs = fit_single_channel(cidx, x,
                               channel, fit_opts, output_opts)
        all_coeffs.append(coeffs)

    return np.array(all_coeffs)

######################################################################

def get_glsl_type(nchannels):

    """GLSL output type is either float for 1D data or vecN for N-dimensional arrays."""

    assert nchannels <= 4

    if nchannels == 1:
        return 'float'
    else:
        return 'vec' + str(nchannels)

######################################################################

def glsl_constant_string(v):

    """Convert scalar or vector to GLSL constant."""

    fmt = '{:.16g}'
    if len(v) == 1:
        return fmt.format(v[0])
    else:
        interior = ', '.join(fmt.format(vi) for vi in v)
        return get_glsl_type(len(v)) + '(' + interior + ')'
    
######################################################################

def glslify_single(coeffs, fit_opts, prefix):

    """Get GLSL output for numerator or denominator."""
    
    assert fit_opts.fit_type in ['poly', 'fourier']

    nchannels = coeffs.shape[1]
    rtype = get_glsl_type(nchannels)

    result = ''
    rval = ''

    if fit_opts.fit_type == 'poly':

        degree = len(coeffs) - 1
        
        degree0_is_1 = False
        
        for i, v in enumerate(reversed(coeffs)):
            if i == 0 and np.all(v == 1):
                degree0_is_1 = True
            else:
                result += '    const {} {}{} = {};\n'.format(
                    rtype, prefix, i, glsl_constant_string(v))
                
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

        n = len(coeffs)
        degree = (n - 1) // 2
        assert n == 2*degree + 1

        if np.all(coeffs[-1] == 1):
            rval += '1.0'
        else:
            rval += glsl_constant_string(coeffs[0])

        rval += prefix

        if np.all(coeffs[-1] == 1):
            if rtype == 'float':
                c0 = '1.0'
            else:
                c0 = '{}(1.0)'.format(rtype)
        else:
            c0 = glsl_constant_string(coeffs[-1])

        result += '    {} {} = {};\n'.format(rtype, prefix, c0)

        for i in range(1, degree+1):
            
            s = coeffs[n-2*i-1]
            c = coeffs[n-2*i]
            
            result += '    {} += {}*cs{}.x;\n'.format(prefix, glsl_constant_string(c), i)
            result += '    {} += {}*cs{}.y;\n'.format(prefix, glsl_constant_string(s), i)

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

def glslify(results, fit_opts, output_opts):

    """Get GLSL output for all datasets."""

    if output_opts.glsl_file is None:
        return
    
    for name, _, coeffs in results:

        nchannels = coeffs.shape[0]

        function = '{} {}(float t) {{\n\n'.format(get_glsl_type(nchannels), name)

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
            coeffs_num, coeffs_denom = split_params(coeffs.T, fit_opts)
            function += glslify_single(coeffs_num, fit_opts, 'n')
            function += glslify_single(coeffs_denom, fit_opts, 'd')
            if fit_opts.fit_type == 'fourier':
                function += '    return n/d;\n\n'
            else:
                function += '    return num/denom;\n\n'
        else:
            function += glslify_single(coeffs.T, fit_opts, 'coeffs')

        function += '}\n'

        print(function, file=output_opts.glsl_file)
    

######################################################################

def plot_single(dataset_name, data, coeffs, fit_opts, output_opts):

    """Plot a single dataset."""

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

    data_matrices = get_data_matrices(x, fit_opts)
    data_matrices_fine = get_data_matrices(x_fine, fit_opts)
        
    ym = 0.05 * (y1 - y0)

    max_err = 0

    reconstructions = []
    losses = []

    regular_data = (nchannels != 2 or not output_opts.plot_shape)

    for cidx, coeffs in enumerate(coeffs):

        color = chan_colors[cidx] * 0.75
        
        channel = data[:, cidx]

        px = evaluate(coeffs, data_matrices, fit_opts)
        losses.append(loss(coeffs, data_matrices, channel, fit_opts))

        if x_fine is not x:
            px_fine = evaluate(coeffs, data_matrices_fine, fit_opts)
        else:
            px_fine = px

        reconstructions.append(px_fine)


        if regular_data:
            plt.plot(x, channel, color=color)
            plt.plot(x_fine, px_fine, ':', color=0.5*color, linewidth=2)

            '''
            y = evaluate(coeffs, data_matrices, fit_opts)
            err = np.abs(y - channel)

            imax = err.argmax()
            ex = x[imax]
            ey = 0.5*(y[imax] + channel[imax])
            
            plt.plot(ex, ey, 'ro', markerfacecolor='none')
            plt.text(ex, ey, 'max err={:.3g}'.format(err[imax]), ha='center', va='top')
            '''
        
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

    loss_name = _LOSS_NAMES[fit_opts.loss]
    losses = np.array(losses)
    #with np.printoptions(formatter={'all':lambda x: '{:.3g}'.format(x)}) as opts:
    lstr = np.array2string(losses, formatter=dict(all=lambda x:'{:.3g}'.format(x)), separator=', ')
    plt.text(tx, ty, '{}: per-channel {}={}, total={:.3g}'.format(
        dataset_name, loss_name, lstr, losses.sum()),
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

    for i, (dataset_name, mapdata, coeffs) in enumerate(results):
        plt.subplot(rows, cols, i+1)
        plot_single(dataset_name, mapdata, coeffs, fit_opts, output_opts)

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
                        default=_DEFAULT_FIT_OPTIONS.fit_type,
                        help='type of fit')

    parser.add_argument('-n', dest='numer_degree',
                        metavar='N',
                        type=int, default=_DEFAULT_FIT_OPTIONS.numer_degree,
                        help='degree of numerator')

    parser.add_argument('-d', dest='denom_degree',
                        metavar='N',
                        type=int, default=_DEFAULT_FIT_OPTIONS.denom_degree,
                        help='degree of denominator')
    
    parser.add_argument('-c', dest='clip_output',
                        metavar='Y0,Y1',
                        type=domain_range,
                        default=_DEFAULT_FIT_OPTIONS.clip_output,
                        help='output data clipping limits (default: none)')

    parser.add_argument('-l', dest='loss',
                        choices=('minimax', 'rmse'),
                        default=_DEFAULT_FIT_OPTIONS.loss,
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
                        default=_DEFAULT_OUTPUT_OPTIONS.plot_title,
                        help='title for plots')

    parser.add_argument('-p', dest='image_filename',
                        default=_DEFAULT_OUTPUT_OPTIONS.image_filename,
                        help='image filename or - to suppress plotting')
    
    parser.add_argument('-x', dest='domain',
                        metavar='X0,X1', default=_DEFAULT_OUTPUT_OPTIONS.domain,
                        type=domain_range, help='domain for plotting')

    parser.add_argument('-y', dest='range',
                        metavar='Y0,Y1', default=_DEFAULT_OUTPUT_OPTIONS.range,
                        type=domain_range, help='range for plotting')
    
    parser.add_argument('-m', dest='min_samples',
                        metavar='N', default=_DEFAULT_OUTPUT_OPTIONS.min_samples,
                        type=int, help='min number of points in domain for plotting')

    assert not _DEFAULT_OUTPUT_OPTIONS.plot_shape

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
        if fit_opts.clip_output is not None:
            opts.plot_title += ' clipped to [{:g},{:g}]'.format(
                *fit_opts.clip_output)
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

    """Our main function."""

    mapfiles, fit_opts, output_opts = parse_cmdline()

    print('Fit options:\n', file=output_opts.console_file)
    for dataset_name, val in fit_opts._asdict().items():
        print('  {:20} {}'.format(dataset_name+':', val),
              file=output_opts.console_file)
    print(file=output_opts.console_file)

    results = []
    
    for i, filename in enumerate(mapfiles):
        datafilename = 'data/' + filename + '.txt'
        if not os.path.exists(filename) and os.path.exists(datafilename):
            filename = datafilename
        dataset_name, _ = os.path.splitext(os.path.basename(filename))
        mapdata = np.genfromtxt(filename)
        if len(mapdata.shape) == 1:
            mapdata = mapdata.reshape(-1, 1)
        coeffs = fit(dataset_name, mapdata, fit_opts, output_opts)
        results.append((dataset_name, mapdata, coeffs))

    glslify(results, fit_opts, output_opts)

    plot(results, fit_opts, output_opts)
    

######################################################################
    
if __name__ == '__main__':
    main()
