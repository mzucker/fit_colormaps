from __future__ import print_function

import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import sys

DEGREE = 6
RATIONAL = True

def polyval(p, x, rational):
    n = len(p)
    if rational:
        assert n % 2 == 1
        num_numerator = n//2 + 1
        pn = p[:num_numerator]
        pd = np.hstack((p[num_numerator:], [1]))
        return np.polyval(pn, x) / np.polyval(pd, x)
    else:
        return np.polyval(p, x)

def polyval01(p, x, rational):
    return np.clip(polyval(p, x, rational), 0, 1)

def compare(etype, e0, e1):
    print('{}: {:.5f} -> {:.5f} ({:.2f}%)'.format(
        etype, e0, e1, 100.0*e1/e0), file=sys.stderr)

def fit_single_channel(x, y, deg, rational):

    def f(x, *p):
        #print('len(p) =', len(p))
        return polyval(np.array(p), x, rational)

    def err(p):
        return y - polyval(p, x, rational)
    
    def l1err(p):
        e = err(p)
        return np.abs(e).max()

    def rmse(p):
        e = err(p)
        return np.sqrt(np.dot(e, e)/len(e))

    p0 = np.polyfit(x, y, deg)
    guess = p0

    if rational:
        p0_denom = np.zeros_like(p0[:-1])
        p0 = np.hstack((p0, p0_denom))
        guess = p0
        
    if rational:
        guess, _ = scipy.optimize.curve_fit(f, x, y, guess,
                                            maxfev=10000,
                                            ftol=1e-5, xtol=1e-5)

    res = scipy.optimize.minimize(l1err, guess, method='Nelder-Mead')
    p1 = res.x
    
    compare('Linf', l1err(p0), l1err(p1))
    
    return p1

def vec3str(xyz):
    return 'vec3({:.16g}, {:.16g}, {:.16g})'.format(*xyz)

def fit(key, data):

    rational = RATIONAL

    data = np.array(data)
    
    x = np.linspace(0, 1, len(data))

    print('fitting', key, file=sys.stderr)

    coeffs = []

    plt.text(1, 0, key, ha='right', va='bottom')
    
    for cidx in range(3):

        color = np.array([0., 0., 0.])
        color[cidx] = 0.75
        
        channel = data[:,cidx]

        p = fit_single_channel(x, channel, DEGREE, rational)
        px = polyval01(p, x, rational)
        
        plt.plot(x, channel, color=color)
        plt.plot(x, px, ':', color=0.5*color, linewidth=2)

        plt.ylim(-0.05, 1.05)

        coeffs.append(p)

    print(file=sys.stderr)

    if rational:
        return

    coeffs = np.array(coeffs).transpose()
    assert coeffs.shape == (DEGREE+1, 3)

    funcstrs = [
        'vec3 {}(float t) {{\n'.format(key),
        '\n',
    ]

    for i, c in enumerate(reversed(coeffs)):
        funcstrs.append('    const vec3 c{} = {};\n'.format(i, vec3str(c)))

    funcstrs.append('\n    return ')

    for i in range(DEGREE+1):
        ci = 'c{}'.format(i)
        funcstrs.append(ci)
        if i < DEGREE:
            funcstrs.append('+t*')
        if i < DEGREE-1:
            funcstrs.append('(')

    funcstrs.append(')'*(DEGREE-1));
    funcstrs.append(';\n')
        
    funcstrs.append('\n}\n')

    print(''.join(funcstrs))
    
def main():

    #cmaps = 'inferno magma plasma viridis turbo'.split()
    cmaps = 'hsluv hpluv'.split()

    print('fitting - degree={}, rational={}\n'.format(DEGREE, RATIONAL),
          file=sys.stderr)

    plt.figure(figsize=(8, 4.5))

    n = len(cmaps)

    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n/rows))

    for i, key in enumerate(cmaps):
        mapdata = np.genfromtxt('data/' + key + '.txt')
        plt.subplot(rows, cols, i+1)
        fit(key, mapdata)

    #filename = 'fit_maps_plots.png'
    #plt.savefig(filename, dpi=144)
    #print('wrote', filename)

    plt.show()

######################################################################
    
if __name__ == '__main__':
    main()
