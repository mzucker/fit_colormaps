# fit_colormaps
Minimax polynomial fitting of colormap data

Fits smooth functions to reconstruct tabulated data of up to four channels (i.e. palettes) 
and converts the results to GLSL for use in shaders.

The fits are either polynomials or Fourier series (a.k.a. trigonometric polynomials). 

Default is to do a rational fit (i.e. ratio of polynomials or Fourier series), but you can
do a normal polynomial/Fourier series, optionally.

I may have invented (re-discovered?) the idea of rational Fourier series. The only source 
I've found so far is <https://apps.dtic.mil/dtic/tr/fulltext/u2/a273660.pdf>.

Also a source on fitting rational functions from NIST: <https://www.itl.nist.gov/div898/handbook/pmd/section8/pmd812.htm>

Here's how fitting works:

  1. Global least-squares fit to non-rational polynomial or Fourier series. 
     For rational polynomials/Fourier series, initialize the denominator
     to be the constant function f(x) = 1.
     
  2. Local search to do least-squares fit for rational polynomial or Fourier
     series, using output of step 1 as initial guess. This is delegated to
     `scipy.optimize.curve_fit`, so probably implemented as Levenberg-Marquardt
     under the hood?
     
  3. Local search to do minimax optimization, using the output of step 2 as 
     an initial guess. In my experience, gradient-based optimizers don't do
     amazingly well minimizing maximum error (infinity norm), so using a 
     derivative-free optimizer like Nelder-Mead is safe.
     
For discussion and details, see <https://twitter.com/matt_zucker/status/1250244924181827591>.

See example output on Shadertoy at <https://www.shadertoy.com/results?query=tag%3Dfitcolormap>

## Setup

Windows:

    python -m venv --system-site-packages --prompt fit_colormaps venv
    venv\scripts\activate.bat
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install numpy scipy matplotlib quadprog

Or on Mac/Linux:

    python -m venv --system-site-packages --prompt fit_colormaps venv
    source venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install numpy scipy matplotlib quadprog
