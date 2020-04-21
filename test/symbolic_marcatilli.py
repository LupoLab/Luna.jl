import scipy as sp
import scipy.constants
import scipy.optimize
import sympy
pi = sp.pi
c = sp.constants.c
w, l = sympy.symbols("w l", real=True)
from sympy import I

def getbetaexp(order, a, unm, B1, B2, C1, C2, scale):
    beta = (w/c*(1 + scale*(B1 * l**2 / (l**2 - C1) + B2 * l**2 / (l**2 - C2))/2 - c**2*unm**2/(2*w**2*a**2))).subs(l, 2e6*pi*c/w)
    return sympy.diff(beta, w, order)

def dispersion(ω, order, a, unm, B1, B2, C1, C2, scale):
    return getbetaexp(order, a, unm, B1, B2, C1, C2, scale).subs(w, ω).evalf()

def zdf(a, unm, B1, B2, C1, C2, scale):
    f = sympy.lambdify(w, getbetaexp(2, a, unm, B1, B2, C1, C2, scale))
    return sp.optimize.brentq(f, 1e14, 1e16)

def getbetaexpfull(order, a, unm, B1, B2, C1, C2, scale, nclad):
    εcl = nclad**2
    beta = (w/c*(sympy.sqrt(1 + scale*(B1 * l**2 / (l**2 - C1) + B2 * l**2 / (l**2 - C2)) - (unm/(w*a/c))**2*(1 - I*(εcl + 1)/(2*sympy.sqrt(εcl - 1))/(w*a/c))**2))).subs(l, 2e6*pi*c/w).as_real_imag()[0]
    return sympy.diff(beta, w, order)

def dispersionfull(ω, order, a, unm, B1, B2, C1, C2, scale, nclad):
    return getbetaexpfull(order, a, unm, B1, B2, C1, C2, scale, nclad).subs(w, ω).evalf()

def zdffull(a, unm, B1, B2, C1, C2, scale, nclad):
    f = sympy.lambdify(w, getbetaexpfull(2, a, unm, B1, B2, C1, C2, scale, nclad))
    return sp.optimize.brentq(f, 1e14, 1e16)

def alphafull(ω, a, unm, B1, B2, C1, C2, scale, nclad):
    εcl = nclad**2
    alpha = (w/c*(sympy.sqrt(1 + scale*(B1 * l**2 / (l**2 - C1) + B2 * l**2 / (l**2 - C2)) - (unm/(w*a/c))**2*(1 - I*(εcl + 1)/(2*sympy.sqrt(εcl - 1))/(w*a/c))**2))).subs(l, 2e6*pi*c/w).as_real_imag()[1]
    return 2*alpha.subs(w, ω).evalf()
    

if __name__ == "__main__":
    ω = 2*pi*c/800e-9
    unm = 2.4048255576957724
    scale = 4.933761614599279e25 # 2 bar Ar at 294 K
    B1 = 20332.29e-8/2.6541047057884805e25 # scaled to 1 bar at 273.15 K
    C1 = 206.12e-6
    B2 = 34458.31e-8/2.6541047057884805e25
    C2 = 8.066e-3
    a = 50e-6
    print("reduced")
    print("0th: ", dispersion(ω, 0, a, unm, B1, B2, C1, C2, scale))
    print("1st: ", dispersion(ω, 1, a, unm, B1, B2, C1, C2, scale))
    print("2nd: ", dispersion(ω, 2, a, unm, B1, B2, C1, C2, scale))
    print("3rd: ", dispersion(ω, 3, a, unm, B1, B2, C1, C2, scale))
    print("4th: ", dispersion(ω, 4, a, unm, B1, B2, C1, C2, scale))
    print("5th: ", dispersion(ω, 5, a, unm, B1, B2, C1, C2, scale))
    print("6th: ", dispersion(ω, 6, a, unm, B1, B2, C1, C2, scale))
    print("7th: ", dispersion(ω, 7, a, unm, B1, B2, C1, C2, scale))
    print("zdw: ", 2*pi*c/zdf(a, unm, B1, B2, C1, C2, scale))
    print("full; nclad=1.45")
    print("0th: ", dispersionfull(ω, 0, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("1st: ", dispersionfull(ω, 1, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("2nd: ", dispersionfull(ω, 2, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("3rd: ", dispersionfull(ω, 3, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("4th: ", dispersionfull(ω, 4, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("5th: ", dispersionfull(ω, 5, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("zdw: ", 2*pi*c/zdffull(a, unm, B1, B2, C1, C2, scale, 1.45))
    print("alpha: ", alphafull(ω, a, unm, B1, B2, C1, C2, scale, 1.45))
    print("full; nclad=0.036759+1j*5.5698")
    nclad = 0.036759+1j*5.5698
    print("0th: ", dispersionfull(ω, 0, a, unm, B1, B2, C1, C2, scale, nclad))
    print("1st: ", dispersionfull(ω, 1, a, unm, B1, B2, C1, C2, scale, nclad))
    print("2nd: ", dispersionfull(ω, 2, a, unm, B1, B2, C1, C2, scale, nclad))
    print("3rd: ", dispersionfull(ω, 3, a, unm, B1, B2, C1, C2, scale, nclad))
    print("4th: ", dispersionfull(ω, 4, a, unm, B1, B2, C1, C2, scale, nclad))
    print("5th: ", dispersionfull(ω, 5, a, unm, B1, B2, C1, C2, scale, nclad))
    print("zdw: ", 2*pi*c/zdffull(a, unm, B1, B2, C1, C2, scale, nclad))
    print("alpha: ", alphafull(ω, a, unm, B1, B2, C1, C2, scale, nclad))