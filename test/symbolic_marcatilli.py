import scipy as sp
import scipy.constants
import scipy.optimize
import sympy
pi = sp.pi
c = sp.constants.c
w, l = sympy.symbols("w l")

def getbetaexp(order, a, unm, B1, B2, C1, C2, scale):
    beta = (w/c*(1 + scale*(B1 * l**2 / (l**2 - C1) + B2 * l**2 / (l**2 - C2))/2 - c**2*unm**2/(2*w**2*a**2))).subs(l, 2e6*pi*c/w)
    return sympy.diff(beta, w, order)

def dispersion(ω, order, a, unm, B1, B2, C1, C2, scale):
    return getbetaexp(order, a, unm, B1, B2, C1, C2, scale).subs(w, ω).evalf()

def zdf(a, unm, B1, B2, C1, C2, scale):
    f = sympy.lambdify(w, getbetaexp(2, a, unm, B1, B2, C1, C2, scale))
    return sp.optimize.brentq(f, 1e14, 1e16)

if __name__ == "__main__":
    ω = 2*pi*c/800e-9
    unm = 2.4048255576957724
    scale = 4.9992223031315305e25 # 2 bar Ar at 294 K
    B1 = 20332.29e-8/2.6907883518864814e25 # scaled to 1 bar at 273 K
    C1 = 206.12e-6
    B2 = 34458.31e-8/2.6907883518864814e25
    C2 = 8.066e-3
    a = 50e-6
    print("0th: ", dispersion(ω, 0, a, unm, B1, B2, C1, C2, scale))
    print("1st: ", dispersion(ω, 1, a, unm, B1, B2, C1, C2, scale))
    print("2nd: ", dispersion(ω, 2, a, unm, B1, B2, C1, C2, scale))
    print("3rd: ", dispersion(ω, 3, a, unm, B1, B2, C1, C2, scale))
    print("4th: ", dispersion(ω, 4, a, unm, B1, B2, C1, C2, scale))
    print("5th: ", dispersion(ω, 5, a, unm, B1, B2, C1, C2, scale))
    print("6th: ", dispersion(ω, 6, a, unm, B1, B2, C1, C2, scale))
    print("7th: ", dispersion(ω, 7, a, unm, B1, B2, C1, C2, scale))
    print("zdw: ", 2*pi*c/zdf(a, unm, B1, B2, C1, C2, scale))
