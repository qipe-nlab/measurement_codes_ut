import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# from scipy.stats import norm
# import scipy.signal as sig

def predict(f,s):
    eld  = eld_fit(f,s)
    s   = np.exp(-1.0j*(2.*np.pi*eld*f))*s
    xc,yc,rc = circle_fit(s.real,s.imag)
    s  -= xc + 1.0j*yc
    alg  = np.unwrap(np.angle(s))
    grad = abs(np.gradient(alg))
    f0   = f[grad.argmax()]
    th0  = alg[grad.argmax()]
    rfreq = f[grad.argmax():][abs(grad[grad.argmax():] - 0.5*(grad.max() + 0.5*(grad[0] + grad[-1]))).argmin()]
    lfreq = f[:grad.argmax()][abs(grad[:grad.argmax()] - 0.5*(grad.max() + 0.5*(grad[0] + grad[-1]))).argmin()]
    fwhm  = abs(rfreq - lfreq)
    a0   = abs(xc + 1.0j*yc + rc*np.exp(1.0j*(th0+np.pi)))
    po   = np.angle(xc + 1.0j*yc + rc*np.exp(1.0j*(th0+np.pi))) + np.pi
    kex  = fwhm*rc/a0
    kin  = fwhm-kex
    return a0,-eld,po,kex,kin,f0

def eld_fit(f,s):
    algs = np.unwrap(np.angle(s))
    eld_under = (algs[-1]-algs[0])/(2*np.pi*(f[-1]-f[0]))
    eld_over  = (abs(algs[-1]-algs[0]) - 2*np.pi)*(algs[-1]-algs[0])/abs(algs[-1]-algs[0])/(2*np.pi*(f[-1]-f[0]))
    def cost_eld(eld):
        s_       = np.exp(-1.0j*(2.*np.pi*eld*f))*s
        xc,yc,rc = circle_fit(s_.real,s_.imag)
        return abs((rc**2 - ((s_.real-xc)**2 + (s_.imag-yc)**2)).sum())
    cost_under = cost_eld(eld_under)
    cost_over  = cost_eld(eld_over)
    if cost_under > cost_over:
        eld = eld_over
    else:
        eld = eld_under
    return eld

# def gaussian_filter(data,scale=0.3):
#     return norm.pdf(x=np.linspace(-1,1,data.size), loc=0, scale=scale)*data

# def bessel_filter(data,Ws=0.1):
#     a,b     = sig.bessel(4,Ws,"low")
#     data    = sig.filtfilt(a,b,data)
#     return data

def circle_fit(x,y):
    z = x**2 + y**2
    mn  = x.size
    mx  = x.sum()
    my  = y.sum()
    mz  = z.sum()
    mxx = (x*x).sum()
    myy = (y*y).sum()
    mzz = (z*z).sum()
    mxy = (x*y).sum()
    myz = (y*z).sum()
    mzx = (z*x).sum()
    M = np.array([
        [mzz, mzx, myz, mz],
        [mzx, mxx, mxy, mx],
        [myz, mxy, myy, my],
        [ mz,  mx,  my, mn]
    ])
    B = np.array([
        [ 0, 0, 0, -2],
        [ 0, 1, 0,  0],
        [ 0, 0, 1,  0],
        [-2, 0, 0,  0]
    ])
    def cost(eta):
        return np.linalg.det(M-eta*B)**2
    res = opt.minimize(cost,x0=0,method="BFGS")
    eig = np.linalg.eig(M-res.x*B)
    A   = eig[1].T[abs(eig[0]).argmin()]
    A  /= (A@B@A)**0.5
    xc = - 0.5*A[1]/A[0]
    yc = - 0.5*A[2]/A[0]
    rc = 0.5/abs(A[0])
    return xc,yc,rc