import numpy as np
#from scipy.special import jv,jvp
#import scipy as scipy
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv
#from scipy.integrate import quad
#from numpy.polynomial.chebyshev import Chebyshev as cheb
#from scipy.linalg import expm
#import numpy.polynomial.chebyshev
import math
import scipy as scipy
from scipy.integrate import quad
from scipy.signal import hilbert as scipy_hilbert
from scipy.interpolate import CubicSpline




#n'th order derivative, FD.
def FD(f,x,n=1,dx=1e-3,args=(),order=3): #f a callable function
    if order <=n:
        order = n + 1 #at least n+1 points must be used to calculate n'th derivative
        order = order + (1-order %2) #an odd-number of points must be used to calculate the derivative
    fd = scipy.misc.derivative(f,x,dx,n,args=args,order=order)
    return fd

def fft(array,axis=-1):
    #return np.fft.fft(array,axis)
    return np.fft.fft(array,axis=axis)

def ifft(array,axis=-1):
    #return np.fft.ifft(array,axis)
    return np.fft.ifft(array,axis=axis)

def fftfreq(n,d):
 #return np.fft.fftfreq(n,d)
    return np.fft.fftfreq(n,d)


class Spline: 
    """
    A wrapper for the scipy.interpolate.CubicSpline class that is changed to specifically suit the needs in AE.
    Creates a cubic spline f(x) from the sample points (X,Y).
    It implements new BC types, namely 
    - 'dirichlet' which sets the function to zero outside the range covered by the sample points ([X.min(), X.max()]), i.e. f(x > X.max() ) = 0 and f(x < X.min()) = 0.
    - 'constant', which is used for antiderivatives of functions with dirichlet BCs. Here f(x > X.max()) = f(X.max()) and f(x < X.min()) = f(X.min()).
    One can additionally specify whether the function is a matrix, which helps determine the shape of the output.
    """
    def __init__(self,x, y, axis=0, bc_type='dirichlet', extrapolate=False,MatrixFunction = False): 
        #bc-type may be 'dirichlet', 'constant', or any normal CubicSpline bc.
        self.bc_type = bc_type
        if bc_type == 'dirichlet' or bc_type =='constant':
            bc_type = 'not-a-knot'
        self.MatrixFunction = MatrixFunction
        cs = CubicSpline(x,y,axis,bc_type,extrapolate)
        self.cs = cs    
        self.X = x
        self.Y = y

    
    def __call__(self,x0):
        x = np.array(x0)
        if self.MatrixFunction and len(x.shape)>2:
            #if the function is a matrix function and the argument already contains the matrix indices, we remove these internally as they will be re-added when the function is called.
            x = x.reshape(x.shape[:-2]) 
        res = self.cs(x)
        if self.bc_type == 'dirichlet':
            res[np.isnan(res)] = 0
        elif self.bc_type == 'constant':
            a,b = self.X.min(), self.X.max()
            res[x < a] = self.cs(a)
            res[x > b] = self.cs(b)
        return res


def exp(x): #exponential function that does not return inf for large input values
    if np.any(np.iscomplex(x)):
        if hasattr(x, '__iter__'):
            x[np.real(x)>709] = 709 + 1j*np.imag(x[np.real(x)>709])
            #x[x<-710]00 = - 710
        else: 
            x = 709 + 1j*np.imag(x) if np.real(x)>709 else x
            #1
            #x = -700 if x<-700 else x
    else:
        if hasattr(x, '__iter__'):
            x[x>709] = 709
            #x[x<-710]00 = - 710
        else: 
            x = 709 if x>709 else x
            #x = -700 if x<-700 else x
    return np.exp(x)


def hilbert(f,axis=-1,padding_length=-1): #computes the hilbert transform of f along axis assuming zero boundary conditions
    shape = list(f.shape)
    f = f.real
    if padding_length == -1:
        padding_length = int(shape[axis]) #pads f with the same number of zeros as its length
        #print(padding_length)
    padding_indices = []
    for i in range(len(shape)):
        padding_indices.append((0,))
    padding_indices[axis]=(padding_length,)

    fpad = np.pad(f,padding_indices)
    #print(fpad.shape)
    hpad = scipy_hilbert(fpad,axis=axis)
    hpad = hpad.swapaxes(0,axis)

    h = hpad[padding_length:-padding_length] #remove padding

    h = h.swapaxes(0,axis)

    return h
    
def get_integration_var(N_int,lower_lim = 0, upper_lim = 1):
    #function to get N_int gauss-legendre quadrature points and weights in the specified interval
    x,w = leggauss(N_int)
    x = (x + 1)/2*(upper_lim-lower_lim) + lower_lim
    w *= (upper_lim - lower_lim)/2
    return x,w

def shift(omega,function,value): #from f(x) calculate f(x - value) assuming zero boundary conditions
    shift_idx = np.argmin(abs(omega - value)) - np.argmin(abs(omega))
    shape = list(function.shape)
    pad_shape = [abs(shift_idx)] + shape[1:]
    zero_padding = np.zeros(pad_shape)
    if shift_idx >= 0:
         new_func = np.concatenate((zero_padding,function[:-shift_idx]))
    else:
         new_func = np.concatenate((function[abs(shift_idx):],zero_padding))
    return new_func





def Pade_poles_and_coeffs(N_F):
    x = Pade_Poles(N_F)
    return x, np.ones(len(x))

def Pade_Poles(N_F):
    Z = np.zeros((N_F, N_F), dtype = np.complex128)
    for i in range(N_F):
        for j in range(N_F): 
            I = i+1
            if j == i+1:
                Z[i,j] = 2 * I * ( 2 * I -1)
            if i == N_F - 1:
                Z[i,j] = -2 * N_F * (2 * N_F -1)
    
    eig, v = np.linalg.eig(Z)
    x = 2 * np.sqrt(eig)
    xr = x.real
    xi = x.imag
    x[xi<0] *= -1
    return x

def FD_expanded(E, xp, beta , mu = 0.0, coeffs = None):
    Xpp = mu  + xp/beta
    Xpm = mu  - xp/beta
    if coeffs is None:
        coeffs = np.ones(len(xp))
    diffs =  (1 / beta) * (1 / np.subtract.outer(E , Xpp)  + 1 / np.subtract.outer(E , Xpm)) * coeffs
    return 1 / 2 - diffs.sum(axis = 1)

def Fermi_dirac(E, beta, mu = 0.0):
    return 1 / (1 + np.exp((E - mu) * beta))

def diff(E, xp, beta, mu = 0.0):
    return Fermi_dirac(E, beta, mu = mu) - FD_expanded(E, xp, beta, mu = mu)

def Hu_b(m):
    return 2 * m -1
def Hu_RN(N):
    return 1/(4 * (N+1) *Hu_b(N+1))

def Hu_Gamma(M):
    Mat = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            I = i+1
            J = j+1
            if  i == j+1 or i == j-1:
                Mat[i,j] = 1/np.sqrt(Hu_b(I) * Hu_b(J))
    return Mat
def Hu_roots_Q(N):
    M = 2 * N
    e,v = np.linalg.eig(Hu_Gamma(M))
    #print(e)
    e = np.sort(e[e>1e-15])[::-1]
    return 2/e
def Hu_roots_P(N):
    M = 2 * N
    e,v = np.linalg.eig(Hu_Gamma(M)[1:, 1:])
    e = e[e>1e-15]
    e = 2/e
    return e

def Hu_coeffs(N):
    Const =  N * Hu_b(N+1)/2
    Qx = Hu_roots_Q(N)
    Px = Hu_roots_P(N)
    coeffs = []
    for i in range(N):
        p1 = Qx**2 - Qx[i]**2
        p1[np.abs(p1)<1e-15] = 1.0
        p1 = np.prod(p1)
        p2 = np.prod(Px ** 2 - Qx[i]**2 )
        coeffs += [Const*p2/p1]
    return np.array(coeffs)

def Hu_poles(N):
    return 1j * Hu_roots_Q(N), Hu_coeffs(N)