
import numpy as np
from scipy.special import jv,jvp
import scipy as scipy
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv
from scipy.integrate import quad
#from numpy.polynomial.chebyshev import Chebyshev as cheb
from scipy.linalg import expm
#import numpy.polynomial.chebyshev
import math
import time
#import PadeDecomp
from numba import njit, prange, jit
#matrix structure of internal variables: #aux, aux , ... T, omega, [device space matrix]
import os, psutil
#process = psutil.Process(os.getpid())
import AE_math

class Potential:
    """
    Class for handling external potentials.

    The raison d'etre is the fact that in calculating the Wigner transforms of electrode self-energies,
    an integral over the entire external potential must be carried out.
    To ensure that this is done correctly, the program needs to know the support of the potential, and 
    this is a parameter that may be specified in this class. 
    By optionally specifiying additional parameters such as the Range of the potential or 
    analytical expressions for its derivative or antiderivative, computation time may be improved significantly.

    Example use:
        pot = Potential(np.cos, support=[0, 2*np.pi],Range=[-1,1],antiderivative = np.sin)
        t = np.linspace(-np.pi, np.pi,100)
        V = pot(t)

        V will now be an array with the same shape as t. For those values of t that are inside the support [0,2 pi],
        V will contain the function value np.cos(t). For those values of t that are outside the support of the
        potential (i.e. the first 50-ish entries), V will contain the value zero.

    If the antiderivative is not specified in the initialization, the function pot.get_numerical_antiderivative()
    should be called before using the object in calculations. This makes the calculation of Wigner transforms much
    faster. It also ensures that the antiderivative holds information about the support of the potential, which may
    be cumbersome to include in a manually specified closed-form antiderivative.
    

    ----------------- Mandatory parameters ----------------- 

    - function: the potential as a function of time. Must be a scalar or a (callable) function.
                If a scalar, the potential is taken to be constantly equal to this value inside the support.
                If a function, the potential is taken to be equal to the function values inside the interval
                specified by the support argument, and zero outside this interval.
    
    ----------------- Optional parameters ----------------- 

    - support: support of the function, list of two values. Defaults to [-np.inf, np.inf]
    
    
    - Range: Range of the function. Specifying this parameter helps to determine more accurately the number of
    Fourier components required to accurately calculate the Wigner transform of self-energies.

    - derivative: specify the derivative of the potential. This is in fact not used in the electrodes.

    - antiderivative: specify the antiderivative of the potential.




    """

    def __init__(self,function,support=[0,0],Range=[None, None],derivative = None,antiderivative = None):
        #support of [0,0] signifies a range of - infty to infty
        self.function = function
        if support == [0,0]:
            support[0] = -np.inf
            support[1] = np.inf
        self.support = support
        if not callable(function): #potential is constant
            self.potential_is_constant_in_support = True
            if self.support_is_infinite():
                self.potential_is_constant_everywhere = True
            else: 
                self.potential_is_constant_everywhere = False
        else:
            self.potential_is_constant_in_support = False
            self.potential_is_constant_everywhere = False
        
        self.Range = Range
        self.dfdt = derivative
        self.antiderivative = antiderivative

    def __call__(self,t):
        t = np.array(t)
        f = np.zeros(t.shape)
        t_idx = self.is_in_support(t)
        if self.potential_is_constant_in_support:
            f[t_idx] = self.function
        else:
            f[t_idx] = self.function(t[t_idx])
        return f

    def is_in_support(self,t): #checks if t is inside the support of the potential
        a,b = self.support #enforce the correct support of function
        if hasattr(t,'__iter__'):
            t = np.array(t)
            in_support = (t>a) * (t<b) #array containing True at index i if t[i] is inside the support of f; False otherwise
        elif t<a:
            in_support = False
        elif t>b:
            in_support = False
        return in_support

    def support_is_finite(self):
        a,b = self.support
        if a == np.inf or a == -np.inf or b == np.inf or b == -np.inf:
            return False
        else:
            return True
    def support_is_infinite(self):
        return not self.support_is_finite()


    def integrate(self,x0=None,x1=None,N=100): #N the number of steps used for numerical integration
        if x0 is None:
            x0 = self.support[0]
        if x1 is None:
            x1 = self.support[1]

        if self.antiderivative is not None:
            F1 = self.antiderivative(x1)
            F0 = self.antiderivative(x0)
            F = F1-F0
        else:
            if x0 == np.inf or x0 == -np.inf or x1 == np.inf or x1 == -np.inf:
                print('Warning: numerical integration over an infinite domain is not stable')

            x,w = AE_math.get_integration_var(N,x0,x1)
            F = np.sum(self.__call__(x)*w)

        return F

    def derivative(self,t,dx=1e-4,order=3,args=()):
        t = np.array(t)
        df = np.zeros(t.shape)
        if self.potential_is_constant_everywhere:
            return df
        elif self.potential_is_constant_in_support:
            #if self.support_is_finite():
            a,b = self.support
            ta_idx = np.where(t==a)[0]
            if len(ta_idx) == 0:
                ta_idx = np.where(np.diff(np.sign(t-a)))[0]
            tb_idx = np.where(t==b)[0]
            if len(tb_idx) == 0:
                tb_idx = np.where(np.diff(np.sign(t-b)))[0]
            if len(ta_idx) > 0:
                dt = t[ta_idx + 1] - t[ta_idx]
                df[ta_idx] = self.function/dt
            if len(tb_idx) > 0:
                dt = t[tb_idx + 1] - t[tb_idx]
                df[tb_idx] = self.function/dt
            return df
        else:
            t_idx = self.is_in_support(t)
            if self.dfdt is None:
                df[t_idx] = AE_math.FD(self.function,t[t_idx],dx=dx,order=order,args=args)
            else:
                df[t_idx] = self.dfdt(t[t_idx])
            return df

    def get_numerical_antiderivative(self,spline_density = 20): #spline density: number of splines per time unit
        f = self.function
        if self.potential_is_constant_everywhere:
            def antiderivative(t):
                t = np.array(t)
                f = t*self.function
                return f
        elif self.potential_is_constant_in_support:
            def antiderivative(t):
                t = np.array(t)
                f = t*self.function
                a,b = self.support                    
                f[t<a] = a*self.function
                f[t>b] = b*self.function
                return f
        else:
            n_int=5
            x,w = AE_math.get_integration_var(n_int)
            if self.support_is_infinite():
                print('error in expint: support of potential must be finite for this integration to be possible!')
                assert 1==0
            a,b = self.support
            N_steps = int(np.ceil((b-a)*spline_density))
            t = np.linspace(a,b,N_steps)

            def int_gauss(f,tmin,tmax,N=n_int):
                t = tmin + x*(tmax-tmin)
                q = w*(tmax-tmin)
                res=0
                for i in range(N):
                    res += f(t[i])*q[i]
                return res

            F=[0]
            Fcum=0
            t0 = a
            for tt in t[1:]: 
                Fcum += int_gauss(f,t0,tt)
                F.append(Fcum)
                t0=tt
            F=np.array(F)

            antiderivative = AE_math.Spline(t,F,bc_type='constant')
        self.antiderivative = antiderivative
        return



#pot = Potential(np.sin,support=[-1,1])
#t = np.linspace(-np.pi,np.pi,101)
#import matplotlib.pyplot as plt
#plt.plot(t,pot(t))
#plt.plot(t,np.sin(t),ls='dashed')
#plt.plot(t,pot.derivative(t))
#plt.show()
#print(pot.integrate(0,1))


class Electrode:
    #Self-variables that should be specified 
    #Gamma
    #kT
    #mu
    #potential

    #optional variables
    #potential_dT
    #potential_int
    #use_aux_modes
    #eta # the value of eta used to calculate Gamma
    #V_max
    #V_min
    
    #internal variables
    #WBL_bool
    #basis_size





    def __init__(self,Gamma=None,kT=0.1,mu=0,potential = Potential(0),bandwidth=None,max_dw=0.01,use_aux_modes=False,eta=0,coupling_index=None):
        
        self.kT = kT
        self.mu = mu
        self.potential = potential
        self.use_aux_modes = use_aux_modes
        self.bandwidth = bandwidth
        self.set_Gamma(Gamma)
        self.eta = eta
        self.max_dw = max_dw
        if coupling_index is None:
            coupling_index_matrix = np.ones((self.basis_size,self.basis_size)) #if not specified, assume that every index couples
        else: 
            coupling_index_matrix = np.zeros((self.basis_size,self.basis_size))
            for idx in coupling_index:
                #coupling_index_matrix[]
                pass

    def fermi(self,eps,T=0):
        if callable(self.mu):
            mu = self.mu(T)
        else:
            mu = self.mu
        exp = AE_math.exp(-(eps-mu)/self.kT)
        return exp/(1+exp)

    def set_Gamma(self,Gamma):
        if Gamma is None:
            self.Gamma = None
            return  
        elif callable(Gamma):
            self.WBL_bool = False
            if self.bandwidth is None:
                print('Warning: bandwidth (support) of self-energy should be specified for best performance')
                print('Defaulting to setting: bandwidth = [-5,5].')
                self.bandwidth = [-5,5]
        else:
            self.WBL_bool = True
            self.bandwidth=[-np.inf,np.inf]
            Gamma_mat = Gamma
            Gamma = lambda x : np.ones(np.array(x).shape)*Gamma_mat #make gamma a callable function so that syntax is the same in all cases
        self.Gamma = Gamma
        
        #get number of basis functions in device
        test_W = np.array([[[0]]])
        self.basis_size = Gamma(test_W).shape[-1] #calls Gamma and returns size of array along last dimension - this should correspond to the device dimension.
        return









class Device:
    #Self-variables that should be specified 
    #H
    #kT
    #mu
    #potential

    #optional variables
    #H_dT
    #H_int
    #potential_dT
    #potential_int
    #use_aux_modes


    basis_size = None

    def __init__(self,H0=None,kT=0.1, Ht = None,T=None, H_dT = None):
        
        self.kT = kT
        self.T = T #Times at which the Hamiltonian is evaluated, in case Ht is given as an array
        self.set_H(H0,Ht)
        self.H_dT = H_dT
        
    def set_H(self,H0,Ht): #function to set self-variable; in future may allow more flexibility for inputting H.
        if Ht is None and H0 is None:
            self.H=None
            self.H0=None
            self.Ht=None
            return
        #Either Ht or H0 is specified
        

        if H0 is not None: #H0 is specified, Ht may or may not be
            H0 = np.array(H0)
            if len(H0.shape) < 2:
                H0 = H0.reshape(1,1)
            self.n_orbitals = H0.shape[-1] 
            self.basis_size = self.n_orbitals
        self.H0 = H0


        if Ht is not None: #Ht is specified, H0 may or may not be
            if not callable(Ht):
                print('Time-dependent part of Hamiltonian is not callable - using spline interpolation instead.')
                if self.T is None:
                    self.T = np.arange(len(Ht))
                if len(Ht.shape) > 3: #shape of input is (T,1,n,n) - this is one index too many. We remove all indices of dimension 1.
                    Ht = np.squeeze(Ht)
                if len(Ht.shape) < 3: #Ht should at least have shape (T,n,n) if it is a matrix. If its dimension is shorter, it cannot be a matrix!
                    Ht = Ht.reshape(-1,1,1)

                Ht = AE_math.Spline(self.T,Ht,MatrixFunction=True)
            H_t0 = Ht(np.array([0]))
            self.n_orbitals = H_t0.shape[-1] 
            self.basis_size = self.n_orbitals
            if H0 is None:
                H0 = np.zeros((self.basis_size,self.basis_size))
        else: #Ht is none: then H0 cannot be
            Ht = lambda t : np.zeros(H0.shape)

        self.Ht = Ht

        self.H = lambda t : self.H0 + self.Ht(t)
        return


class Meromorphic:
    #class to handle meromorphic functions, i.e. functions specified in terms of a set of poles and residues.
    def __init__(self,poles,residues,real=False): 
        #poles: complex np.array with shape (N,)
        #residues: complex np.ndarray with shape (N,X,Y,Z,...). The first indices are the tensor value of the residue at the N'th pole
        #real: specifies if the function is real. If true, only poles in one of the half-planes must be specified.
        self.poles = np.array(poles)
        self.residues = np.array(residues)
        self.N = len(self.poles)
        self.dim = self.residues.shape[1:] #dimension of the range of our function
        self.real = real
        if np.any(np.imag(self.poles) == 0):
            print('Warning in Meromorphic initialization: There is a pole on the real axis')

    def __call__(self,x):
        in_shape = list(x.shape) #shape = [1,300] for instance
        result_shape = in_shape + list(self.dim)  #[1,300,2,2]
        x_new_shape = in_shape + list(np.ones(len(self.dim),dtype=int)) #[1,300,1,1]
        x = x.reshape(x_new_shape) #[1,300,1,1]
        result = np.zeros(result_shape,dtype=np.complex128) #[1,300,2,2]
        for i in range(self.N):
            result = result + self.residues[i]/(x - self.poles[i]) #[2,2] / ([1,300,1,1] - #) #=? [1,300,2,2]
        if self.real and np.all(np.isreal(x)):
            return result.real
        return result

    def __mul__(self,f): #f another meromorphic function
        f_poles = f.poles
        f_residues = f.residues
        try:
            new_dim = self.dim# @ f.dim
        except:
            print('Error in __mul__: shape mismatch! Shape, ',self.dim, 'is not compatible with shape ', f.dim)
            assert 1==0
        new_N = self.N + f.N
        if self.real and f.real:
            new_real = True
        else:
            new_real = False
        new_poles = np.concatenate((self.poles,f.poles))
        new_residues = np.zeros(([new_N] + list(new_dim)),dtype=np.complex128)
        for i in range(self.N):
            new_residues[i] = self.residues[i] @ f(self.poles[i])
        for i in range(f.N):
            new_residues[self.N + i] = self.__call__(f.poles[i]) @ f.residues[i]
        g = Meromorphic(new_poles,new_residues,new_real)
        return g


    def integrate(self,f=lambda t : 1,half_plane = 'upper'):
        #integrates self around a contour going from -infty to +infty along the real axis, then closing itself in the 
        #in the upper or lower infinite half plane.
        #f is an optionally specified analytic function against which self is integrated
        
        if half_plane == 'upper':
            idx_poles_inside_contour = np.where(np.imag(self.poles)>0)
            prefactor = 2j * np.pi
        elif half_plane == 'lower':
            idx_poles_inside_contour = np.where(np.imag(self.poles)<0)
            prefactor = -2j * np.pi
        #!TODO fix so that f may return a vector or matrix ... 
        integral = prefactor * np.sum(self.residues[idx_poles_inside_contour] * f(self.poles[idx_poles_inside_contour]),axis=0)
        return integral




"""


poles = np.array([1j,-1j])
residues = np.array([1])*np.array([1j,-1j]).reshape(-1,1,1)
print(residues.shape)
f = Meromorphic(poles,residues,real=True)
g = Meromorphic(poles+1,residues,real=True)
x = np.linspace(-3,3,100)
print(f.integrate(half_plane='lower'))
print('f.dim = ', f.dim)
fx = f(x)
fx = fx.real
print('fx shape = ', fx.shape)
import matplotlib.pyplot as plt
plt.plot(x,fx[:,0,0])
plt.plot(x,g(x)[:,0,0])

#def __mul__(self,f):
print(g.poles)
h = f*g
plt.plot(x,h(x)[:,0,0])
plt.show()
print(h.poles)
print(h(x).shape)
print(h.residues)
print(f.residues[0]@g(f.poles[0]))
print(h.integrate())
print(h.real)
"""
"""
def calc_Sigma_less(self,T,omega,derivative=0,extension_length=300,nyquist_damping=5,use_aux_modes=None): #new one with aux modes
        #derivative calculates the derivative of the specified order w.r.t omega
        #nyquist_damping: the damping of the Gaussian filter at the nyquist frequency given in standard deviations. 
        #default corresponds to 3 standard deviations.
        #    f = self.f_R
        #    Gamma = self.Gamma_R
        #    potential_int = self.potential_R_int
        #    Lorentz_params = self.Lorentz_params_R
        #    Fermi_params = self.Fermi_params_R
        #    coupling_index = self.coupling_index_R
        #    #Delta_alpha = self.Delta_R
        #    WBL_check = self.WBL_R
        Gamma = self.Gamma
        if Gamma == None:
            print('Error in calc_Sigma_less: Gamma not specified!')
            assert 1==0
        if use_aux_modes == None:
            use_aux_modes = self.use_aux_modes
        T=np.array(T).reshape(-1,1)
        omega=np.array(omega).reshape(1,-1)   
        exp = np.exp
        omega_min = omega.min()
        omega_max = omega.max()
        dw = (omega_max - omega_min)/(np.size(omega)-1)
        #extend internal arrays by extension_length. this is useful if the specified range of integration does not include the entire support of the given functions.
        extension = dw*np.linspace(1,extension_length,extension_length)
        w = np.concatenate(((omega_min - np.flip(extension)),omega.flatten(),omega_max + extension))
        w=w.reshape(1,-1)
        #print('w[extension_length] == omega_min? w[ext_len]==', w[extension_length],'omega_min ==',omega_min)
        #dw_taumax = 2*np.(Tmax - Tmin)
        #Calculate range of integration needed for sufficiently fine tau sampling (dt = 2pi/(b-a)) and nyquist: 1/dt > 2*max freq
        potential = self.potential(T)
        V_max = np.max(potential)
        V_min = np.min(potential)
        max_freq = V_max - V_min
        if max_freq==0:
            print('error in calc_sigma_less: V_max == V_min.')
            print('code here is incomplete; the function should be able to handle a constant potential...')
        dtau_max = 1/(2.1*max_freq) #ensure that the tau sampling occurs above the Nyquist frequency of the "filter", exp(-\infint \Delta_\alpha ...)
        min_domain_length = 2*np.pi/dtau_max
        domain_length = max(w.max()-w.min(), min_domain_length) #We should at least include the entire range [omega_min, omega_max].
        N1 = np.ceil(domain_length/dw) 
        #tau_max_for_entire_domain = 2*(Tmax - Tmin)
        #N2 = np.ceil(tau_max_for_entire_domain/(2*np.pi)*domain_length + 1)
        #N3=max(N1,N2)
        N = int(2**np.ceil(np.log(N1)/np.log(2))) #Make N a power of two for most efficient FFT. do so in a way that cannot decrease the value of N.
        #if N>1e5:
            #print("N in calc_Sigma_less larger than 100k!")
        #x = domain_length/N * np.arange(N) #generate x variable for FFT.
        x = dw*np.arange(N)
        domain_length= x.max() * N/(N-1)
        x = x.reshape(1,-1) #shape to proper size
        
        dtau = 2*np.pi/domain_length
        tau = dtau*np.arange(N) #generate tau variable for FFT
        tau = tau.reshape(1,-1) #reshape for use in expint.
        
        Nyquist_freq = tau.max()/2
        Nyquist_filter = exp(-(nyquist_damping*tau/Nyquist_freq)**2/2) #filter that reduces the amplitude at the Nyquist freq by 3 standard deviations of the normal distribution
        #np.save('Nyquist_*filter',Nyquist_filter)
        #np.save('tau',tau)
        #calculate expint
        if self.potential_int is None:
            expint = exp(-1j*self.expint3(T,tau,alpha))[:,:,0,0]
        else:
            expint = exp(-1j*(potential_int(T+tau/2) - potential_int(T-tau/2)))

        Sigma_less = np.zeros((np.size(T),np.size(omega),self.basis_size,self.basis_size),dtype=np.complex128)
        omega_index = slice(extension_length,(np.size(omega)+extension_length))
        if not use_aux_modes:
            fermi = self.fermi(x+w.min())
            Gam = Gamma(x.reshape(1,-1,1,1)+w.min())
        for i in range(self.device_dim):
            for j in range(self.device_dim):
                if coupling_index[i,j]: #each iteration calculates one matrix element of Sigma. Variables in the loop do not carry matrix indices.
                    if self.use_aux_modes:
                        fermi_int_residues = lambda E : Gamma(E,i,j)*exp(-1j*(E)*tau)
                        Lorentz_int_residues = lambda E : self.Fermi_pade(E,alpha)*exp(-1j*(E)*tau)#f(E)*exp(-1j*(E)*taup)#

                        Lor_poles, Lor_res = Lorentz_params
                        Lorentz_params_ij = [Lor_poles,Lor_res[:,i,j]]
                        fermi_int = self.int_residues2(fermi_int_residues,Fermi_params,halfplane='lower')
                        Lorentz_int = self.int_residues2(Lorentz_int_residues,Lorentz_params_ij,halfplane='lower')
                        forward_trans = (fermi_int + Lorentz_int)/(2*np.pi)
                    else:
                        #forward transform (energy integral)
                        fxm = domain_length/(2*np.pi*N)*Gam[:,:,i,j]*fermi        
                        forward_trans = self.fft(fxm,axis=1) * exp(-1j*w.min()*tau)
                    
                    forward_trans = forward_trans * Nyquist_filter
                    if derivative > 0:
                        forward_trans *= (1j*tau)**derivative
                    forward_trans = forward_trans * expint
                    forward_trans[:,0] /= 2
                    back_trans = tau.max() * self.ifft(exp(1j*w.min()*tau)*forward_trans,axis=1)

                    Sigma_less[:,:,i,j] =   2j*np.real(back_trans[:,omega_index])

        return Sigma_less



    def expint3(self,T,tau,alpha): #T is Nx1, tau is 1xM. alpha is a char specifying the left lead, right lead or central region.
        #Suppose the support of the pulse is contained in the interval [Tmin, Tmax].
        #t0=time.time()

        #set the correct potential
        if alpha == 'L':
            pot=self.potential_L
        elif alpha =='R':
            pot = self.potential_R
        else:
            pot = self.potential
            #print('error in expint3: alpha not ==R or L. This should be specified at this point!')
            assert 1==0

        #check if a new calculation is necessary
        if np.array_equal(T,self.T) and np.array_equal(tau,self.tau):
            if alpha == 'L':
                if not np.all(self.expint_L==None):
                    #print('expint3: returning stored value, alpha == L')
                    return self.expint_L #return alreay calculated value; otherwise proceed.
            elif alpha=='R':
                if not np.all(self.expint_R==None):
                    #print('expint3: returning stored value, alpha == R')
                    return self.expint_R

        n_int=5
        x,w = self.get_integration_var(n_int)

        def int_gauss(f,tmin,tmax,N=n_int):
            t = tmin + x*(tmax-tmin)
            q = w*(tmax-tmin)
            res=0
            for i in range(N):
                res += f(t[i])*q[i]
            return res
        def antiderivative(t,t0=None): #integrates the function pot and gives the antiderivative evaluated as F(t) - F(t0)
            F=[]
            Fcum=0
            if t0 is None:
                t0 = t[0]
            for tt in t:
                Fcum += int_gauss(pot,t0,tt)
                F.append(Fcum)
                t0=tt
            F=np.array(F)
            return F
        T1=T.flatten()
        tau1=tau.flatten()
        F = np.zeros((np.size(T),np.size(tau)))
        for i, TT in enumerate(T1):
            #for every t, calc the integral with bounds T-tau/2 .. T+tau/2 for every value of tau.
            #this is done by calculating the antiderivative for every value of T+tau/2 and every value T-tau/2, then subtracting the two.
                tlist1 = TT+tau1/2
                tlist2 = TT-tau1/2
                F1 = antiderivative(tlist1)
                F2 = antiderivative(tlist2)
                t0_correction = 0
                if tlist1[0] != tlist2[0]: #t0 is not the same in the two derivatives, leading to constant offset. this should fix
                    t01 = tlist1[0]
                    t02 = tlist2[0]
                    t0_correction = quad(pot,t01,t02)[0] #integrate with scipy
                F[i]=F1-F2-t0_correction
        #print('calculated integral of Vbias in %.2f'%(time.time()-t0),flush=True)

        F=F.reshape(np.size(T),np.size(tau),1,1)
        if np.array_equal(T,self.T) and np.array_equal(tau,self.tau):
            if alpha == 'L':
                self.expint_L = F
                #print('setting self.expint_L = F')
            elif alpha=='R':
                self.expint_R = F
                #print('setting self.expint_R = F')
        return F
        """