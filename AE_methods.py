import numpy as np
#from scipy.special import jv,jvp
import scipy as scipy
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv
from scipy.integrate import quad
#from numpy.polynomial.chebyshev import Chebyshev as cheb
#from scipy.linalg import expm
#import numpy.polynomial.chebyshev
import math
import time
import matplotlib.pyplot as plt
#import PadeDecomp
#from numba import njit, prange, jit
#from scipy.interpolate import CubicSpline
#matrix structure of internal variables: #aux, aux , ... T, omega, [device space matrix]
#import os, psutil
#process = psutil.Process(os.getpid())
from scipy.signal import hilbert as scipy_hilbert
import AE_math
from AE_classes import *

import joblib as jl



def calc_current(T,Device,Electrode_L,Electrode_R,order=2,eta=None,omega=None,nyquist_damping=5,calc_density=False,side='both',save_arrays='none',name=None):
    """
This function is the main workhorse of the code; using the adiabatic expansion technique, it calculates Greens functions and currents to a desired order (max 2) in the central time derivative.
It uses the functions calc_Sigma_R and calc_Sigma_less to calculate the Wigner transforms of the specified self-energies for each electrode.

The mandatory parameters are

- T (scalar or np.array): list of times for which the calculation should be carried out

- Device (instance of AE_classes.Device): An object specifying the device, which should include the time-dependent device Hamiltonian. See AE_classes.Device

- Electrode_L (instance of AE_classes.Electrode): An object specifying the left electrode, see AE_classes.Electrode

- Electrode_R (instance of AE_classes.Electrode): An object specifying the right electrode, see AE_classes.Electrode

------------------ optional parameters ------------------ 

- order: The max order to which the calculation is carried out. Must be 0, 1, or 2.

- eta: The imaginary eta used to smoothen out functions. If not specified, defaults to the maximum eta used in the electrodes. Must be non-negative.

- omega (np.array): frequency points at which the parameters should be calculated. Must be uniformly spaced. Finer sampling leads to more accurate results, and the range should cover all energies where dynamics may occur. 

- nyquist_damping: Width of filter used to smoothen the wigner-transformed self-energies. Removes rapid oscillations near discontinuities of fourier-transformed variables. Has similar effect as an imaginary eta. Larger values -> more filtering. 

- calc_density (boolean): specifies whether to calculate and return the density of electrons in the device as a function of time. This is calculated from the lesser GF.

- side: specify whether to calculate the current into the device from both electrodes, or only one of them. Must be either 'left', 'right', or 'both'.

- save_arrays: specifies whether to save the Green's functions and energy-resolved current matrices used in the internal calculations. Must be either 'none', in which case nothing is saved;
                'diag' in which case the diagonal of the arrays are saved (i.e. corresponding to on-site elements of the Device hamiltonian)';
                or 'all', in which case the entire arrays are saved.

- name: if specified, saves arrays to files with the given name as a prefix

------------------ returns ------------------ 

- out: a dict containing the specified output variables.


"""    
    out = {} #dict holding the output variables
    initial_start_time = time.time()
    deriv = AE_math.FD
    T=np.array(T).reshape(-1,1,1,1)
    omega = np.array(omega).reshape(1,-1,1,1)

    if eta is None:
        eta = max(Electrode_L.eta,Electrode_R.eta)

    if omega is None:     
        print('error in calc_current: omega was None. For now omega must be specified')
        assert 1==0

    else:
        w=omega.flatten()
        omega_weights = (w[1:] - w[:-1]) #calculate dw
        omega_weights[0] = omega_weights[0]/2 #trapez: divide first weight by 2
        omega_weights = list(omega_weights)
        omega_weights.append(omega_weights[-1]/2)
        omega_weights = np.array(omega_weights).reshape(np.shape(omega))
    f_Sigma_L_R = lambda t : calc_Sigma_R(t,omega,Electrode_L,nyquist_damping=nyquist_damping)
    f_Sigma_R_R = lambda t : calc_Sigma_R(t,omega,Electrode_R,nyquist_damping=nyquist_damping)
    f_Sigma_L_less = lambda t : calc_Sigma_less(t,omega,Electrode_L,nyquist_damping=nyquist_damping)
    f_Sigma_R_less = lambda t : calc_Sigma_less(t,omega,Electrode_R,nyquist_damping=nyquist_damping)
    J0_L = np.zeros(len(T))
    J1_L = np.zeros(len(T))
    J2_L = np.zeros(len(T))
    J0_R = np.zeros(len(T))
    J1_R = np.zeros(len(T))
    J2_R = np.zeros(len(T))
    if calc_density:
        density0 = np.zeros((np.size(T),Device.basis_size,Device.basis_size))
        density1 = np.zeros((np.size(T),Device.basis_size,Device.basis_size))
        density2 = np.zeros((np.size(T),Device.basis_size,Device.basis_size))
    if save_arrays=='all':
        G0_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        G0_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi0_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi0_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        G1_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        G1_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi1_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi1_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        G2_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        G2_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi2_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
        Pi2_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size,Device.basis_size),dtype=np.complex128)
    elif save_arrays=='diag':
        G0_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        G0_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi0_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi0_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        G1_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        G1_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi1_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi1_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        G2_less_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        G2_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi2_L_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
        Pi2_R_array = np.zeros((np.size(T),np.size(omega),Device.basis_size),dtype=np.complex128)
    

    #for i in prange(len(T)):

    #print('loop %d/%d'%(i+1,len(T)),flush=True)
    t=T#T[i].reshape(1,1,1,1)
    if order == 0:
        Sigma_L_R = f_Sigma_L_R(t)
        Sigma_R_R = f_Sigma_R_R(t)
        Sigma_L_less = f_Sigma_L_less(t)
        Sigma_R_less = f_Sigma_R_less(t)

    elif order == 1:
        Sigma_L_R, Sigma_L_R_dw = calc_Sigma_R(t,omega,Electrode_L,nyquist_damping=nyquist_damping,derivative=[0,1])
        Sigma_R_R, Sigma_R_R_dw = calc_Sigma_R(t,omega,Electrode_R,nyquist_damping=nyquist_damping,derivative=[0,1])
        Sigma_L_less, Sigma_L_less_dw = calc_Sigma_less(t,omega,Electrode_L,nyquist_damping=nyquist_damping,derivative=[0,1])
        Sigma_R_less, Sigma_R_less_dw = calc_Sigma_less(t,omega,Electrode_R,nyquist_damping=nyquist_damping,derivative=[0,1])

    elif order == 2:
        Sigma_L_R, Sigma_L_R_dw, Sigma_L_R_d2w = calc_Sigma_R(t,omega,Electrode_L,nyquist_damping=nyquist_damping,derivative=[0,1,2])
        Sigma_R_R, Sigma_R_R_dw, Sigma_R_R_d2w = calc_Sigma_R(t,omega,Electrode_R,nyquist_damping=nyquist_damping,derivative=[0,1,2])
        Sigma_L_less, Sigma_L_less_dw, Sigma_L_less_d2w = calc_Sigma_less(t,omega,Electrode_L,nyquist_damping=nyquist_damping,derivative=[0,1,2])
        Sigma_R_less, Sigma_R_less_dw, Sigma_R_less_d2w = calc_Sigma_less(t,omega,Electrode_R,nyquist_damping=nyquist_damping,derivative=[0,1,2])

    Sigma_L_A = np.conjugate(Sigma_L_R)
    Sigma_R_A = np.conjugate(Sigma_R_R)
    Sigma_R = Sigma_R_R + Sigma_L_R
    #Sigma_A = Sigma_R_A + Sigma_L_A
    Sigma_less = Sigma_L_less + Sigma_R_less

    Sigma_L_less_dT = deriv(f_Sigma_L_less ,t)
    Sigma_R_less_dT = deriv(f_Sigma_R_less ,t)
    Sigma_L_R_dT = deriv(f_Sigma_L_R ,t)
    Sigma_R_R_dT = deriv(f_Sigma_R_R ,t)

    Sigma_L_A_dT = np.conjugate(Sigma_L_R_dT)
    Sigma_R_A_dT = np.conjugate(Sigma_R_R_dT)

    Sigma_L_A_dw = np.conjugate(Sigma_L_R_dw)
    Sigma_R_A_dw = np.conjugate(Sigma_R_R_dw)

    #Total self energies
    Sigma_less_dT = Sigma_L_less_dT + Sigma_R_less_dT
    Sigma_less_dw = Sigma_L_less_dw + Sigma_R_less_dw

    Sigma_R_dT =  Sigma_R_R_dT + Sigma_L_R_dT
    Sigma_A_dT =  Sigma_R_A_dT + Sigma_L_A_dT
    Sigma_R_dw=  Sigma_R_R_dw + Sigma_L_R_dw
    Sigma_A_dw=  Sigma_R_A_dw + Sigma_L_A_dw
    #print('Calculated derivatives',flush=True)


    H =  Device.H(t)
    if Device.H_dT is None:
        H_dT = deriv(Device.H,t)
    else:
        H_dT = Device.H_dT(t)

    domain_length = omega.max() - omega.min()
    dtau = 2*np.pi/domain_length
    tau = dtau*np.arange(omega.size).reshape(1,-1,1,1)
    #Nyquist_freq = tau.max()/2
    #Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2)
    L0_R = (omega+1j*eta)*np.identity(Device.basis_size) - H - Sigma_R
    L0_A = np.conjugate(L0_R)
    G0_R_unfiltered = inv(L0_R)
    #G0_R_filtered = ifft(Nyquist_filter*fft(G0_R_unfiltered,axis=1),axis=1)
    #change_in_norm = np.sqrt(np.sum(G0_R_unfiltered**2)/np.sum(G0_R_filtered**2))
    #if  np.abs(change_in_norm - 1) > 0.1:
    #print('Filtering G0_R changed its norm by more than 10%. Results may not be reliable.')
    
    G0_R = G0_R_unfiltered
    G0_A = np.conjugate(G0_R)
    G0_less = G0_R@Sigma_less@G0_A
    #Current matrices - zero order. Eq (35)
    if side=='left' or side=='both':
        Pi0_L = G0_less@Sigma_L_A + G0_R@Sigma_L_less
        

        #The integral over omega is calculated as the vector product with the gauss-legendre weights. Eq (34)
        #J0_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_L)*omega_weights,axis1=2,axis2=3),axis=1)
        J0_L = 1/np.pi * np.sum(np.trace(np.real(Pi0_L)*omega_weights,axis1=2,axis2=3),axis=1)


    if side=='right' or side=='both':
        Pi0_R = G0_less@Sigma_R_A + G0_R@Sigma_R_less
        #J0_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi0_R)*omega_weights,axis1=2,axis2=3),axis=1)
        J0_R = 1/np.pi * np.sum(np.trace(np.real(Pi0_R)*omega_weights,axis1=2,axis2=3),axis=1)

    if calc_density:
        #density0[i]=np.sum(omega_weights*np.imag(G0_less),axis=1)/(2*np.pi) 
        density0=np.sum(omega_weights*np.imag(G0_less),axis=1)/(2*np.pi) 
        
    if save_arrays == 'all':
        #G0_R_array[i] = G0_R
        G0_R_array = G0_R
        #G0_less_array[i] = G0_less
        G0_less_array = G0_less
        if side=='left' or side=='both':
            #Pi0_L_array[i] = Pi0_L
            Pi0_L_array = Pi0_L
        if side=='right' or side=='both':
            #Pi0_R_array[i] = Pi0_R
            Pi0_R_array = Pi0_R
    elif save_arrays== 'diag':
        #G0_R_array[i] = np.diagonal(G0_R,axis1=2,axis2=3)
        G0_R_array = np.diagonal(G0_R,axis1=2,axis2=3)
        #G0_less_array[i] = np.diagonal(G0_less,axis1=2,axis2=3)
        G0_less_array = np.diagonal(G0_less,axis1=2,axis2=3)
        if side=='left' or side=='both':
            #Pi0_L_array[i] = np.diagonal(Pi0_L,axis1=2,axis2=3)
            Pi0_L_array = np.diagonal(Pi0_L,axis1=2,axis2=3)
        if side=='right' or side=='both':
            #Pi0_R_array[i] = np.diagonal(Pi0_R,axis1=2,axis2=3)
            Pi0_R_array = np.diagonal(Pi0_R,axis1=2,axis2=3)

    #if order == 0:
     #   continue

    #FIRST ORDER
    if order > 0:
        #green function derivatives. Eqs (30), (29), (31)
        G0_R_dT = G0_R @ (H_dT+Sigma_R_dT)@G0_R
        G0_A_dT = np.conjugate(G0_R_dT)

        G0_R_dw = -G0_R@G0_R + G0_R@Sigma_R_dw@G0_R 
        G0_A_dw = np.conjugate(G0_R_dw)

        G0_less_dT = G0_R_dT@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dT + G0_R@Sigma_less_dT@G0_A
        G0_less_dw = G0_R_dw@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_dw + G0_R@Sigma_less_dw@G0_A

        #First order green functions. Eqs (28) and (32)
        G1_R =  1j/2 * G0_R @  (-G0_R_dT - H_dT@G0_R_dw - Sigma_R_dT@G0_R_dw + Sigma_R_dw @ G0_R_dT)

        G1_A =  - np.conjugate(G1_R) #1j/2 * G0_A @  (-G0_A_dT - H_dT@G0_A_dw - Sigma_A_dT@G0_A_dw + Sigma_A_dw @ G0_A_dT)

        G1_less = G0_R@Sigma_less@G1_A + 1j/2*G0_R@(
                    -G0_less_dT - H_dT @ G0_less_dw - Sigma_less_dT@G0_A_dw + Sigma_less_dw@G0_A_dT - Sigma_R_dT@G0_less_dw + Sigma_R_dw@G0_less_dT
                    )

        #print('calculated G1',flush=True)
        #First order current matrices
        if side=='left' or side=='both':
            Pi1_L = G1_less@Sigma_L_A + G1_R@Sigma_L_less - 1j/2 * G0_less_dT@Sigma_L_A_dw + 1j/2*G0_less_dw@Sigma_L_A_dT -1j/2*G0_R_dT@Sigma_L_less_dw + 1j/2*G0_R_dw@Sigma_L_less_dT
            #J1_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_L)*omega_weights,axis1=2,axis2=3),axis=1)
            J1_L = 1/np.pi * np.sum(np.trace(np.real(Pi1_L)*omega_weights,axis1=2,axis2=3),axis=1)
        #del(Pi1_L)
        if side=='right' or side=='both':
            Pi1_R = G1_less@Sigma_R_A + G1_R@Sigma_R_less - 1j/2 * G0_less_dT@Sigma_R_A_dw + 1j/2*G0_less_dw@Sigma_R_A_dT -1j/2*G0_R_dT@Sigma_R_less_dw + 1j/2*G0_R_dw@Sigma_R_less_dT
            #J1_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi1_R)*omega_weights,axis1=2,axis2=3),axis=1)
            J1_R = 1/np.pi * np.sum(np.trace(np.real(Pi1_R)*omega_weights,axis1=2,axis2=3),axis=1)
       
        if calc_density:
            #density1[i]=np.sum(omega_weights*np.imag(G1_less),axis=1)/(2*np.pi) 
            density1=np.sum(omega_weights*np.imag(G1_less),axis=1)/(2*np.pi) 
            
        if save_arrays == 'all':
            #G1_R_array[i] = G1_R
            G1_R_array = G1_R
            #G1_less_array[i] = G1_less
            G1_less_array = G1_less
            if side=='left' or side=='both':
                #Pi1_L_array[i] = Pi1_L
                Pi1_L_array = Pi1_L
            if side=='right' or side=='both':
                #Pi1_R_array[i] = Pi1_R
                Pi1_R_array = Pi1_R
        elif save_arrays== 'diag':
            #G1_R_array[i] = np.diagonal(G1_R,axis1=2,axis2=3)
            G1_R_array = np.diagonal(G1_R,axis1=2,axis2=3)
            #G1_less_array[i] = np.diagonal(G1_less,axis1=2,axis2=3)
            G1_less_array = np.diagonal(G1_less,axis1=2,axis2=3)
            if side=='left' or side=='both':
                #Pi1_L_array[i] = np.diagonal(Pi1_L,axis1=2,axis2=3)
                Pi1_L_array = np.diagonal(Pi1_L,axis1=2,axis2=3)
            if side=='right' or side=='both':
                #Pi1_R_array[i] = np.diagonal(Pi1_R,axis1=2,axis2=3)
                Pi1_R_array = np.diagonal(Pi1_R,axis1=2,axis2=3)



    #if order==1: 
     #   continue

    #SECOND ORDER
    if order > 1:
        #Sigma_L_less_d2w = calc_Sigma_less(t,omega,Electrode_L,derivative=2,nyquist_damping=nyquist_damping)
        #Sigma_R_less_d2w = calc_Sigma_less(t,omega,Electrode_R,derivative=2,nyquist_damping=nyquist_damping)
        #Sigma_L_R_d2w = calc_Sigma_R(t,omega,Electrode_L,derivative=2,nyquist_damping=nyquist_damping)
        #Sigma_R_R_d2w = calc_Sigma_R(t,omega,Electrode_R,derivative=2,nyquist_damping=nyquist_damping)
        Sigma_L_less_d2T = deriv(f_Sigma_L_less ,t,n=2)
        Sigma_R_less_d2T = deriv(f_Sigma_R_less ,t,n=2)
        Sigma_L_R_d2T = deriv(f_Sigma_L_R ,t,n=2)
        Sigma_R_R_d2T = deriv(f_Sigma_R_R ,t,n=2)

        f_Sigma_L_less_dw = lambda t : calc_Sigma_less(t,omega,Electrode_L,derivative=1,nyquist_damping=nyquist_damping)
        f_Sigma_R_less_dw = lambda t : calc_Sigma_less(t,omega,Electrode_R,derivative=1,nyquist_damping=nyquist_damping)
        f_Sigma_L_R_dw = lambda t : calc_Sigma_R(t,omega,Electrode_L,derivative=1,nyquist_damping=nyquist_damping)
        f_Sigma_R_R_dw = lambda t : calc_Sigma_R(t,omega,Electrode_R,derivative=1,nyquist_damping=nyquist_damping)
        
        Sigma_L_less_dTdw = deriv(f_Sigma_L_less_dw ,t)
        Sigma_R_less_dTdw = deriv(f_Sigma_R_less_dw ,t)
        Sigma_L_R_dTdw = deriv(f_Sigma_L_R_dw ,t)
        Sigma_R_R_dTdw = deriv(f_Sigma_R_R_dw ,t)        

        Sigma_L_A_d2w = np.conjugate(Sigma_L_R_d2w)
        Sigma_R_A_d2w = np.conjugate(Sigma_R_R_d2w)
        Sigma_L_A_d2T = np.conjugate(Sigma_L_R_d2T)
        Sigma_R_A_d2T = np.conjugate(Sigma_R_R_d2T)
        Sigma_L_A_dTdw = np.conjugate(Sigma_L_R_dTdw)
        Sigma_R_A_dTdw = np.conjugate(Sigma_R_R_dTdw)

        Sigma_less_dTdw  = Sigma_R_less_dTdw + Sigma_L_less_dTdw 
        Sigma_R_dTdw  = Sigma_R_R_dTdw + Sigma_L_R_dTdw 
        Sigma_A_d2w  = Sigma_R_A_d2w + Sigma_L_A_d2w 
        Sigma_A_d2T  = Sigma_R_A_d2T + Sigma_L_A_d2T 
        Sigma_A_dTdw  = Sigma_R_A_dTdw + Sigma_L_A_dTdw 
        Sigma_less_d2w  = Sigma_R_less_d2w + Sigma_L_less_d2w 
        Sigma_R_d2w  = Sigma_R_R_d2w + Sigma_L_R_d2w 
        Sigma_less_d2T  = Sigma_R_less_d2T + Sigma_L_less_d2T 
        Sigma_R_d2T  = Sigma_R_R_d2T + Sigma_L_R_d2T 


        H_d2T = deriv(Device.H,t,n=2)

        G0_R_d2w = 2*G0_R_dw@L0_R@G0_R_dw + G0_R@(Sigma_R_d2w)@G0_R
        G0_R_d2T = 2*G0_R_dT@L0_R@G0_R_dT + G0_R@(H_d2T + Sigma_R_d2T)@G0_R
        G0_R_dTdw = G0_R_dw@L0_R@G0_R_dT + G0_R_dT @L0_R @ G0_R_dw +  G0_R@(Sigma_R_dTdw)@G0_R
        G0_A_d2w = np.conjugate(G0_R_d2w)
        G0_A_d2T = np.conjugate(G0_R_d2T)
        G0_A_dTdw = np.conjugate(G0_R_dTdw)

        G1_R_dw = G0_R_dw @ L0_R @ G1_R + 1j/2*(G0_R_dw @ L0_R @ G0_R_dTdw - G0_R_dT @ L0_R @ G0_R_d2w - G0_R @ ((Sigma_R_dTdw) @ G0_R_dw - (Sigma_R_d2w) @G0_R_dT))
        G1_R_dT = G0_R_dT @ L0_R @ G1_R + 1j/2*(G0_R_dw @ L0_R @ G0_R_d2T - G0_R_dT @ L0_R @ G0_R_dTdw - G0_R @ ((Sigma_R_d2T + H_d2T) @ G0_R_dw - (Sigma_R_dTdw)@ G0_R_dT))
        G1_A_dw = G0_A_dw @ L0_A @ G1_A + 1j/2*(G0_A_dw @ L0_A @ G0_A_dTdw - G0_A_dT @ L0_A @ G0_A_d2w - G0_A @ ((Sigma_A_dTdw) @ G0_A_dw - (Sigma_A_d2w) @G0_A_dT))
        G1_A_dT = G0_A_dT @ L0_A @ G1_A + 1j/2*(G0_A_dw @ L0_A @ G0_A_d2T - G0_A_dT @ L0_A @ G0_A_dTdw - G0_A @ ((Sigma_A_d2T + H_d2T) @ G0_A_dw - (Sigma_A_dTdw)@ G0_A_dT))

        G0_less_d2T = G0_R_d2T@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_d2T + G0_R@Sigma_less_d2T@G0_A + 2*G0_R_dT@Sigma_less_dT@G0_A + 2*G0_R_dT@Sigma_less@G0_A_dT + 2*G0_R@Sigma_less_dT@G0_A_dT
        G0_less_d2w = G0_R_d2w@Sigma_less@G0_A + G0_R@Sigma_less@G0_A_d2w + G0_R@Sigma_less_d2w@G0_A + 2*G0_R_dw@Sigma_less_dw@G0_A + 2*G0_R_dw@Sigma_less@G0_A_dw + 2*G0_R@Sigma_less_dw@G0_A_dw
        G0_less_dTdw = G0_R_dTdw@Sigma_less@G0_A + G0_R_dw@Sigma_less@G0_A_dT + G0_R_dw@Sigma_less_dT@G0_A +G0_R_dT@Sigma_less_dw@G0_A + G0_R@Sigma_less_dw@G0_A_dT + G0_R@Sigma_less_dTdw@G0_A \
                    + G0_R_dT@Sigma_less@G0_A_dw + G0_R@Sigma_less@G0_A_dTdw + G0_R@Sigma_less_dT@G0_A_dw

        G1_less_dT = G0_R_dT @ L0_R @ G1_less \
                   + G0_R@(Sigma_less_dT@G1_A + 1j/2*(-G0_less_d2T - H_d2T @ G0_less_dw - Sigma_less_d2T@G0_A_dw + Sigma_less_dTdw@G0_A_dT - Sigma_R_d2T@G0_less_dw + Sigma_R_dTdw@G0_less_dT))\
                   + G0_R@(Sigma_less@G1_A_dT + 1j/2*(             - H_dT @ G0_less_dTdw - Sigma_less_dT@G0_A_dTdw + Sigma_less_dw@G0_A_d2T - Sigma_R_dT@G0_less_dTdw + Sigma_R_dw@G0_less_d2T))



        G1_less_dw = G0_R_dw @ L0_R @ G1_less \
                   + G0_R@(Sigma_less_dw@G1_A + 1j/2*(-G0_less_dTdw -        Sigma_less_dTdw@G0_A_dw + Sigma_less_d2w@G0_A_dT - Sigma_R_dTdw@G0_less_dw + Sigma_R_d2w@G0_less_dT)) \
                   + G0_R@(Sigma_less@G1_A_dw + 1j/2*(- H_dT @ G0_less_d2w - Sigma_less_dT@G0_A_d2w + Sigma_less_dw@G0_A_dTdw - Sigma_R_dT@G0_less_d2w + Sigma_R_dw@G0_less_dTdw))


        G2_R = - G0_R @ (\
        1/8*(H_d2T @ G0_R_d2w + Sigma_R_d2T@G0_R_d2w + Sigma_R_d2w@G0_R_d2T - 2*Sigma_R_dTdw @ G0_R_dTdw)\
        + 1j/2 * ((H_dT + Sigma_R_dT) @ G1_R_dw + (np.identity(Device.basis_size) - Sigma_R_dw) @ G1_R_dT)
         )

        G2_A = - G0_A @ (\
        1/8*(H_d2T @ G0_A_d2w + Sigma_A_d2T@G0_A_d2w + Sigma_A_d2w@G0_A_d2T - 2*Sigma_A_dTdw @ G0_A_dTdw)\
        + 1j/2 * ((H_dT + Sigma_A_dT) @ G1_A_dw + (np.identity(Device.basis_size) - Sigma_A_dw) @ G1_A_dT)
         )


        G2_less = G0_R @ (\
        Sigma_less @ G2_A - 1j/2 * (Sigma_less_dT@G1_A_dw - Sigma_less_dw@G1_A_dT)\
        - 1/8 * (Sigma_less_d2T@G0_A_d2w + Sigma_less_d2w@G0_A_d2T - 2*Sigma_less_dTdw@G0_A_dTdw)\
        - 1/8*(H_d2T @ G0_less_d2w + Sigma_R_d2T@G0_less_d2w + Sigma_R_d2w@G0_less_d2T - 2*Sigma_R_dTdw @ G0_less_dTdw)\
        - 1j/2 * ((H_dT + Sigma_R_dT) @ G1_less_dw + (np.identity(Device.basis_size) - Sigma_R_dw) @ G1_less_dT)\
         )


        #Second order current matrices
        if side=='left' or side=='both':
            Pi2_L = G2_less@Sigma_L_A + G2_R@Sigma_L_less\
            - 1j/2 * G1_less_dT@Sigma_L_A_dw + 1j/2 * G1_less_dw@Sigma_L_A_dT\
            - 1j/2 * G1_R_dT@Sigma_L_less_dw + 1j/2 * G1_R_dw@Sigma_L_less_dT\
            - 1/8 * (G0_less_d2T @ Sigma_L_A_d2w + G0_less_d2w @ Sigma_L_A_d2T - 2*G0_less_dTdw @ Sigma_L_A_dTdw )\
            - 1/8 * (G0_R_d2T @ Sigma_L_less_d2w + G0_R_d2w @ Sigma_L_less_d2T - 2*G0_R_dTdw @ Sigma_L_less_dTdw )
            
            #J2_L[i] = 1/np.pi * np.sum(np.trace(np.real(Pi2_L)*omega_weights,axis1=2,axis2=3),axis=1)
            J2_L = 1/np.pi * np.sum(np.trace(np.real(Pi2_L)*omega_weights,axis1=2,axis2=3),axis=1)
        #del(Pi1_L)
        if side=='right' or side=='both':
            Pi2_R = G2_less@Sigma_R_A + G2_R@Sigma_R_less - 1j/2 * G1_less_dT@Sigma_R_A_dw + 1j/2*G1_less_dw@Sigma_R_A_dT -1j/2*G1_R_dT@Sigma_R_less_dw + 1j/2*G1_R_dw@Sigma_R_less_dT\
            - 1/8 * (G0_less_d2T @ Sigma_R_A_d2w + G0_less_d2w @ Sigma_R_A_d2T - 2*G0_less_dTdw @ Sigma_R_A_dTdw )\
            - 1/8 * (G0_R_d2T @ Sigma_R_less_d2w + G0_R_d2w @ Sigma_R_less_d2T - 2*G0_R_dTdw @ Sigma_R_less_dTdw )
            
            #J2_R[i] = 1/np.pi * np.sum(np.trace(np.real(Pi2_R)*omega_weights,axis1=2,axis2=3),axis=1)
            J2_R = 1/np.pi * np.sum(np.trace(np.real(Pi2_R)*omega_weights,axis1=2,axis2=3),axis=1)

        if calc_density:
            #density2[i]=np.sum(omega_weights*np.imag(G2_less),axis=1)/(2*np.pi) 
            density2=np.sum(omega_weights*np.imag(G2_less),axis=1)/(2*np.pi) 
            
        if save_arrays == 'all':
            #G2_R_array[i] = G2_R
            G2_R_array = G2_R
            #G2_less_array[i] = G2_less
            G2_less_array = G2_less
            if side=='left' or side=='both':
                #Pi2_L_array[i] = Pi2_L
                Pi2_L_array = Pi2_L
            if side=='right' or side=='both':
                #Pi2_R_array[i] = Pi2_R
                Pi2_R_array = Pi2_R
        elif save_arrays== 'diag':
            #G2_R_array[i] = np.diagonal(G2_R,axis1=2,axis2=3)
            G2_R_array = np.diagonal(G2_R,axis1=2,axis2=3)
            #G2_less_array[i] = np.diagonal(G2_less,axis1=2,axis2=3)
            G2_less_array = np.diagonal(G2_less,axis1=2,axis2=3)
            if side=='left' or side=='both':
                #Pi2_L_array[i] = np.diagonal(Pi2_L,axis1=2,axis2=3)
                Pi2_L_array = np.diagonal(Pi2_L,axis1=2,axis2=3)
            if side=='right' or side=='both':
                #Pi2_R_array[i] = np.diagonal(Pi2_R,axis1=2,axis2=3)
                Pi2_R_array = np.diagonal(Pi2_R,axis1=2,axis2=3)


    end_time = time.time()
    #print('calc_current loop calculated in %.2f seconds'%(end_time-start_time))


    if save_arrays == 'all' or save_arrays == 'diag':
        out['G0_R'] = G0_R_array
        out['G0_less'] = G0_less_array
        if order > 0:
            out['G1_R'] = G1_R_array
            out['G1_less'] = G1_less_array
        if order > 1:
            out['G2_R'] = G2_R_array
            out['G2_less'] = G2_less_array
        if side=='left' or side=='both':
            out['Pi0_L'] = Pi0_L_array
            if order > 0:
                out['Pi1_L'] = Pi1_L_array
            if order > 1:
                out['Pi2_L'] = Pi2_L_array
        if side=='right' or side=='both':
            out['Pi0_R'] = Pi0_R_array
            if order > 0:
                out['Pi1_R'] = Pi1_R_array
            if order > 1:
                out['Pi2_R'] = Pi2_R_array

    end_time = time.time()
    print('calc_current total calculated in %.2f seconds'%(end_time-initial_start_time))

    if side =='left': 
        out['J0_L'] = J0_L
        if order > 0:
            out['J1_L'] = J1_L
        if order > 1:
            out['J2_L'] = J2_L
    elif side=='right':
        out['J0_R'] = J0_R
        if order > 0:
            out['J1_R'] = J1_R
        if order > 1:
            out['J2_R'] = J2_R
    elif side=='both':
        out['J0_L'] = J0_L
        out['J0_R'] = J0_R
        if order > 0:
            out['J1_L'] = J1_L
            out['J1_R'] = J1_R
        if order > 1:
            out['J2_L'] = J2_L
            out['J2_R'] = J2_R
    if calc_density:
        out['density0'] = density0
        if order > 0:
            out['density1'] = density1
        if order > 1:
            out['density2'] = density2
    return out




def old_calc_Sigma_R(T,omega,Electrode,derivative=0,extension_length=None,nyquist_damping=3): #new one with aux modes
    #derivative calculates the derivative of the specified order w.r.t omega
    #nyquist_damping: the damping of the Gaussian filter at the nyquist frequency given in standard deviations. 
    #default corresponds to 3 standard deviations.
    start_time = time.time()
    Gamma = Electrode.Gamma
    potential = Electrode.potential
    T = np.array(T).reshape(-1,1,1,1)
    omega = np.array(omega).reshape(1,-1,1,1)
    if Electrode.WBL_bool: #we are in the wide-band limit, and so Sigma_R is constant. Return the constant value.
        #Gamma_mat = Gamma
        if derivative == 0:
            return -1j*Gamma(0)/2 * np.ones((omega*T).shape)
        else: 
            shape = (Gamma(0) * np.ones((omega*T).shape)).shape
            return np.zeros(shape)

    if potential.potential_is_constant_everywhere:
            if derivative == 0:
                Sigma0 = -1j/2*AE_math.hilbert(Gamma(omega-potential.function),axis=1)
                return Sigma0
            else:
                Sigma0 = lambda w : Gamma(w-potential.function)
                Sigma0_dw = AE_math.FD(Sigma0,omega,n=derivative)   
                Sigma0_dw = -1j/2*AE_math.hilbert(Sigma0_dw,axis=1)
                """
                N = omega.size
                domain_length = omega.max() - omega.min()
                tau_max = 2*np.pi * (N-1)/domain_length
                tau = np.linspace(0,tau_max,N)
                tau = tau.reshape(omega.shape)
                Sigma0_fft = (N-1)/(N)*AE_math.fft(Sigma0,axis=1)
                Nyquist_freq = tau.max()/2
                nyquist_damping =5    
                Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2)
                Sigma0_fft = Sigma0_fft * Nyquist_filter
                Sigma0_fft[0] /=2
                Sigma0_dw = AE_math.ifft((1j*tau)**derivative * Sigma0_fft,axis=1)
                """
                
                return Sigma0_dw 
                

    if not potential.support_is_finite():
        print('Support of specified potential is infinite. A different method is necessary for calculating Sigma_R!')
        assert 1==0


    #T=np.array(T).reshape(-1,1)
    #omega=np.array(omega).reshape(1,-1)   
    exp = np.exp
    omega_min = omega.min()
    omega_max = omega.max()
    dw = (omega_max - omega_min)/(np.size(omega)-1)
    if extension_length is None:
        extension_length = omega.size * 3
    #extend internal arrays by extension_length. this is useful if the specified range of integration does not include the entire support of the given functions.
    extension = dw*np.linspace(1,extension_length,extension_length)
    w_extended = np.concatenate(((omega_min - np.flip(extension)),omega.flatten(),omega_max + extension))
    w_extended=w_extended.reshape(1,-1)
    #print('w_extended[extension_length] == omega_min? w_extended[ext_len]==', w_extended[extension_length],'omega_min ==',omega_min)
    #dw_taumax = 2*np.(Tmax - Tmin)
    #Calculate range of integration needed for sufficiently fine tau sampling (dt = 2pi/(b-a)) and nyquist: 1/dt > 2*max freq
    V_min, V_max = potential.Range
    support = potential.support
    if V_max is None or V_min is None:
        print('Warning: Range of electrode potential should be specified to ensure correct frequency integration')
        test_pot = potential(np.linspace(support[0],support[1],200))
        V_max = test_pot.max()
        V_max += 0.1*abs(V_max)
        V_min = test_pot.min()
        V_min -= 0.1*abs(V_min)
        print('V_min and V_max set to ', V_min, ' and ', V_max,'.')


    if V_max == V_min: #potential is constant everywhere
        G = Gamma(omega - T*V_max)
        print('Potential constant - are you sure you want a time-dependent calculation?')
        return -1j/2 *AE_math.hilbert(G,axis=1)




    max_freq = V_max - V_min
    dtau_max = 1/(2.1*max_freq) #ensure that the tau sampling occurs above the Nyquist frequency of the "filter", exp(-\infint \Delta_\alpha ...)
    min_domain_length = 2*np.pi/dtau_max
    domain_length = max(w_extended.max()-w_extended.min(), min_domain_length) #We should at least include the entire range [omega_min, omega_max].
    N1 = np.ceil(domain_length/dw) 
    #tau_max_for_entire_domain = 2*(Tmax - Tmin)
    #N2 = np.ceil(tau_max_for_entire_domain/(2*np.pi)*domain_length + 1)
    #N3=max(N1,N2)
    N = int(2**np.ceil(np.log(N1)/np.log(2))) #Make N a power of two for most efficient FFT. do so in a way that cannot decrease the value of N.
    #if N>1e5:
    #    print("N in calc_Sigma_R larger than 100k!")
    #x = domain_length/N * np.arange(N) #generate x variable for FFT.
    w_fft_var = dw*np.arange(N)
    domain_length= w_fft_var.max() * N/(N-1)
    w_fft_var = w_fft_var.reshape(1,-1) #shape to proper size
    
    dtau = 2*np.pi/domain_length
    tau = dtau*np.arange(N) #generate tau variable for FFT
    tau = tau.reshape(1,-1) #reshape for use in expint.
    
    Nyquist_freq = tau.max()/2
    
    Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2) #filter that reduces the amplitude at the Nyquist freq by 3 standard deviations of the normal distribution
    #np.save('Nyquist_*filter',Nyquist_filter)
    #np.save('tau',tau)
    #calculate expint
    if potential.antiderivative is None:
        expint = np.exp(-1j*integrate_potential(T,tau,potential))
    else:
        expint = np.exp(-1j*(potential.antiderivative(T+tau/2) - potential.antiderivative(T-tau/2))).reshape(np.size(T),np.size(tau))

    Sigma_R = np.zeros((np.size(T),np.size(omega),Electrode.basis_size,Electrode.basis_size),dtype=np.complex128)
    #!TODO : sparse matrices
    omega_index = slice(extension_length,(np.size(omega)+extension_length))
    if not Electrode.use_aux_modes:
        Gam = Gamma(w_fft_var.reshape(1,-1,1,1)+w_extended.min())
    coupling_index = np.ones((Electrode.basis_size,Electrode.basis_size))
    for i in range(Electrode.basis_size):
        for j in range(Electrode.basis_size):
            if coupling_index[i,j]: #each iteration calculates one matrix element of Sigma. Variables in the loop do not carry matrix indices.
                if Electrode.use_aux_modes: #potential must be of the meromorphic class
                    #Lorentz_int_residues = lambda E : exp(-1j*(E)*tau)#f(E)*exp(-1j*(E)*taup)#
                    #Lor_poles, Lor_res = Lorentz_params
                    #Lorentz_params_ij = [Lor_poles,Lor_res[:,i,j]]
                    #Lorentz_int = self.int_residues2(Lorentz_int_residues,Lorentz_params_ij,halfplane='lower')
                    #forward_trans = Lorentz_int/(2*np.pi)
                    exponential = lambda E : exp(-1j*E*tau)
                    forward_trans = potential.integrate(f=exponential)
                else:
                    #forward transform (energy integral)
                    fxm = domain_length/(2*np.pi*N)*Gam[:,:,i,j]
                    forward_trans = AE_math.fft(fxm,axis=1) * exp(-1j*w_extended.min()*tau)
                
                forward_trans = forward_trans * Nyquist_filter
                if derivative > 0:
                    forward_trans *= (1j*tau)**derivative
                forward_trans = forward_trans * expint
                forward_trans[:,0] /= 2
                back_trans = tau.max() * AE_math.ifft(exp(1j*w_extended.min()*tau)*forward_trans,axis=1)

                Sigma_R[:,:,i,j] =   -1j*back_trans[:,omega_index]
    end_time = time.time()
    #print('calc_Sigma_R calculated in %.2f seconds'%(end_time-start_time))
    return Sigma_R


def calc_Sigma_R(T,omega,Electrode,derivative=0,extension_length=4,nyquist_damping=5): #new one with aux modes

    potential = Electrode.potential
    T = np.array(T).reshape(-1,1,1,1)
    omega = np.array(omega).reshape(1,-1,1,1)
    if omega.size > 1:
        dw = min((omega.max() - omega.min())/(omega.size - 1),0.01)
    else:
        dw = 0.01

    if Electrode.WBL_bool: #we are in the wide-band limit, and so Sigma_R is constant. Return the constant value.
        if hasattr(derivative,'__iter__'):
            Sig_list = []
            for d in derivative:
                if d == 0:
                    Sig_list.append(-1j*Electrode.Gamma(0)/2 * np.ones((omega*T).shape))
                else: 
                    shape = (Electrode.Gamma(0) * np.ones((omega*T).shape)).shape
                    Sig_list.append(np.zeros(shape))
            return Sig_list
        
        if derivative == 0:
            return -1j*Electrode.Gamma(0)/2 * np.ones((omega*T).shape)
        else: 
            shape = (Electrode.Gamma(0) * np.ones((omega*T).shape)).shape
            return np.zeros(shape)

    if potential.potential_is_constant_everywhere:
        if hasattr(derivative,'__iter__'):
            Sig_list = []
            for d in derivative:
                if d == 0:
                    Sigma0 = -1j/2*AE_math.hilbert(Electrode.Gamma(omega-potential.function),axis=1)
                    Sig_list.append(Sigma0)
                else:
                    Sigma0 = lambda w : Electrode.Gamma(w-potential.function)
                    Sigma0_dw = AE_math.FD(Sigma0,omega,n=d)   
                    Sigma0_dw = -1j/2*AE_math.hilbert(Sigma0_dw,axis=1)
                    Sig_list.append(Sigma0_dw )
            return Sig_list
        
        if derivative == 0:
            Sigma0 = -1j/2*AE_math.hilbert(Electrode.Gamma(omega-potential.function),axis=1)
            return Sigma0
        else:
            Sigma0 = lambda w : Electrode.Gamma(w-potential.function)
            Sigma0_dw = AE_math.FD(Sigma0,omega,n=derivative)   
            Sigma0_dw = -1j/2*AE_math.hilbert(Sigma0_dw,axis=1)
            return Sigma0_dw 

    Gamma = Electrode.getGammaAsArray(dw=dw)
    a,b = Electrode.bandwidth

    if not potential.support_is_finite():
        print('Support of specified potential is infinite. A different method is necessary for calculating Sigma_R!')
        assert 1==0

   
    V_min, V_max = potential.Range
    support = potential.support
    if V_max is None or V_min is None:
        print('Warning: Range of electrode potential should be specified to ensure correct frequency integration')
        test_pot = potential(np.linspace(support[0],support[1],200))
        V_max = test_pot.max()
        V_max += 0.1*abs(V_max)
        V_min = test_pot.min()
        V_min -= 0.1*abs(V_min)
        print('V_min and V_max set to ', V_min, ' and ', V_max,'.')

    max_freq = V_max - V_min
    dtau_max = 1/(2.1*max_freq) #ensure that the tau sampling occurs above the Nyquist frequency of the "filter", exp(-\infint \Delta_\alpha ...)
    min_domain_length = max(Gamma.shape[1]*(1+extension_length), 2*np.pi/dtau_max/dw)
    N = int(2**np.ceil(np.log(min_domain_length)/np.log(2))) #Make N a power of two for most efficient FFT. do so in a way that cannot decrease the value of N.
    print('number of fourier components: ', N)
    tau = np.fft.fftfreq(N,d=dw).reshape(1,-1,1,1) * 2 * np.pi
    Nyquist_freq = tau.max()/2
    
    Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2) 

    if potential.antiderivative is None:
        expint = np.exp(-1j*integrate_potential(T,tau,potential))
    else:
        expint = np.exp(-1j*(potential.antiderivative(T+tau/2) - potential.antiderivative(T-tau/2)))

    G1 = expint*Nyquist_filter * np.fft.fft(Gamma,axis=1,n=N)
    G1[:,0] /= 2
    G1 = G1 *(tau >= 0)

    if hasattr(derivative,'__iter__'):
        Sig_list = []
        for d in derivative:
            G = G1 * (1j*tau)**d
            Sig_list.append(1j* np.fft.ifft(G*np.exp(-1j*tau*(a - omega.min())),axis=1).real[:,:omega.size])
            return Sig_list
    if derivative > 0:
        G1 = G1 * (1j*tau)**derivative
    Sigma_R = -1j* np.fft.ifft(G1*np.exp(-1j*tau*(a - omega.min())),axis=1)
    #print('calc_Sigma_R calculated in %.2f seconds'%(end_time-start_time))
    return Sigma_R[:,:omega.size]

def calc_Sigma_less(T,omega,Electrode,derivative=0,extension_length=4,nyquist_damping=5): #new one with aux modes
    
    potential = Electrode.potential
    T = np.array(T).reshape(-1,1,1,1)
    omega = np.array(omega).reshape(1,-1,1,1)
    if omega.size > 1:
        dw = min((omega.max() - omega.min())/(omega.size - 1),0.01)
    else:
        dw = 0.01
    if potential.potential_is_constant_everywhere:
        if hasattr(derivative,'__iter__'):
            Sig_list = []
            for d in derivative:
                if d == 0:
                    Sigma0 = 1j*Electrode.Gamma(omega-potential.function)*Electrode.fermi(omega - potential.function,T)
                    Sig_list.append(Sigma0)
                else:
                    Sigma0 = lambda w : 1j*Electrode.Gamma(w-potential.function)*Electrode.fermi(w - potential.function,T)
                    Sigma0_dw = AE_math.FD(Sigma0,omega,n=d)
                    Sig_list.append(Sigma0_dw)
            return Sig_list
        if derivative == 0:
            Sigma0 = 1j*Electrode.Gamma(omega-potential.function)*Electrode.fermi(omega - potential.function,T)
            return Sigma0
        else:
            Sigma0 = lambda w : 1j*Electrode.Gamma(w-potential.function)*Electrode.fermi(w - potential.function,T)
            Sigma0_dw = AE_math.FD(Sigma0,omega,n=derivative)
            return Sigma0_dw 
    if not potential.support_is_finite():
        print('Support of specified potential is infinite. A different method is necessary for calculating Sigma_R!')
        assert 1==0

    Gamma = Electrode.getGammaFermiAsArray(dw=dw,T=T)
    a,b = Electrode.bandwidth
    exp = np.exp

    V_min, V_max = potential.Range
    support = potential.support
    if V_max is None or V_min is None:
        print('Warning: Range of electrode potential should be specified to ensure correct frequency integration')
        test_pot = potential(np.linspace(support[0],support[1],200))
        V_max = test_pot.max()
        V_max += 0.1*abs(V_max)
        V_min = test_pot.min()
        V_min -= 0.1*abs(V_min)
        print('V_min and V_max set to ', V_min, ' and ', V_max,'.')

    max_freq = V_max - V_min
    dtau_max = 1/(2.1*max_freq) #ensure that the tau sampling occurs above the Nyquist frequency of the "filter", exp(-\infint \Delta_\alpha ...)
    min_domain_length = max(Gamma.shape[1]*(1+extension_length), 2*np.pi/dtau_max/dw)
    N = int(2**np.ceil(np.log(min_domain_length)/np.log(2))) #Make N a power of two for most efficient FFT. do so in a way that cannot decrease the value of N.
    
    tau = np.fft.fftfreq(N,d=dw).reshape(1,-1,1,1) * 2 * np.pi
    Nyquist_freq = tau.max()/2
    
    Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2) 

    if potential.antiderivative is None:
        expint = np.exp(-1j*integrate_potential(T,tau,potential))
    else:
        expint = np.exp(-1j*(potential.antiderivative(T+tau/2) - potential.antiderivative(T-tau/2)))

    G1 = expint*Nyquist_filter * np.fft.fft(Gamma,axis=1,n=N)

    if hasattr(derivative,'__iter__'):
        Sig_list = []
        for d in derivative:
            G = G1 * (1j*tau)**d
            Sig_list.append(1j* np.fft.ifft(G*np.exp(-1j*tau*(a - omega.min())),axis=1).real[:,:omega.size])
        return Sig_list

    if derivative > 0:
        G1 = G1 * (1j*tau)**derivative
    Sigma_less = 1j* np.fft.ifft(G1*np.exp(-1j*tau*(a - omega.min())),axis=1).real
    #print('calc_Sigma_R calculated in %.2f seconds'%(end_time-start_time))
    return Sigma_less[:,:omega.size]



def old_calc_Sigma_less(T,omega,Electrode,derivative=0,extension_length=None,nyquist_damping=3): #new one with aux modes
    #derivative calculates the derivative of the specified order w.r.t omega
    #nyquist_damping: the damping of the Gaussian filter at the nyquist frequency given in standard deviations. 
    #default corresponds to 3 standard deviations.
    start_time = time.time()
    Gamma = Electrode.Gamma
    potential = Electrode.potential
    T = np.array(T).reshape(-1,1,1,1)
    omega = np.array(omega).reshape(1,-1,1,1)
    if potential.potential_is_constant_everywhere:
        if derivative == 0:
            Sigma0 = 1j*Gamma(omega-potential.function)*Electrode.fermi(omega - potential.function)
            return Sigma0
        else:
            Sigma0 = lambda w : 1j*Gamma(w-potential.function)*Electrode.fermi(w - potential.function)
            """
            N = omega.size
            domain_length = omega.max() - omega.min()
            tau_max = 2*np.pi * (N-1)/domain_length
            tau = np.linspace(0,tau_max,N)
            tau = tau.reshape(omega.shape)
            Sigma0_fft = (N-1)/(N)*AE_math.fft(Sigma0,axis=1)
            Nyquist_freq = tau.max()/2
            nyquist_damping =5    
            Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2)
            Sigma0_fft = Sigma0_fft * Nyquist_filter
            Sigma0_fft[0] /=2
            Sigma0_dw = AE_math.ifft((1j*tau)**derivative * Sigma0_fft,axis=1)
            """
            Sigma0_dw = AE_math.FD(Sigma0,omega,n=derivative)
            return Sigma0_dw 
    if not potential.support_is_finite():
        print('Support of specified potential is infinite. A different method is necessary for calculating Sigma_R!')
        assert 1==0

    exp = np.exp
    omega_min = omega.min()
    omega_max = omega.max()
    dw = (omega_max - omega_min)/(np.size(omega)-1)
    if extension_length is None:
        extension_length = omega.size * 3
    #extend internal arrays by extension_length. this is useful if the specified range of integration does not include the entire support of the given functions.
    extension = dw*np.linspace(1,extension_length,extension_length)
    w_extended = np.concatenate(((omega_min - np.flip(extension)),omega.flatten(),omega_max + extension))
    w_extended=w_extended.reshape(1,-1)
    #print('w_extended[extension_length] == omega_min? w_extended[ext_len]==', w_extended[extension_length],'omega_min ==',omega_min)
    #dw_taumax = 2*np.(Tmax - Tmin)
    #Calculate range of integration needed for sufficiently fine tau sampling (dt = 2pi/(b-a)) and nyquist: 1/dt > 2*max freq
    V_min, V_max = potential.Range
    support = potential.support
    if V_max is None or V_min is None:
        print('Warning: Range of electrode potential should be specified to ensure correct frequency integration')
        test_pot = potential(np.linspace(support[0],support[1],200))
        V_max = test_pot.max()
        V_max += 0.1*abs(V_max)
        V_min = test_pot.min()
        V_min -= 0.1*abs(V_min)
        print('V_min and V_max set to ', V_min, ' and ', V_max,'.')


    if V_max == V_min: #potential is constant everywhere
        Sigma = Gamma(omega - T*V_max)*Electrode.fermi(omega - T*V_max)
        print('Potential constant - are you sure you want a time-dependent calculation?')
        return 1j*Sigma


    max_freq = V_max - V_min
    dtau_max = 1/(2.1*max_freq) #ensure that the tau sampling occurs above the Nyquist frequency of the "filter", exp(-\infint \Delta_\alpha ...)
    min_domain_length = 2*np.pi/dtau_max
    domain_length = max(w_extended.max()-w_extended.min(), min_domain_length) #We should at least include the entire range [omega_min, omega_max].
    N1 = np.ceil(domain_length/dw) 
    #tau_max_for_entire_domain = 2*(Tmax - Tmin)
    #N2 = np.ceil(tau_max_for_entire_domain/(2*np.pi)*domain_length + 1)
    #N3=max(N1,N2)
    N = int(2**np.ceil(np.log(N1)/np.log(2))) #Make N a power of two for most efficient FFT. do so in a way that cannot decrease the value of N.
    #if N>1e5:
    #    print("N in calc_Sigma_R larger than 100k!")
    #x = domain_length/N * np.arange(N) #generate x variable for FFT.
    w_fft_var = dw*np.arange(N)
    domain_length= w_fft_var.max() * N/(N-1)
    w_fft_var = w_fft_var.reshape(1,-1) #shape to proper size
    
    dtau = 2*np.pi/domain_length
    tau = dtau*np.arange(N) #generate tau variable for FFT
    tau = tau.reshape(1,-1) #reshape for use in expint.
    print('internal fourier array size:',tau.size)
    Nyquist_freq = tau.max()/2
    
    Nyquist_filter = np.exp(-(nyquist_damping*tau/Nyquist_freq)**2/2) #filter that reduces the amplitude at the Nyquist freq by 3 standard deviations of the normal distribution
    #np.save('Nyquist_*filter',Nyquist_filter)
    #np.save('tau',tau)
    #calculate expint
    if potential.antiderivative is None:
        expint = np.exp(-1j*integrate_potential(T,tau,potential))
    else:
        expint = np.exp(-1j*(potential.antiderivative(T+tau/2) - potential.antiderivative(T-tau/2))).reshape(np.size(T),np.size(tau))

    Sigma_less = np.zeros((np.size(T),np.size(omega),Electrode.basis_size,Electrode.basis_size),dtype=np.complex128)
    #!TODO : sparse matrices
    omega_index = slice(extension_length,(np.size(omega)+extension_length))
    if not Electrode.use_aux_modes:
        Gam = Gamma(w_fft_var.reshape(1,-1,1,1)+w_extended.min())
        fermi = Electrode.fermi(w_fft_var+w_extended.min())
    coupling_index = np.ones((Electrode.basis_size,Electrode.basis_size))
    #coupling_index = None
    if coupling_index is None: #no coupling index; calculate entire matrices at once for faster execution (at the cost of more memory)
        if Electrode.use_aux_modes:
            print('error: aux modes not implemented!')
            assert 1==0
        else:
            fermi_shape = list(fermi.shape)
            fermi_shape.append(1)
            fermi_shape.append(1)
            fermi = fermi.reshape(fermi_shape)

            #forward transform (energy integral)
            fxm = domain_length/(2*np.pi*N)*Gam*fermi        
            forward_trans = AE_math.fft(fxm,axis=1) * exp(-1j*w_extended.min()*tau.reshape(1,-1,1,1))
            
        forward_trans = forward_trans * Nyquist_filter.reshape(1,-1,1,1)
        if derivative > 0:
            forward_trans *= (1j*tau.reshape(1,-1,1,1))**derivative
        expint_shape = list(expint.shape)
        expint_shape.append(1)
        expint_shape.append(1)
        expint = expint.reshape(expint_shape)

        forward_trans = forward_trans * expint
        forward_trans[:,0] /= 2
        back_trans = tau.max() * AE_math.ifft(exp(1j*w_extended.min()*tau.reshape(1,-1,1,1))*forward_trans,axis=1)

        Sigma_less = 2j*np.real(back_trans[:,omega_index])




    elif coupling_index is not None:
        for i in range(Electrode.basis_size):
            for j in range(Electrode.basis_size):
                if coupling_index[i,j]: #each iteration calculates one matrix element of Sigma. Variables in the loop do not carry matrix indices.
                    if Electrode.use_aux_modes: #potential must be of the meromorphic class
                        #Lorentz_int_residues = lambda E : exp(-1j*(E)*tau)#f(E)*exp(-1j*(E)*taup)#
                        #Lor_poles, Lor_res = Lorentz_params
                        #Lorentz_params_ij = [Lor_poles,Lor_res[:,i,j]]
                        #Lorentz_int = self.int_residues2(Lorentz_int_residues,Lorentz_params_ij,halfplane='lower')
                        #forward_trans = Lorentz_int/(2*np.pi)
                        exponential = lambda E : exp(-1j*E*tau)
                        forward_trans = potential.integrate(f=exponential)
                    else:
                        #forward transform (energy integral)
                        fxm = domain_length/(2*np.pi*N)*Gam[:,:,i,j]*fermi        
                        forward_trans = AE_math.fft(fxm,axis=1) * exp(-1j*w_extended.min()*tau)
                        
                    forward_trans = forward_trans * Nyquist_filter
                    if derivative > 0:
                        forward_trans *= (1j*tau)**derivative
                    forward_trans = forward_trans * expint
                    forward_trans[:,0] /= 2
                    back_trans = tau.max() * AE_math.ifft(exp(1j*w_extended.min()*tau)*forward_trans,axis=1)

                    Sigma_less[:,:,i,j] =   2j*np.real(back_trans[:,omega_index])

    end_time = time.time()
    #print('calc_Sigma_less calculated in %.2f seconds'%(end_time-start_time))
    return Sigma_less









def integrate_potential(T,tau,potential): 
    #Integrates potential numerically in the range T-tau/2 .. T+tau/2.
    start_time = time.time()
    #The tau values must be evenly spaced and start at 0.
    #The function takes advantage of the structure of the integral by computing each new integration as the sum of the previous integration plus a small extra part

     #
        #Suppose the support of the pulse is contained in the interval [Tmin, Tmax].
        #t0=time.time()

    n_int=5
    x,w = AE_math.get_integration_var(n_int)
    if potential.support_is_infinite():
        print('error in expint: support of potential must be finite for this integration to be possible!')
        assert 1==0
    a,b = potential.support


    def int_gauss(f,tmin,tmax,N=n_int):
        t = tmin + x*(tmax-tmin)
        q = w*(tmax-tmin)
        res=0
        for i in range(N):
            res += f(t[i])*q[i]
            #res f(t[i])*q[i]
        return res
    def antiderivative(t,t0=None): #integrates the function pot and gives the antiderivative evaluated as F(t) - F(t0)
        F=[]
        Fcum=0
        if t0 is None:
            t0 = t[0]
        for tt in t:
            Fcum += int_gauss(potential,t0,tt)
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
            if tlist1[0] != tlist2[0]: #t0 is not the same in the two derivatives, leading to constant offset. this should fix
                t01 = tlist1[0]
                t02 = tlist2[0]
                t0_correction = quad(potential,t01,t02)[0] #integrate with scipy
                F1 = F1 - t0_correction
            F[i]=F1-F2
    end_time = time.time()
    print('integrate_potential calculated in %.2f seconds'%(end_time-start_time))
    return F


def run_parallel(n_jobs,T,Device,Electrode_L,Electrode_R,omega=None,omega_weights=None,side='left',order=2,eta=None,nyquist_damping=5,calc_density=False,save_arrays='none',name=''):
    #syntax for calc_current: (T,Device,Electrode_L,Electrode_R,order=2,eta=None,omega=None,nyquist_damping=5,calc_density=False,side='both',save_arrays='none',name=None):
    start_time = time.time()
    N_times_per_run = int(T.size / n_jobs)
    remainder_times = T.size % n_jobs
    T_list = []
    idx = 0

    #divide list of times into n_jobs separate lists
    for i in range(n_jobs):
        if i < remainder_times:
            T_list.append(T[idx:idx + N_times_per_run + 1])
            idx +=N_times_per_run + 1
        else:
            T_list.append(T[idx:idx + N_times_per_run])
            idx +=N_times_per_run


    #perform parallel computation
    global _global_func_randomid_FAWDAUYGWHIUANP
    def _global_func_randomid_FAWDAUYGWHIUANP(T):
        res = calc_current(T,Device,Electrode_L,Electrode_R,order,eta,omega,nyquist_damping,calc_density,side,save_arrays,name)
        return res
    res = jl.Parallel(n_jobs=n_jobs,backend='multiprocessing')(jl.delayed(_global_func_randomid_FAWDAUYGWHIUANP)(Ti) for Ti in T_list)
    del _global_func_randomid_FAWDAUYGWHIUANP

    #merge results into single dict
    out = {}
    res0 = res[0]
    for key in res0.keys():
        val = res0[key]
        for dd in res[1:]:
            val = np.concatenate((val,dd[key])) 
        out[key] = val

    return out





    # calc_current(T,Device,Electrode_L,Electrode_R,omega=omega,omega_weights=omega_weights,side=side,order=order,eta=eta,nyquist_damping=nyquist_damping,calc_density=calc_density,save_arrays=save_arrays,name=name):










"""

f = Potential(np.sin,support=[-2*np.pi,2*np.pi])
T = np.array([0.5,1])
tau = np.linspace(0,2*np.pi,50)
F = expint(T,tau,f)
import matplotlib.pyplot as plt
plt.plot(tau,-np.cos(0.5+tau/2) + np.cos(0.5-tau/2))
plt.plot(tau,-np.cos(1+tau/2) + np.cos(1-tau/2))
plt.plot(tau,F[0,:],ls='dashed')
plt.plot(tau,F[1,:],ls='dashed')
plt.show()
"""




"""
def Gamma(w):
    w = np.array(w)
    w_idx = (w < 2) * (w > -2)
    G = np.zeros(w.shape,dtype=np.complex128)
    G[w_idx] = np.sqrt(1-(w[w_idx]/2)**2)
    return G
def Gamma_L(w):
    return np.array([[1,0],[0,0]])*Gamma(w)
def Gamma_R(w):
    return np.array([[0,0],[0,1]])*Gamma(w)
pot_L = Potential(lambda t : np.sin(t/10)/5,support=[-np.pi*10,np.pi*10],Range=[-0.2,0.2])
pot_R = Potential(lambda t : -np.sin(t/10)/5,support=[-np.pi*10,np.pi*10],Range=[-0.2,0.2])
Elec_L = Electrode(Gamma_L,potential=pot_L,kT=0.1)
Elec_R = Electrode(Gamma_R,potential=pot_R,kT=0.1)
omega = np.linspace(-6,6,1001).reshape(1,-1,1,1)
T=np.linspace(-15*np.pi,15*np.pi,50).reshape(-1,1,1,1)
#sig_r = calc_Sigma_R(T,omega,Elec_L)
#sig_less = calc_Sigma_less(T,omega,Elec_L)
dw = (omega.max() - omega.min())/(np.size(omega) - 1)
omega_weights = dw * np.ones(omega.shape)
H0 = np.array([[1,1/2],[1/2,-1]])
dev = Device(H0)
J0, J1 = calc_current(T,Device=dev,Electrode_L=Elec_L,Electrode_R=Elec_R,omega=omega,omega_weights=omega_weights,
    eta = 5e-3)

w = omega.flatten()
t = T.flatten()
plt.plot(t,J0)
plt.plot(t,J0+J1)
plt.grid()
plt.show()



"""
