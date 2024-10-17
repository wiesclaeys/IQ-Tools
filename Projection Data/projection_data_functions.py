# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:36:47 2024

@author: wclaey6
"""

import numpy as np
import scipy as sp

def theoretical_profile(x, x0, C, mu, R):
    
    a = np.where((x-x0)**2 < R**2, C / mu * (1-np.exp(-2*mu*np.sqrt(R**2-(x-x0)**2))), 0)
    a = np.array(a, dtype = np.float32)
    # print(len(a))
    # print(a)

    return a

# def theoretical_profile2(x, x0, C, mu, R):
#     a = np.where((x-x0)**2 < R**2, C / mu * (1-np.exp(-2*mu*np.sqrt(R**2-(x-x0)**2))), 0)
#     a = np.array(a, dtype = np.float32)
#     # print(len(a))
#     print(a)

    return a.flatten()

def poisson(x, mu):
    return mu**x*np.exp(-mu)/sp.special.factorial(x)

def rebin_data(freqs, expected_freqs):
    """
    Rebins the data to make sure the expected number of events is large enough to reliably perform a chi^2 test
    
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

    """
    #initialize
    Es = [] # expected frequencies
    Os = [] # observed frequencies
    E = 0 
    O = 0
    
    # run over data
    for i in range(len(freqs)):
        E += expected_freqs[i]
        O += freqs[i]
        
        if E >= 5: # only add data to category if the expected number of events is large enough (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
            Es.append(E)
            Os.append(O)
            
            O = 0
            E = 0
            
    # add remaining data to last category
    Es[-1] = Es[-1] + E + expected_freqs[-1]
    Os[-1] = Os[-1] + O
    
    return Os, Es