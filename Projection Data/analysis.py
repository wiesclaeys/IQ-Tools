# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:58:57 2024

@author: wclaey6
"""

import os
import math

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import pydicom as pd

from IQ_functions import *

def theoretical_profile(x, x0, C, mu, R):
    
    a = np.where((x-x0)**2 < R**2, C / mu * (1-np.exp(-2*mu*np.sqrt(R**2-(x-x0)**2))), 0)
    a = np.array(a, dtype = np.float32)
    # print(len(a))
    # print(a)

    return a

def theoretical_profile2(x, x0, C, mu, R):
    a = np.where((x-x0)**2 < R**2, C / mu * (1-np.exp(-2*mu*np.sqrt(R**2-(x-x0)**2))), 0)
    a = np.array(a, dtype = np.float32)
    # print(len(a))
    print(a)

    return a.flatten()

def poisson(x, mu):
    return mu**x*np.exp(-mu)/sp.special.factorial(x)

# =============================================================================
# INPUTS AND DATA READING
# =============================================================================
# Reading the data and post-smoothing
# =============================================================================

path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\'
name = '20'



# # Other parameters (more parameters can be set in each block of code)
# units = "suv"         # units of the image data 

# sm_fwhm_mm    = 0       # width of the Gaussian kernel used for post smoothing
# radius        = 100     # known radius of the cylindrical phantom

scale_factor = 1
# =============================================================================

# # reading data


dcm_dir     = path + name
# dcm         = pmf.DicomVolume(os.path.join(dcm_dir, '*')) # join multiple slices to a single volume
# vol         = dcm.get_data() 
# vol         = vol * scale_factor

dcm_dir     = path + name
dcm         = pd.dcmread(dcm_dir + "\\" + os.listdir(dcm_dir)[0])


vol         = dcm.pixel_array 
vol         = vol * scale_factor

pixel_size = dcm.PixelSpacing[0]
num_rows = dcm.Rows
num_columns = dcm.Columns

PP = vol[0:120]
LS = vol[120:240]

plt.imshow(PP[1,:,:])
plt.colorbar()
plt.show()

x_profiles = np.sum(PP, axis = 1)
y_profiles = np.sum(PP, axis = 2)
xs = np.arange(256)

for i in range(1):
    plt.plot(x_profiles[i,:])
plt.show()

for i in range(1):
    plt.plot(y_profiles[i,:])
plt.show()

# plt.hist(PP[1,:,:], bins = 35)
# plt.show()

plt.plot(xs, theoretical_profile2(xs, 128, 5, 0.01, 100))
plt.show()



#%% find center of mass for segmentation

RR, CC = np.indices(np.shape(PP[0,:,:]),)

columns = np.arange(num_columns)
rows = np.arange(num_rows)

xs = (columns - (num_columns - 1) / 2) * pixel_size  
ys = (rows - (num_rows - 1) / 2) * pixel_size  

xx, yy = np.meshgrid(xs, ys, indexing = 'xy')

#frames without bed: first 25 are safe





x_centers = np.sum(PP * xx, axis = (1,2)) / np.sum(PP, axis = (1,2))
y_centers = np.sum(PP * yy, axis = (1,2)) / np.sum(PP, axis = (1,2))

plt.plot(x_centers, label = 'x')
plt.plot(y_centers, label = 'y')
plt.legend()
plt.show()

#%% ROI




# for i in range(25):
#     ROI = PP[i,120:132,120:132]
    
#     x_prof = np.sum(ROI, axis = 1)
#     y_prof = np.sum(ROI, axis = 0)
    
#     # plt.plot(x_prof, label = 'x')
#     # plt.plot(y_prof, label = 'y')
#     # plt.legend()
#     # plt.show()
    
#     # plt.imshow(ROI)
#     # plt.show()
#     plt.hist(ROI.flatten(), bins = np.arange(35))
#     plt.plot(np.arange(35), np.size(ROI)*poisson(np.arange(35), mu = np.mean(ROI)))
#     plt.show()
    
    
#     print("mean =", np.mean(ROI))
#     print("var = ", np.var(ROI, ddof = 1))


#%% spline fitting to y profile
s = 200000

# tck = sp.interpolate.bisplrep(xx[100:200,100:200], yy[100:200,100:200], PP[0,100:200,100:200])

tck = sp.interpolate.splrep(ys, y_profiles[0], s = s)
num_knots = len(tck[0])
print("number of knots = ", num_knots)

spline = sp.interpolate.splev(ys, tck)

plt.plot(ys, y_profiles[0], label = 'data')
plt.plot(ys, spline, label = 'spline')
plt.legend()
plt.xlim(-150,150)
plt.show()

#%% spline fitting to x profile

s = 70000
k = 3

tck = sp.interpolate.splrep(xs, x_profiles[0], s = s, k = k)
num_knots = len(tck[0])
print("number of knots = ", num_knots)

spline = sp.interpolate.splev(xs, tck)

plt.plot(xs, x_profiles[0], label = 'data')
plt.plot(xs, spline, label = 'spline')
plt.legend()
plt.xlim(-150,150)
plt.show()


#%%





#%%
# for i in range(60):
#     plt.imshow(PP[i,75:175, 75:175], vmax = 35, vmin = 0)
#     plt.colorbar()
#     plt.scatter(x_centers[i]-75, y_centers[i]-75, color = 'red')
    
#     plt.title('frame' + str(i))
#     plt.show()


#%%
# a = theoretical_profile2(xs, 128,5,0.01,100)

# params, rest = sp.optimize.curve_fit(theoretical_profile2, xs, x_profiles, p0 = [128,5,0.01,100])


