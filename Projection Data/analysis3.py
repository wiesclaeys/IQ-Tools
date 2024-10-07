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
# Process the full count data first
# =============================================================================
# 
# =============================================================================

frame_index = 121

path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\PP\\'
path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_Tb\\'
name = '2024-07-03'

dcm_dir     = path + name
dcm         = pd.dcmread(dcm_dir + "\\" + os.listdir(dcm_dir)[0])
vol         = dcm.pixel_array 

if name == '100':
    rescale = 1
else:
    field = dcm[0x0040,0x9096][0]
    rescale = field[0x0040,0x9225].value
vol = vol * rescale


frame       = vol[frame_index, :, :]

pixel_size = dcm.PixelSpacing[0]
num_rows = dcm.Rows
num_columns = dcm.Columns



#%% get coordinates etc

RR, CC = np.indices(np.shape(frame[:,:]),)

columns = np.arange(num_columns)
rows = np.arange(num_rows)

xs = (columns - (num_columns - 1) / 2) * pixel_size  
ys = (rows - (num_rows - 1) / 2) * pixel_size  

xx, yy = np.meshgrid(xs, ys, indexing = 'xy')

#%% Show the selected frame

plt.imshow(frame)
plt.colorbar()
plt.title("Frame " + str(frame_index + 1))
plt.show()

#%% Plot x and y profiles (full FOV)


x_profile_full = np.sum(frame, axis = 0)
y_profile_full = np.sum(frame, axis = 1)

plt.plot(xs, x_profile_full)
plt.title('x profile (full FOV)')
plt.xlabel('position (mm)')
plt.ylabel('counts')
plt.ylim(0, 1.05 * np.max(x_profile_full))
plt.show()

plt.plot(ys, y_profile_full)
plt.title('y profile (full FOV)')
plt.xlabel('position (mm)')
plt.ylabel('counts')
plt.ylim(0, 1.05 * np.max(y_profile_full))
plt.show()


#%% segmentation in y direction

start = 100
stop = 150


plt.plot(columns, y_profile_full)
plt.vlines([start, stop], 0, 1.1 * np.max(y_profile_full), color = 'red')

plt.title('y profile (full FOV)')
plt.xlabel('row')
plt.ylabel('counts')
plt.ylim(0, 1.05 * np.max(y_profile_full))
plt.show()

ROI = slice(start, stop)


#%% find center of mass for segmentation

# fwhm = 10 #mm
# thresh = 0.05

# frame_smoothed = gaussian_filter(frame, sigma = fwhm / (2.355 * pixel_size))

# plt.imshow(frame_smoothed)
# plt.colorbar()
# plt.title("Frame " + str(frame_index + 1) + " (smoothed)")
# plt.show()


# mask = np.where(frame_smoothed > thresh * np.max(frame_smoothed), 1, 0)

# plt.imshow(mask)
# plt.show()

# y_profile_mask = np.sum(frame * mask, axis = 0)
# plt.plot(y_profile_mask)
# plt.show()

# inds = np.nonzero(mask)
# top = np.min(inds[0])
# bottom = np.max(inds[0])
# left = np.min(inds[1])
# right = np.max(inds[1])




#%% spline fitting to x profile

s = 15000
k = 3


x_profile = np.sum(frame[ROI, :], axis = 0)

tck = sp.interpolate.splrep(xs, x_profile, s = s, k = k)
num_knots = len(tck[0])
print("number of knots = ", num_knots)

spline = sp.interpolate.splev(xs, tck)


plt.plot(xs, x_profile, label = 'data')
plt.plot(xs, spline, label = 'spline')
plt.legend()
plt.xlim(-150,150)
plt.title("x profile in ROI")
plt.show()


#%% Compare fitted spline to profiles of reduced frames

names = os.listdir(path)

# names = ['100']
for name in names:    
    p = int(name)
    
    dcm_dir     = path + name
    dcm         = pd.dcmread(dcm_dir + "\\" + os.listdir(dcm_dir)[0])
    new_vol         = dcm.pixel_array 
    
    if name == '100':
        rescale = 1
    else:
        field = dcm[0x0040,0x9096][0]
        rescale = field[0x0040,0x9225].value
    new_vol = new_vol * rescale
    
    new_frame = new_vol[frame_index, :, :]
    
    new_x_profile = np.sum(new_frame[ROI, :], axis = 0)
    scaled_spline = spline * p / 100
    
    
    plt.plot(new_x_profile, label = p)
    plt.plot(scaled_spline)
    plt.title("Reduced profile " + str(p) + "%")
    plt.show()
    
    plt.plot((new_x_profile - scaled_spline) / scaled_spline * 100)
    plt.title("Deviation " + str(p) + "%" )
    plt.ylim(-10,10)
    plt.xlim(75,175)
    plt.show()


#%% segmentation in x direction

start = 115
stop = 135


plt.plot(columns, x_profile_full)
plt.vlines([start, stop], 0, 1.1 * np.max(x_profile_full), color = 'red')

plt.title('x profile (full FOV)')
plt.xlabel('row')
plt.ylabel('counts')
plt.ylim(0, 1.05 * np.max(x_profile_full))
plt.show()

ROI2 = slice(start, stop)

#%% check for poisson statistics in central region
M = np.max(vol)
bins = np.arange(M + 1)

for i in range(10):
    new_frame = vol[i,:,:]
    CFOV = new_frame[ROI, ROI2]
    
    mu = np.mean(CFOV)
    
    plt.hist(CFOV.flatten(), bins = bins)
    plt.plot(bins, poisson(bins, mu = mu) * np.size(CFOV))
    plt.title(str(i))
    plt.show()
    print(i ,": mean=", np.round(mu, 2), "----- std = ", np.round(np.var(CFOV, ddof = 1)))