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
import pydicom as py

# from IQ_functions import *
from projection_data_functions import *
from data_reading import *



# =============================================================================
# Select patient and load data
# =============================================================================

path = "C:\\Wies Data\\Data for Python"

patient_name = "QCintevo_Tb"
patient_ID = "Cilinder"


database = create_database(path)
list_patients(database)

patient = lookup_patient(database, patient_name, patient_ID = patient_ID)

#%% 
# =============================================================================
# Select (collection of) series
# =============================================================================

series_description = "reduced"
date = None

series = find_series(patient, series_description, date= date)
dcm_list = load_series(series)


#%%
dcm = dcm_list[2]

# reading the required data
vol         = get_rescaled_data(dcm)
frame_index = 0
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

# Show an example frame

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

s = 20000
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

fractions = [10,20,30,40,50,60,70,80,90]

# names = ['100']





# for i in range(len(fractions)):    
#     p = fractions[i]
    
#     full_path = path + "\\" + selection['path2'].values[i]

#     dcm_dir     = full_path
#     dcm         = py.dcmread(dcm_dir + "\\" + os.listdir(dcm_dir)[0])
#     new_vol     = get_rescaled_data(dcm)
      
#     new_frame = new_vol[frame_index, :, :]
    
#     new_x_profile = np.sum(new_frame[ROI, :], axis = 0)
#     scaled_spline = spline * p / 10
    
    
#     plt.plot(new_x_profile, label = p)
#     plt.plot(scaled_spline, label = 'spline')
#     plt.title("Reduced profile " + str(p) + "%")
#     plt.legend()
#     plt.show()
    
#     plt.plot((new_x_profile - scaled_spline) / scaled_spline * 100)
#     plt.title("Deviation " + str(p) + "%" )
#     plt.ylim(-10,10)
#     plt.xlim(75,175)
#     plt.show()


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


# still todo

M = np.max(vol[19,:,:])
bins = np.arange(M + 2) - 0.5
xs = bins[:-1]  + 0.5

devs = []
for i in range(20):
    new_frame = vol[i,:,:]
    CFOV = new_frame[ROI, ROI2]
    
    mu = np.mean(CFOV)
    var = np.var(CFOV, ddof = 1)
    
    devs.append(mu - var)
    
    plt.hist(CFOV.flatten(), bins = bins)
    plt.plot(xs, poisson(xs, mu = mu) * np.size(CFOV))
    plt.title(str(i))
    plt.show()
    print(i ,": mean=", np.round(mu, 2), "----- var = ", np.round(var,2))
    

    
plt.plot(devs, '.')
plt.title("deviation between variance and mean")
plt.xlim(0,20)
plt.xlabel("frame number")
plt.show()

print("mean deviation =", round(np.mean(devs),4), "=" , round(np.mean(devs)/mu*100, 2), '%')

#%%

sumframe = np.sum(vol[0:1,:,:], axis = 0)
CFOV = sumframe[ROI, ROI2]
data = CFOV.flatten()

M = np.max(CFOV)
bins = np.arange(M + 2) - 0.5
xs = bins[:-1]  + 0.5

freqs, b = np.histogram(data, bins = bins) 

N = np.size(data)
mu = np.mean(data)

y = np.log(freqs) + sp.special.gammaln(xs + 1)
y_fix = np.log(N) - mu + xs * np.log(mu)

res = y - y_fix

variance = ( 1 - poisson(xs, mu)) / ( N * poisson(xs, mu))
std = np.sqrt(variance)

plt.plot(y, '.')
plt.plot(y_fix)


# plt.xlim(0,len(bins))
plt.title('Poisson-ness plot')
plt.show()



plt.plot(res, '.')
plt.hlines(0, 0, len(bins), color = 'orange')
plt.plot(- std, linestyle = "dotted", color = 'red')
plt.plot(std, linestyle = "dotted", color = 'green')
plt.plot(- 2*std, linestyle = "dashed", color = 'red')
plt.plot(2*std, linestyle = "dashed", color = 'green')
plt.title('deviations')
plt.ylim(1.1 * np.min(res[np.isfinite(res)]), 1.1 * np.max(res[np.isfinite(res)]))
plt.xlim(0, bins[-1])
plt.show()




# try to do some chi2 magic

#expected values


# 
ps = poisson(xs, mu = mu)           # calculated probabilities
ps = np.append(ps, 1-np.sum(ps))    # add probability of everything bigger
expected_freqs = ps * N             # get expected frequencies

Os, Es = rebin_data(freqs, expected_freqs) # prepare data for chi-square test


t = sp.stats.chisquare(Os, Es)

print("Chisquare =", round(t[0],2), ", p-value =", round(t[1],3))






