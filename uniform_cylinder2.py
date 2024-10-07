# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:40:57 2024

@author: wclaey6

Analysis of uniform cylinder data.
1) look at the average transversal profile and choose the slices that will be used
2) find the axis of the cylinder
3) plot the average radial profile and choose the radius that will be used
4) plot the average angular profile
5) look at the profiles in some more detail


-------
v2 27/09/2024: bit more advanced version. e.g. calculates gibbs statistics

"""

import os
import math

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import pymirc.fileio as pmf
import pydicom

from IQ_functions import *

# =============================================================================
# INPUTS AND DATA READING
# =============================================================================
# Reading the data and post-smoothing
# =============================================================================

path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\2024-09__Studies\\'
name = '01'



# Other parameters (more parameters can be set in each block of code)
units = "cnts"         # units of the image data 

sm_fwhm_mm    = 0       # width of the Gaussian kernel used for post smoothing
radius        = 100     # known radius of the cylindrical phantom

scale_factor = 1
# =============================================================================

# reading data
print("Loading series ", name, " with units ", units)
dcm_dir     = path + name
dcm         = pmf.DicomVolume(os.path.join(dcm_dir, '*')) # join multiple slices to a single volume
vol         = dcm.get_data() 
vol         = vol * scale_factor

if units == "cnts":             #change data type to avoid overflow probles later
    vol = vol.astype(np.uint)

# reading metadata
voxsizes    = dcm.voxsize 
voxsize = voxsizes[0]
slice_thickness = voxsizes[2]

# post smoothing
if sm_fwhm_mm > 0:
  vol = gaussian_filter(vol, sigma = sm_fwhm_mm / (2.355*voxsize))
  print("Post-smoothing the image with a ", sm_fwhm_mm, "mm Gaussian filter")
  
  
#%%
# =============================================================================
# TRANSVERSAL PROFILE
# =============================================================================
# First look at the cylinder
# Determine the slices of interest
# =============================================================================

cutoff = 5  # adjust to decrease the size of the FOV

# =============================================================================

# other parameters
thresh = 0.1    # threshold for rough segmentation of the cylinder
length = 200    # cylinder length in mm


# initialization
num_slices = np.size(vol, axis = 2)
slices = np.linspace(1, num_slices, num_slices)

slices_in_cylinder = length / slice_thickness

# create a mask of the cylinder via thresholding
mask = np.where(vol > thresh * np.max(vol), 1, 0)


# calculate center of mass (in the z direction) of the cilinder
weights = np.sum(vol * mask, axis = (0,1))
z_center = np.sum(slices * weights) / np.sum(weights)

# calculate edges of the cylinder
left = int(z_center - slices_in_cylinder / 2)
right = int(z_center + slices_in_cylinder / 2)
left = max(0, left)
right = min(right, num_slices)

# calculate slices of interest
start = int(left + cutoff)
stop = int(right - cutoff)

ROI = slice(start, stop)

# plotting
M  = np.max(np.sum(vol, axis = (0,1))) #normalization constant 
plt.plot(np.sum(vol, axis = (0,1)) / M, label = 'raw')
plt.plot(weights / M, label = 'thresholded')
plt.vlines([left, right], 0, 1.1, color = 'red', label = 'edge')
plt.vlines([start, stop], 0, 1.1, color = 'green', label = 'ROI')
plt.title("Transversal profile")
plt.xlabel("Slice number")
plt.ylabel("Counts (normalized)")
plt.ylim(0,1.1)
# plt.xlim(110,215)
plt.legend()
plt.show()

# printouts
print("Analyzing transversal profile")
print("Calculated phantom length =", round((int(right) - int(left)) * slice_thickness / 10,2), "cm")
print("Slice thickness =", round(slice_thickness / 10,2), "cm")
print("Discarding", cutoff, "slices at each end")
print("ROI legth =", round((stop - start) * slice_thickness / 10,2), " cm")
  
#%%
# =============================================================================
# SEGMENT THE CILINDER
# =============================================================================
# Principle: find the center of mass of each slice
# Fit a linear function through the center to find the cilinder axis
# =============================================================================
plot_slices = False
# =============================================================================


# get the center of each slice
centers = get_slice_centers(vol, voxsizes, weights, relth = 0.2)

# fit a linear function to the x and y coordinates of the centers to get the cilinder axis
params_x, rest_x = sp.optimize.curve_fit(linear, slices[ROI], centers[ROI, 0], sigma = 1 / weights[ROI])
params_y, rest_y = sp.optimize.curve_fit(linear, slices[ROI], centers[ROI, 1], sigma = 1 / weights[ROI])

# calculate fitted centers from the cilinder axis
fitted_centers_x = linear(slices, params_x[0], params_x[1])
fitted_centers_y = linear(slices, params_y[0], params_y[1])

# plotting
plt.plot(slices[left:right + 1], centers[left:right + 1, 0], label = "x center")
plt.plot(slices[left:right + 1], centers[left:right + 1, 1], label = "y center")
plt.plot(slices, fitted_centers_x, label = "x fit")
plt.plot(slices, fitted_centers_y, label = "y fit")
plt.title("Calculated sphere centers")
plt.xlabel("slice number")
plt.ylabel("position (mm)")
plt.legend()
plt.show()

print("Finding cylinder axis")

# convert to cylindrical coordinates
r, theta, z = get_cylindrical_coordinates(vol, fitted_centers_x, fitted_centers_y, voxsize)
print("Calculating cylindrical coordinates")

# show the slices with the calculated centers (optional)
if plot_slices:
    for i in range(start, stop + 1):
        plt.imshow(np.transpose(vol[:,:,i]), vmax = np.max(vol))
        point = np.unravel_index(np.argmin(r[:,:,i]), np.shape(vol[:,:,i]))
        # plt.scatter(point[0], point[1], s = 1)
        plt.colorbar(label = units)
        plt.title("Slice "+ str(i))
        plt.show()


#%%
# =============================================================================
# GET AVERAGE RADIAL PROFILE & DETERMINE RADIUS OF INTEREST
# =============================================================================
# optional: fit a smoothing spline to the profile to quantify the uniformity


cutoff_radius = 65    # mm

res = 2          # bin width in mm
r_max = 120        # mm

fit_spline = True       # whether or not to fit a spline
fit_gibbs = True        # whether or not to calculate Gibbs artifact metrics. If true, the cutoff_radius will be updated using this information 

s = 50              # smoothness of the spline (default = None)
# =============================================================================

# get r bins and edges
r_bins = np.arange(0, r_max + res / 2, res)
r_bin_centers = r_bins[:-1] + res / 2

# get relevant data
rs = r[:,:,ROI].flatten()
counts = vol[:,:,ROI].flatten()
counts =counts

# split data into r bins using numpy's histogram function
sum_hist, bins = np.histogram(rs, r_bins, weights = counts)     # sum of the counts in each r bin
point_hist, bins = np.histogram(rs, r_bins)                     # number of points in each r bin
r_profile = sum_hist / point_hist                               # average of the counts in each r bin


#plotting radial profile
plt.plot(r_bin_centers, r_profile, marker = '.', linestyle = 'None', label = 'data')
# plt.vlines(cutoff_radius, 0, 1.05 * np.nanmax(r_profile), color = 'green', label = 'ROI')
plt.vlines(radius, 0, 1.05 * np.nanmax(r_profile), color = 'red', label = 'edge')


## investigate Gibbs artifact


# fitting smoothing spline to the data
if fit_spline:   
    # calculate weights
    wts = np.where(sum_hist != 0, point_hist / (np.sqrt(sum_hist)), 0.001)    # weight is in verse of Poisson standard deviation

    # remove Nan's
    wts2 = wts[~np.isnan(r_profile)]
    r_profile2 = r_profile[~np.isnan(r_profile)]
    r_bin_centers2 = r_bin_centers[~np.isnan(r_profile)]
    
    # get spline parameters
    tck = sp.interpolate.splrep(r_bin_centers2, r_profile2, s = s, w = wts)
    if s is None:
        m = np.size(r_bin_centers2)
        s = m - np.sqrt(2*m)
    print("Fitting spline with smoothness ", round(s))
    print("Number of knots = ", np.size(tck[0]))
    
    # plot spline
    a = np.linspace(0, r_max, 10 * r_max + 1)
    y = sp.interpolate.splev(a, tck)
    plt.plot(a, y, label = 'spline')
    
    if fit_gibbs:
        R0 = cutoff_radius
        i0 = np.argmin(np.abs(a - R0))
        M_gibbs = np.max(y[i0:])
        i_M_gibbs = i0 + np.argmax(y[i0:])
        m_gibbs = np.min(y[i0:i_M_gibbs])
        i_m_gibbs = i0 + np.argmin(y[i0:i_M_gibbs])
        gibbs = (M_gibbs - m_gibbs) / (M_gibbs + m_gibbs) * 100
        gibbs_width = a[i_M_gibbs] - a[i_m_gibbs]
        print("Gibbs height =", round(gibbs, 2), "%")
        print("Gibbs width =", round(gibbs_width, 2), "mm")
        print("Gibbs max position =", round(a[i_M_gibbs], 2), "mm")
        plt.vlines([a[i_M_gibbs], a[i_m_gibbs]], 0, 1.05 * np.nanmax(r_profile), color = 'blue', label = 'gibbs')
        cutoff_radius = a[i_m_gibbs] - gibbs_width
    plt.vlines(cutoff_radius, 0, 1.05 * np.nanmax(r_profile), color = 'green', label = 'ROI')
    
    # calculate uniformity metrics
    i_min = math.ceil(10 * np.min(r_bin_centers)) # avoid extrapolation before first datapoint
    i_max = np.argmin(np.abs(a - cutoff_radius))  # avoid PVE
    M = np.max(y[i_min:i_max])
    M_loc = a[i_min + np.argmax(y[i_min:i_max])]
    m = np.min(y[i_min:i_max])
    m_loc = a[i_min + np.argmin(y[i_min:i_max])]
    dev = (M - m)
    dev2 = (M - m) / (M + m) * 200
    print("Maximum: ", np.round(M, 2), units, "at", M_loc , " mm")
    print("Minimum: ", np.round(m, 2), units, "at", m_loc , " mm")
    print("Deviation = ", round(dev, 2), units)
    print("Deviation = ", round(dev2, 2), "%")
    
    plt.vlines(m_loc, 0, 1.05 * np.nanmax(r_profile), color = 'purple', label = 'min', linestyle = 'dotted')
    plt.vlines(M_loc, 0, 1.05 * np.nanmax(r_profile), color = 'purple', label = 'max', linestyle = 'dashed')

# plot layout
plt.xlabel("Radius (mm)")
plt.ylabel(units)
plt.title("Radial Profile")
plt.ylim(0, 1.05* np.nanmax(r_profile))
plt.xlim(0, r_max)
plt.legend()
plt.show()




#%%

# =============================================================================
# GET AVERAGE ANGULAR PROFILE & DETERMINE RADIUS OF INTEREST
# =============================================================================
# optional: fit a smoothing spline to the profile to quantify the uniformity

res = 2           # bin width in degrees


fit_spline = True       # whether or not to fit a spline
s = 100000                # smoothness of the spline (default = None)

decimal_places = 2     #number of decimal places in the printed results
# =============================================================================


mask = np.where(r[:,:,ROI] < cutoff_radius, 1, 0)

theta_bins = np.arange(-180, 180 + res / 2, res)
theta_bin_centers = theta_bins[:-1] + res / 2

indices = np.nonzero(mask.flatten())[0]

rs = r[:,:,ROI].flatten()
thetas = theta[:,:,ROI].flatten()
counts = vol[:,:,ROI].flatten()

counts = counts[indices]
thetas = thetas[indices]

# split data into theta bins using numpy's histogram function
sum_hist, bins = np.histogram(thetas, theta_bins, weights = counts)     # sum of the counts in each theta bin
point_hist, bins = np.histogram(thetas, theta_bins)                     # number of points in each theta bin
theta_profile = sum_hist / point_hist                               # average of the counts in each theta bin

# plotting
plt.plot(theta_bin_centers, theta_profile, label = "data")


# fitting smoothing spline to the data
if fit_spline:   
    # calculate weights
    wts = np.where(sum_hist != 0, point_hist / (np.sqrt(sum_hist)), 0)    # weight is in verse of Poisson standard deviation

    # remove Nan's
    wts2 = wts[~np.isnan(theta_profile)]
    theta_profile2 = theta_profile[~np.isnan(theta_profile)]
    theta_bin_centers2 = theta_bin_centers[~np.isnan(theta_profile)]
    
    # get spline parameters
    tck = sp.interpolate.splrep(theta_bin_centers2, theta_profile2, s = s, w = wts2)
    if s is None:
        m = np.size(theta_bin_centers2)
        s = m - np.sqrt(2*m)
    print("Fitting spline with smoothness ", round(s))
    print("Number of knots = ", np.size(tck[0]))
    
    # plot spline
    a = np.linspace(-180, 180, 3601)
    y = sp.interpolate.splev(a, tck)
    plt.plot(a, y, label = 'spline')
    
    # calculate uniformity metrics
    i_min = math.ceil(1800 + 10 * np.min(theta_bin_centers)) # avoid extrapolation before first datapoint
    i_max = math.ceil(1800 + 10 * np.max(theta_bin_centers)) # avoid extrapolation after last datapoint
    M = np.max(y[i_min : i_max])
    M_loc = a[i_min + np.argmax(y[i_min : i_max])]
    m = np.min(y[i_min : i_max])
    m_loc = a[i_min + np.argmin(y[i_min : i_max])]
    dev = (M - m)
    dev2 = (M - m) / (M + m) * 200
    print("Maximum: ", np.round(M, decimal_places), units, "at", np.round(M_loc), "°")
    print("Minimum: ", np.round(m, decimal_places), units, "at", np.round(m_loc), "°")
    print("Deviation = ", round(dev, decimal_places), units)
    print("Deviation = ", round(dev2, decimal_places), "%")


# plot layout
plt.title("Angular profile")
plt.xlabel("Theta (°)")
plt.ylabel(units)
plt.xlim(-180,180)
# plt.ylim(0,1.05 * np.max(theta_profile))
plt.legend()
plt.show()


# calculating and printing some metrics (spline metrics are probably more accurate)
# print("Maximum deviation = ", round((np.max(theta_profile)-np.min(theta_profile))/np.mean(theta_profile) * 100, 2), "%")
# print("Standard deviation = ", round(np.std(theta_profile) / np.mean(theta_profile) * 100, 2), "%")





#%%
# =============================================================================
# RADIAL PROFILES FOR DIFFERENT SLICES
# =============================================================================
# 
# =============================================================================
res = 1           # bin width in mm
r_max = 150       # mm

selected_slices = [25, 35, 45, 55, 65, 75, 85, 95]   # the slices to plot
# =============================================================================

print("Slices in ROI: ", start , "-", stop)

# get r bins and edges
r_bins = np.arange(0, r_max + res / 2, res)
r_bin_centers = r_bins[:-1] + res / 2

for i in selected_slices:
    if i > np.size(vol, axis = 2):
        print("Slice ", i , "out of range")
    else:
        # get relevant data
        rs = r[:,:,i].flatten()
        counts = vol[:,:,i].flatten()
        counts =counts
        
        # split data into r bins using numpy's histogram function
        sum_hist, bins = np.histogram(rs, r_bins, weights = counts)     # sum of the counts in each r bin
        point_hist, bins = np.histogram(rs, r_bins)                     # number of points in each r bin
        r_profile = sum_hist / point_hist                               # average of the counts in each r bin
        
        #plots
        plt.plot(r_bin_centers, r_profile, label = "slice " + str(i))
    
plt.vlines(cutoff_radius, 0, 1.05 * np.nanmax(r_profile), color = 'green', label = "")
plt.vlines(radius, 0, 1.05 * np.nanmax(r_profile), color = 'red', label = "")
plt.xlabel("Radius (mm)")
plt.ylabel(units)
plt.title("Radial profile per slice")
plt.ylim(0, 1.5* np.nanmax(r_profile))
plt.xlim(0, r_max)
plt.legend()
plt.show()


#%%
# =============================================================================
# RADIAL PROFILES FOR DIFFERENT ANGLES
# =============================================================================
# 
# =============================================================================

# bin settings
r_max = 120
r_res = 2        # r bin width in mm
theta_res = 90  # theta bin width in °

# =============================================================================


# get the relevant data

rs = r[:,:,ROI].flatten()
thetas = theta[:,:,ROI].flatten()
counts = vol[:,:,ROI].flatten()

# get r bins and edges
r_bins = np.arange(0, r_max + r_res / 2, r_res)
r_bin_centers = r_bins[:-1] + r_res / 2

# get theta bins and edges
theta_bins = np.arange(-180, 180 + theta_res / 2, theta_res)
theta_bin_centers = theta_bins[:-1] + theta_res / 2


data, rest1, rest2 = np.histogram2d(rs, thetas, bins = [r_bins, theta_bins], weights = counts)
numbers, rest1, rest2 = np.histogram2d(rs, thetas, bins = [r_bins, theta_bins])

r_profile = np.sum(data, axis = 1) / np.sum(numbers, axis = 1)
r_profiles = data / numbers



for i in range(np.size(r_profiles, axis = 1)):
    plt.plot(r_bin_centers, r_profiles[:,i], marker = '.', label = str(i * theta_res) + "°")

# plt.ylim(0.45,0.55)
plt.legend()
plt.title("Radial profile per angle")
plt.ylabel(units)
plt.xlabel("Radius (mm)")
plt.show()

#%%
# =============================================================================
# ANGULAR PROFILES FOR DIFFERENT RADII
# =============================================================================
# 
# =============================================================================
# bin settings
r_max = 90
r_res = 15
theta_res = 5
# =============================================================================


# get the relevant data
rs = r[:,:,ROI].flatten()
thetas = theta[:,:,ROI].flatten()
counts = vol[:,:,ROI].flatten()

# get r bins and edges
r_bins = np.arange(0, r_max + r_res / 2, r_res)
r_bin_centers = r_bins[:-1] + r_res / 2

# get theta bins and edges
theta_bins = np.arange(-180, 180 + theta_res / 2, theta_res)
theta_bin_centers = theta_bins[:-1] + theta_res / 2


data, rest1, rest2 = np.histogram2d(rs, thetas, bins = [r_bins, theta_bins], weights = counts)
numbers, rest1, rest2 = np.histogram2d(rs, thetas, bins = [r_bins, theta_bins])

theta_profile = np.sum(data, axis = 1) / np.sum(numbers, axis = 1)
theta_profiles = data / numbers


for i in range(np.size(theta_profiles, axis = 0)):
    plt.plot(theta_bin_centers, theta_profiles[i,:], marker = '.', label = str(r_bin_centers[i]) + ' mm')

# plt.ylim(0.45,0.55)

plt.legend()
plt.title("Angular profile per radius")
plt.ylabel(units)
plt.xlabel("Angle (°)")
plt.show()

#%%
# =============================================================================
# TRANSVERSAL UNIFORMITY TESTS
# =============================================================================
# Redo the transversal uniformity analysis with the properly segmented phantom
# =============================================================================
decimal_places = 3     #number of decimal places in the printed results

# =============================================================================

# creating the segmentation mask
mask = np.where(r < cutoff_radius, 1, 0)

# calculating standard uniformity metrics
means = np.sum(mask * vol, axis = (0,1)) / np.sum(mask , axis = (0,1))
dev = means - np.mean(means[ROI])   # deviation from the mean
dev_perc = (means - np.mean(means[ROI])) / np.mean(means[ROI]) * 100 # devuation from the mean (%)

mean = np.mean(means[ROI])
m = np.min(means[ROI])
m_loc = start + np.argmin(means[ROI])
M = np.max(means[ROI])
M_loc = start + np.argmax(means[ROI])

# print uniformity metrics
print("Uniformity metrics: ")
print("Mean activity = ", round(mean), units)
print("Maximum: slice ", M_loc, "with", round(M), units, "(+", round((M - mean) / mean * 100, 2), "%)")
print("Minimum: slice ", m_loc, "with", round(m), units, "(-", round((mean - m) / mean * 100, 2), "%)")


# fitting a linear function to the profile to detect any gradients
shifted_slices = slices - z_center      # shift slices relative to phantom center
params, rest = sp.optimize.curve_fit(linear, shifted_slices[ROI], means[ROI])

a, b = params
a_err, b_err = np.sqrt(np.diag(rest))

# print fit results
print("---------------")
print("Fit results: ")
print("Mean = ", round(b, decimal_places), "+/-", round(b_err, decimal_places), units)
print("Slope = ", round(a, decimal_places), "+/-", round(a_err, decimal_places), units, "/slice")
print("Slope = ", round(a / b * 100, decimal_places), " % per slice")

# plotting the profile
M  = np.max(means) #normalization constant 
plt.plot(slices, means, label = 'profile')
plt.plot(slices, linear(shifted_slices, a, b))
plt.vlines([left, right], 0, 1.1 * M, color = 'red', label = 'edge')
plt.vlines([start, stop], 0, 1.1 * M, color = 'green', label = 'ROI')
plt.title("Transversal profile (segemented)")
plt.xlabel("Slice number")
plt.ylabel(units)
plt.ylim(0,1.1 * M)
# plt.xlim(110,215)
plt.legend()
plt.show()

# plotting the deviations
plt.plot(dev_perc, label = 'data')
plt.plot(slices, (a * shifted_slices + b - np.mean(means[ROI])) / np.mean(means[ROI])*100, label = 'linear fit')
plt.hlines(0, 0, num_slices, color = 'red', linestyle = 'dashed')
plt.title("Transversal uniformity")
plt.xlabel("Slice number")
plt.ylabel("Deviation (%)")
plt.ylim(1.2 * np.min(dev_perc[ROI]),1.2 * np.max(dev_perc[ROI]))
plt.xlim(left, right)
plt.vlines([start, stop], -1.1 * M, 1.1 * M, color = 'green', label = 'ROI')
plt.legend()
plt.show()




































