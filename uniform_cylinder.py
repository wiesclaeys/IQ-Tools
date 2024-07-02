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

"""

import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import pymirc.fileio as pmf

from IQ_functions import *

# =============================================================================
# INPUTS AND DATA READING
# =============================================================================
# Reading the data and post-smoothing
# =============================================================================

path          = 'data'
name          = 'Truepoint'


# Other parameters (more parameters can be set in each block of code)
units = "cnts"         # units of the image data

sm_fwhm_mm    = 0       # width of the Gaussian kernel used for post smoothing
radius        = 100     # known radius of the cylindrical phantom

# =============================================================================

# reading data
print("Loading series ", name, " with units ", units)
dcm_dir     = path + name
dcm         = pmf.DicomVolume(os.path.join(dcm_dir, '*')) # join multiple slices to a single volume
vol         = dcm.get_data() 

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

cutoff = 8  # adjust to decrease the size of the FOV

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
print("Calculated phantom length = ", round((int(right) - int(left)) * slice_thickness / 10,2), "cm")
print("Slice thickness = ", round(slice_thickness / 10,2), "cm")
print("Discarding ", cutoff, "slices at each end")
print("ROI legth = ", round((stop - start) * slice_thickness / 10,2), " cm")
  
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

cutoff_radius = 75    # mm

res = 2          # bin width in mm
r_max = 150        # mm

fit_spline = True       # whether or not to fit a spline
s = 500                # smoothness of the spline (default = None)
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
plt.vlines(cutoff_radius, 0, 1.05 * np.nanmax(r_profile), color = 'green', label = 'ROI')
plt.vlines(radius, 0, 1.05 * np.nanmax(r_profile), color = 'red', label = 'edge')


# fitting smoothing spline to the data
if fit_spline:   
    # calculate weights
    wts = point_hist / (np.sqrt(sum_hist))    # square root of total number of counts in each bin
    # plt.plot(r_bin_centers, 100 * wts)
    # remove Nan's
    wts2 = wts[~np.isnan(r_profile)]
    r_profile2 = r_profile[~np.isnan(r_profile)]
    r_bin_centers2 = r_bin_centers[~np.isnan(r_profile)]
    
    # get spline parameters
    tck = sp.interpolate.splrep(r_bin_centers2, r_profile2, s = s, w = wts2)
    if s is None:
        m = np.size(r_bin_centers2)
        s = m - np.sqrt(2*m)
    print("Fitting spline with smoothness ", round(s))
    print("Number of knots = ", np.size(tck[0]))
    
    # plot spline
    a = np.linspace(0, r_max, r_max + 1)
    y = sp.interpolate.splev(a, tck)
    plt.plot(a, y, label = 'spline')

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

res = 2           # bin width in degrees
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
plt.plot(theta_bin_centers, theta_profile)
plt.title("Angular profile")
plt.xlabel("Theta (°)")
plt.ylabel(units)
plt.xlim(-180,180)
# plt.ylim(0,1.05 * np.max(theta_profile))
plt.show()


# calculating and printing some metrics
print("Maximum deviation = ", round((np.max(theta_profile)-np.min(theta_profile))/np.mean(theta_profile) * 100, 2), "%")
print("Standard deviation = ", round(np.std(theta_profile) / np.mean(theta_profile) * 100, 2), "%")





#%%
# =============================================================================
# RADIAL PROFILES FOR DIFFERENT SLICES
# =============================================================================
# 
# =============================================================================
res = 1           # bin width in mm
r_max = 150       # mm
# =============================================================================

print("Slices in ROI: ", start , "-", stop)

# get r bins and edges
r_bins = np.arange(0, r_max + res / 2, res)
r_bin_centers = r_bins[:-1] + res / 2

selected_slices = [40,45,50,55,60]
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
plt.ylim(0, 1.05* np.nanmax(r_profile))
plt.xlim(0, r_max)
plt.legend()
plt.show()


#%%
# =============================================================================
# RADIAL PROFILES FOR DIFFERENT ANGLES
# =============================================================================
# 
# =============================================================================
res = 1           # bin width in mm
r_max = 150       # mm
# =============================================================================

# bin settings
r_max = 120
r_res = 2
theta_res = 90



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
res = 1           # bin width in mm
r_max = 150       # mm
# =============================================================================
# bin settings
r_max = 90
r_res = 15
theta_res = 5



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


# =============================================================================

# get the relevant data
mask = np.where(r < cutoff_radius, 1, 0)
center = stop - start
shifted_slices = slices - center
means = np.sum(mask * vol, axis = (0,1)) / np.sum(mask , axis = (0,1))

params, rest = sp.optimize.curve_fit(linear, shifted_slices[ROI], means[ROI])

a, b = params
a_err, b_err = np.sqrt(np.diag(rest))


print("Mean = ", round(b, 2), "+/-", round(b_err, 2), units)
print("Slope = ", round(a, 2), "+/-", round(a_err, 2), units, "/slice")

# plotting
M  = np.max(means) #normalization constant 
plt.plot(slices, means, label = 'segmented')
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


dev = means - np.mean(means[ROI])
plt.plot(dev)
plt.plot(slices, a * shifted_slices + b - np.mean(means[ROI]))
plt.hlines(0, 0, num_slices, color = 'red', linestyle = 'dashed')
plt.title("Transversal uniformity")
plt.xlabel("Slice number")
plt.ylabel(units)
plt.ylim(1.2 * np.min(dev[ROI]),1.2 * np.max(dev[ROI]))
plt.xlim(left, right)
plt.vlines([start, stop], -1.1 * M, 1.1 * M, color = 'green', label = 'ROI')
plt.show()




































