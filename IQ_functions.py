# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:22:48 2023

@author: wclaey6
"""


from scipy.ndimage import gaussian_filter
import numpy as np


def linear(x, a, b):
    """ Just a linear function """
    return a * x + b

def get_slice_centers(vol, 
                      voxsizes,
                      sl_weights,
                      relth = 0.5, fwhm = 5):
    """ Get the center of gravity in each slice within the cylinder
    
    Parameters
    ----------
    vol : 3d numpy array 
      containing the volume
      
    sl_weights : 1d numpy array
      the weights (= total number of counts) of each slice

    voxsizes : 3 component array 
      with the voxel sizes

    relth : float, optional
      the relative threshold (fraction of maximum) for the coarse 
      delination of the cylinder (voxel based) - default 0.5
      
    relth2 : float, optional
      the relative threshold (fraction of maximum) for the coarse 
      delination of the cylinder (slice based) - default 0.5
      
    fwhm : float
      width of smoothing kernel (mm)
      
   Returns
   -------
   centers : array of float
       [x position, y position]
    """
    ## Strategy: 2 steps (based on pynemaiqpet)
    # 1) center of mass of thresholded mask
    # 2) center of mass of smoothed slice
        
    ### STEP 1: Initialization:
   
    # Simple thresholding: create a mask corresponding to all voxels above relth*max_value
    absth = relth*(vol.max())

    mask              = np.zeros_like(vol, dtype = np.uint8)
    mask[vol > absth] = 1
    
    # Preparing indices
    i0, i1     = np.indices(vol[:,:,0].shape)
    i0         = i0*voxsizes[0]
    i1         = i1*voxsizes[1]
      
    # calculate the maxmimum radius of the subvolumes
    # all voxels with a distance bigger than rmax (=FOV/2) will not be included in the fit
    rmax = np.min((i0.max(),i1.max()))/2

    # Smoothing the volume
    sigma = fwhm / (2.355*voxsizes)
    vol_sm = gaussian_filter(vol, sigma = sigma)
    
    ### STEP 2: Slice per Slice Calculation
    centers = []
    for i in range(np.size(vol, axis = 2)):
        sl = vol[:,:,i]
        sl_sm = vol_sm[:,:,i]
        sl_mask = mask[:,:,i]
        
        # First get the COM of the coarsly delineated cilinder (thresholding)
        x_weights = np.sum(sl * sl_mask, axis = 1)
        y_weights = np.sum(sl * sl_mask, axis = 0)
        summedweights = np.sum(x_weights)

        
        if summedweights > 0:
            c0 = np.sum(i0 * sl * sl_mask) / summedweights  
            c1 = np.sum(i1 * sl * sl_mask) / summedweights  
            
            # print(c0, c1)
            
            r  = np.sqrt((i0 - c0)**2 + (i1 - c1)**2)
        
        
            # Then get the COM of the smoothed cilinder 
        
            weights       = sl_sm[r <= rmax]
            summedweights = np.sum(weights)
            
            sl_mask2 = np.where(r <= rmax, 1, 0)
        
            d0 = np.sum(i0 * sl * sl_mask2) / summedweights  
            d1 = np.sum(i1 * sl * sl_mask2) / summedweights

            centers.append([d0, d1])
        
        else:
            centers.append([None, None])

    return np.array(centers)

def get_cylindrical_coordinates(vol, centers_x, centers_y, voxsize):
    """
    Calculate the cylindrical coordinates relative to a given set of centers

    Parameters
    ----------
    data : 3d numpy array
        array having the same shape as the data
    fitted_centers_x : 1d numpy array
        the x coordinates of the centers
    fitted_centers_y : 1d numpy array
        the y coordinates of the centers
    voxsize : float
        voxel size in mm

    Returns
    -------
    coords : 3-tuple of numpy arrays
        3d arrays containing the r, theta and z coordinate of each point respectively

    """
    
    # Get Cartesian coordinates
    i0, i1, i2 = np.indices(vol.shape)
    i0 = i0 * voxsize                   # x axis (mm)
    i1 = i1 * voxsize                   # y axis (mm)
    i2 = i2                             # z axis (index)

    # Get Cartesian coordinates centered around the cilinder for each slice
    x = []
    y = []
    for i in range(np.size(vol, axis = 2)):
        x.append(i0[:,:,i]-centers_x[i])
        y.append(i1[:,:,i]-centers_y[i])       
    x = np.moveaxis(np.array(x),0,-1)
    y = np.moveaxis(np.array(y),0,-1)
            
    # Convert centered Cartesian coordinate to polar coordinates
    theta = np.arctan2(y, x) * 360 / (2 * np.pi)
    r = np.sqrt( x**2 + y**2 )
    z = i2
    
    return r, theta, z