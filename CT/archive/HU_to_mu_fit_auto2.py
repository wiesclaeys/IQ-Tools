# -*- coding: utf-8 -*-
"""
Fit a piecewise linear function density-HU data obtained by scanning a phantom containing inserts of varying density.


This script uses the pwlf package (https://pypi.org/project/pwlf/) to handle the fitting.
This is needed to since the choice of break points cannot be made using standard non-linear least squares fitting (e.g. scipy).

This version automatically performs the analysis for a whole folder

created on 12/08/2024

"""

import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

import pwlf

from functions import *

# user inputs
path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\raw_data\\'
save_path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\'

save_dir = 'results_guess_15\\'


num_segments = 2      # number of linear segments

breaks = [15]           # None or list containing the known or guessed break points
fixed_breaks = False   # if true, the exact positions of the breaks are kept, if False they are determined during fitting

average_solid_water = False

save = True             # whether or not to save the image data
verbose = True         # whether or not to print all results to the screen



# check for existing folder of the same name
new_path = save_path + save_dir
if save == True and not os.path.exists(new_path):
    os.makedirs(new_path)
else:
    print("WARNING: ")
    print("This folder already exists, the data won't be saved")
    print("location = ", save_path)
    print("folder name = ", save_dir)
    c = input("Continue?")
    save = False    
    

# if save == True:
#     results_file = save_path + save_name + ".csv"
#     if os.path.isfile(results_file):
#         print("WARNING: ")
#         print("This file already exists, please choose another filename or location")
#         print("location = ", save_path)
#         print("filename = ", save_name)
    

names = os.listdir(path)
file_names = [name.split('.c')[0] for name in names if os.path.isfile(path + name)]
lines = []
for name in file_names:
    # read data
    filename = path + name + ".csv"
    print(name)
    
    data = pd.read_csv(filename)
    
    densities = data['Density'].values
    HUs = data['Mean'].values
    materials = data['Material'].values
    
    # get duplicated solid water results
    inds, = np.where(materials == "solid water" )
    indsC, = np.where(materials != "solid water")
    
    M = np.mean(HUs[inds])
    std = np.std(HUs[inds], ddof = 1)
    
    print('--- Solid water data ---')
    print('mean = ', M)
    print('std = ', std)
    
    if average_solid_water:
        HUs = np.append(HUs[indsC], M)
        densities = np.append(densities[indsC], densities[inds[0]])    
    # stds = data['Standard Deviation'].values
    
    # fit the model to the data
    model = pwlf.PiecewiseLinFit(HUs, densities)    # use the pwlf library to handle the fitting
    
    if breaks == None:
        model.fit(num_segments)
    elif fixed_breaks:
        breaks2 = [-1024] + breaks + [3071]
        model.fit_with_breaks(breaks2)
    else:
        model.fit_guess(breaks)
    
    # get the model parameters
    betas = model.beta
    breaks_x = model.fit_breaks
    # if fixed_breaks == True and not breaks == None:
    #     breaks_x = [-1024] + breaks + [3071]
    
    

    breaks_y, slopes, offsets = convert_beta_parameters(betas, breaks_x, num_segments=num_segments)

    if verbose:
        print("----- Results -----")
        
        for i in range(num_segments):
            print((round(breaks_x[i],1), round(breaks_y[i], 3)))
            print("slope = ", round(slopes[i],5))
            print("offset = ", round(offsets[i],5))
        print((breaks_x[i+1], round(breaks_y[i+1], 3)))
        
        
        # Doing some checks and printing the values
        rho_air = model.predict(-1000)[0]
        rho_water = model.predict(0)[0]
        print("----- Checks -----")
        print("Density at -1000 HU (air):", round(rho_air, 2), "g/ml")
        print("Density at 0 HU (water):", round(rho_water, 2), "g/ml")



    #%% Plotting and the results
    # plot data and fit
    x = np.linspace(-1024,1500,4096)
    y = model.predict(x)
    
    # the raw data
    plt.plot(HUs, densities, marker = '.', linestyle = 'None', label = 'data')
    # the fit
    plt.plot(x, y, '-', label = 'fit')
    # lines indicating the line breaks
    plt.vlines(breaks_x[1:-1], ymin = 0, ymax = 2, color = 'green', linestyle = 'dashed', label = 'breaks')
    
    plt.xlabel("CT-number (HU)")
    plt.ylabel("Density (g/ml)")
    plt.title(name)
    plt.ylim(0,2)
    plt.grid(True)
    plt.legend()
    if save == True:
        plt.savefig(new_path + name + ".png")
    plt.show()
    

#%% Analyze the difference between the fit and the data

    dev = densities - model.predict(HUs)
    max_dev = np.max(np.abs(dev / densities * 100))
    
    plt.plot(HUs, dev / densities * 100, '.')
    
    plt.grid(True)
    plt.title(name)
    plt.xlabel("CT-number (HU)")
    plt.ylabel("Deviation (%)")
    plt.show()
    
    if verbose:
        print("----- Deviation -----")
        print("Maximum deviation = ", round(max_dev,2), "%")

    
    line = [name.strip(".csv")]
    line.append(M)
    line.append(std)
    for br in range(len(breaks_x)):
        line.append(breaks_x[br])
        line.append(breaks_y[br])
    for segm in range(num_segments):
        line.append(slopes[segm])
        line.append(offsets[segm])
    line.append(rho_air)
    line.append(rho_water)
    line.append(max_dev)

    lines.append(line)

# saving the results to a csv file
if save == True:
    f = open(new_path + "analysis.csv", "w")
    header = "name," + "solid water mean HU," + "solid water std HU," + "x,y," * (num_segments + 1) + "slope, offset, " * num_segments + "density at -1000 HU (g/ml)," + "density at 0 HU (g/ml)," + "max deviation (%)" 
    header = header.strip(",") + "\n"
    f.write(header)
    for line in lines:
        string = ""
        for word in line:
            string = string + str(word) + ","
        string = string.strip(",") + "\n"
        f.write(string)
    f.close()
