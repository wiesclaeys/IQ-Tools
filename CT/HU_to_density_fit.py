# -*- coding: utf-8 -*-
"""
Fit a piecewise linear function density-HU data obtained by scanning a phantom containing inserts of varying density.


This script uses the pwlf package (https://pypi.org/project/pwlf/) to handle the fitting.
This is needed to since the choice of break points cannot be made using standard non-linear least squares fitting (e.g. scipy).

There are three possible fitting modes:
    - free: the break points are completely determined by the fitting algorithm
    - guess: initial break point positions are specified, the fitting starts from here
    - fixed: break points are fixed from the start


created on 12/08/2024

"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pwlf

from functions import *


# =============================================================================
# # User inputs
# =============================================================================
# file locations
data_path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\raw_data\\'    # directory from where the raw data is read
save_path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\'              # folder where the results directory is created

save_dir = 'results_guess_15_av'                                                   # name of the results directory

# fit settings
num_segments = 2       # Number of linear segments

breaks = [15]          # List of the guessed or knonw break points between the linear segments. If None, the breaks points are determined completely during fitting
fixed_breaks = False   # If true, the exact positions of the breaks are kept, if False they are optimized during fitting

# options
average_solid_water = True     # whether or not to average the solid water data before fitting
save = True            # whether or not to save the results (graphs and csv)
verbose = True         # whether or not to print the results to the screen


# =============================================================================
# End of user inputs
# =============================================================================



#%% Making sure no data is lost:
    # if the selected folder already exists, don't save
    # print warning if the data will not be saved
    
new_path = save_path + save_dir + '\\'    # path where the data will be saved

if save == True and not os.path.exists(new_path):
    os.makedirs(new_path)   # create new directory for the results
elif save ==True:
    print("WARNING: ")
    print("This folder already exists, the results won't be saved")
    print("location = ", save_path)
    print("folder name = ", save_dir)
    c = input("Press enter to continue")
    save = False
else:
    print("WARNING: ")
    print("The results will not be saved")
    c = input("Press enter to continue")


#%% Performing the fit

lines = []  # initialize list from where the results file is written

# get all files inside the data folder
names = os.listdir(data_path)   # get all files in folder
file_names = [name.split('.csv')[0] for name in names if os.path.isfile(data_path + name)]

# perform the analysis file by file
for name in file_names:
    
    # read data
    filename = data_path + name + ".csv"    
    data = pd.read_csv(filename)
    
    densities = data['Density'].values
    HUs = data['Mean'].values
    materials = data['Material'].values
    
    # average solid water results (optional)
    if average_solid_water:
        inds, = np.where(materials == "solid water" )
        indsC, = np.where(materials != "solid water")
        
        M = np.mean(HUs[inds])
        
        HUs = np.append(HUs[indsC], M)
        densities = np.append(densities[indsC], densities[inds[0]])  
    
    # initialize the fitting model
    model = pwlf.PiecewiseLinFit(HUs, densities)    # use the pwlf library to handle the fitting
    
    # perform the fit in the correct mode:
    # free mode mode
    if breaks == None:  
        model.fit(num_segments)
    # fixed mode
    elif fixed_breaks:
        breaks2 = [-1024] + breaks + [3071] # endpoints need to be included for this fitting mode
        model.fit_with_breaks(breaks2)
    # guess mode
    else:       
        model.fit_guess(breaks)
    
    # get the model parameters
    betas = model.beta
    breaks_x = model.fit_breaks


    # convert the model parameters
    breaks_y, slopes, offsets = convert_beta_parameters(betas, breaks_x, num_segments=num_segments)

    # Printing the results (optional)
    if verbose:
        print("----- Results -----")
        print(name)
        print('-' * len(name))

        # Printing the break points, slopes and offsets
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


    #%% Plotting the results
    
    x = np.linspace(-1024,1500,4096)
    y = model.predict(x)
    
    # the raw data
    plt.plot(HUs, densities, marker = '.', linestyle = 'None', label = 'data')
    # the fit
    plt.plot(x, y, '-', label = 'fit')
    # lines indicating the line breaks
    plt.vlines(breaks_x[1:-1], ymin = 0, ymax = 2, color = 'green', linestyle = 'dashed', label = 'breaks')
    
    # layout
    plt.xlabel("CT-number (HU)")
    plt.ylabel("Density (g/ml)")
    plt.title(name)
    plt.ylim(0,2)
    plt.grid(True)
    plt.legend()
    if save == True:    # saving the plot (optional)
        plt.savefig(new_path + name + ".png")
    plt.show()
    

    #%% Analyzing the difference between the fit and the data

    # calculating
    dev = densities - model.predict(HUs)            # calculate difference between data and fit
    max_dev = np.max(np.abs(dev / densities * 100)) # maximum deviation (%)
    
    # plotting
    plt.plot(HUs, dev / densities * 100, '.')
    
    # layout
    plt.grid(True)
    plt.title(name)
    plt.xlabel("CT-number (HU)")
    plt.ylabel("Deviation (%)")
    
    # saving the plots (optional)
    if save == True:
        plt.savefig(new_path + name + "_deviation.png")
    plt.show()
    
    
    # printing the results (optional)
    if verbose:
        print("----- Deviation -----")
        print("Maximum deviation = ", round(max_dev,2), "%")

    #%% Processing the results for saving to file
    line = [name.strip(".csv")]
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

#%% Saving the results to a csv file

if save == True:
    f = open(new_path + "analysis.csv", "w")
    header = "name," + "x,y," * (num_segments + 1) + "slope, offset, " * num_segments + "density at -1000 HU (g/ml)," + "density at 0 HU (g/ml)," + "max deviation (%)" 
    header = header.strip(",") + "\n"
    f.write(header)
    for line in lines:
        string = ""
        for word in line:
            string = string + str(word) + ","
        string = string.strip(",") + "\n"
        f.write(string)
    f.close()
