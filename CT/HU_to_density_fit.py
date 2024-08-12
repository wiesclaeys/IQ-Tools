# -*- coding: utf-8 -*-
"""
Fit a piecewise linear function density-HU data obtained by scanning a phantom containing inserts of varying density.


This script uses the pwlf package (https://pypi.org/project/pwlf/) to handle the fitting.
This is needed to since the choice of break points cannot be made using standard non-linear least squares fitting (e.g. scipy).


created on 12/08/2024

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

import pwlf


# user inputs
path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\'
name = '130kV Abdomen HD  3.0  B30s'

num_segments = 2      # number of linear segments


# read data
filename = path + name + '.csv'

data = pd.read_csv(filename)

densities = data['Density'].values
HUs = data['Mean'].values
# stds = data['Standard Deviation'].values


# fit the model to the data
model = pwlf.PiecewiseLinFit(HUs, densities)    # use the pwlf library to handle 
model.fit(num_segments)

# get the model parameters
betas = model.beta
breaks_x = model.fit_breaks

#%% convert beta parameters 
# the variabels used by pwlf are
#   - b parameters: indicating the x position of the transition point from each linear segment to the next
#   - beta parameters: beta1 is the y-coordinate of the first break point
#                       all other betas indicate the slope of the segments
# More precisely:
# y_1(x) = beta1 + beta2 (x - b1)
# y_n(x) = y(b_n) + beta_(n+1) (x - b_n)
# Note that the number of breaks is one larger than the number of segments (both end points are included)
# See also: https://jekel.me/2018/Continous-piecewise-linear-regression/
#
# This block converts the beta parameters to slopes a and offsets b for each segment

slopes = []
offsets = []
breaks_y = [betas[0]]

next_break = betas[0]
current_slope = 0
current_offset = betas[0]

for i in range(num_segments):
    current_slope += betas[i+1]
    current_offset -= breaks_x[i] * betas[i+1]
    next_break += current_slope * (breaks_x[i+1] - breaks_x[i])
    
    breaks_y.append(next_break)
    slopes.append(current_slope)
    offsets.append(current_offset)
    
#%% Printing some results
print("----- Results -----")

for i in range(num_segments):
    print((round(breaks_x[i],1), round(breaks_y[i], 3)))
    print("slope = ", round(slopes[i],5))
    print("offset = ", round(offsets[i],5))
print((breaks_x[i+1], round(breaks_y[i+1], 3)))


# Printing some values
print("----- Checks -----")
print("Density at -1000 HU (air):", round(model.predict(-1000)[0], 2), "g/ml")
print("Density at 0 HU (water):", round(model.predict(0)[0], 2), "g/ml")


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
plt.title("Conversion curve")
plt.ylim(0,2)
plt.grid(True)
plt.legend()
plt.show()


#%% Analyze the difference between the fit and the data

dev = densities - model.predict(HUs)

plt.plot(HUs, dev / densities * 100, '.')

plt.grid(True)
plt.title("Deviation fit - data")
plt.xlabel("CT-number (HU)")
plt.ylabel("Deviation (%)")
plt.show()

print("----- Deviation -----")
print("Maximum deviation = ", round(np.max(np.abs(dev / densities * 100)),2), "%")
