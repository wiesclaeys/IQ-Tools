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

from functions import *

import pwlf


# user inputs
path = 'C:\\Users\\wclaey6\\Data\\Electron Density\\Intevo\\raw_data\\'


name = '130kV Abdomen HD  3.0  B30s'

num_segments = 2      # number of linear segments
energy = 440

# read data
filename = path + name + '.csv'

data = pd.read_csv(filename)

densities = data['Density'].values
HUs = data['Mean'].values
materials = data['Material']
# stds = data['Standard Deviation'].values

attenuation_data = pd.read_csv('C:\\Users\\wclaey6\\Data\\Electron Density\\materials_attenuation.csv')

#%% read mu values from file


mus = []
for material in materials:
    mus.append(attenuation_data.loc[energy, material])
    


# fit the model to the data
model = pwlf.PiecewiseLinFit(HUs, mus)    # use the pwlf library to handle 
model.fit(num_segments)

# get the model parameters
betas = model.beta
breaks_x = model.fit_breaks

breaks_y, slopes, offsets = convert_beta_parameters(betas, breaks_x, num_segments)
    
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
plt.plot(HUs, mus, marker = '.', linestyle = 'None', label = 'data')
# the fit
plt.plot(x, y, '-', label = 'fit')
# lines indicating the line breaks
plt.vlines(breaks_x[1:-1], ymin = 0, ymax = 0.25, color = 'green', linestyle = 'dashed', label = 'breaks')

plt.xlabel("CT-number (HU)")
plt.ylabel("Density (g/ml)")
plt.title("Conversion curve")
# plt.ylim(0,2)
plt.grid(True)
plt.legend()
plt.show()


#%% Analyze the difference between the fit and the data

dev = mus - model.predict(HUs)

plt.plot(HUs, dev / mus * 100, '.')

plt.grid(True)
plt.title("Deviation fit - data")
plt.xlabel("CT-number (HU)")
plt.ylabel("Deviation (%)")
plt.show()

print("----- Deviation -----")
print("Maximum deviation = ", round(np.max(np.abs(dev / mus * 100)),2), "%")
