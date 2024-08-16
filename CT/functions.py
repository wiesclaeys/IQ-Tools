# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:18:31 2024

@author: wclaey6
"""

def convert_beta_parameters(betas, breaks_x)
"""the variabels used by pwlf are
       - b parameters: indicating the x position of the transition point from each linear segment to the next
       - beta parameters: beta1 is the y-coordinate of the first break point
                           all other betas indicate the slope of the segments
    More precisely:
    y_1(x) = beta1 + beta2 (x - b1)
    y_n(x) = y(b_n) + beta_(n+1) (x - b_n)
    Note that the number of breaks is one larger than the number of segments (both end points are included)
    See also: https://jekel.me/2018/Continous-piecewise-linear-regression/
    
    This function converts the beta parameters to slopes a and offsets b for each segment"""

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
    return breaks_y, slopes, offsets