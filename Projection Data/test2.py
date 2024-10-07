# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:33:31 2024

@author: wclaey6
"""

# import pydicom as pd
import dicom_parser as parser
from dicom_parser import Image

import os

from IQ_functions import *


path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\PP\\'
name = '100'
# path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_Tb\\'
# name = '2024-07-03'

dcm_dir     = path + name
# dcm         = pd.dcmread(dcm_dir + "\\" + os.listdir(dcm_dir)[0])
# vol         = dcm.pixel_array 

# if name == '100':
#     rescale = 1
# else:
#     field = dcm[0x0040,0x9096][0]
#     rescale = field[0x0040,0x9225].value
# vol = vol * rescale


# # frame       = vol[frame_index, :, :]

# pixel_size = dcm.PixelSpacing[0]
# num_rows = dcm.Rows
# num_columns = dcm.Columns

#%%

# field = dcm[0x0029,0x1010]
# arr = field._convert_value


# field2 = dcm[0x0029,0x1140][0]

path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\100.dcm'

image = Image(path)
