# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:27:21 2024

@author: wclaey6
"""

import pydicom as pd
import dicom_parser as parser
from dicom_parser import Image

import matplotlib.pyplot as plt
import numpy as np

import chardet


path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\'
name = '100.dcm'



image = Image(path + name)

print(image.header)


header = image.header


#%%

csa = image.header.get("DetectorInformationSequence")




#%%

from dicom_parser import Image

image = Image(path + name)

print(image.header)
csa = image.header #.raw[0x0029,0x1010]
print(csa)

# image.header.get('NumberOfViews')


# csa.value

# plt.plot(csa.value)


#%%


other_path = "C:\\Users\\wclaey6\\Data\\Noise Generator\\2024-09__Studies\\20\\test.dcm"
other_path = "C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\SUV\\100.dcm"

image = Image(other_path)


from dicom_parser.utils.siemens.csa.header import CsaHeader
raw_csa = image.header.get("CSASeriesHeaderInfo", parsed=False)
type(raw_csa)



dcm = pd.dcmread(other_path)
im = dcm.pixel_array

plt.imshow(im[63,:,:])


b = dcm[0x0008,0x0021].value

raw = dcm[0x0029,0x1010].value

detected = chardet.detect(raw)
decoded = data.decode(detected["encoding"])

raw.decode("utf-32")
# a = np.array(dcm[0x0029,0x1010].value, dtype = np.float32)




#%%
third_path = 'C:\\Users\\wclaey6\\Data\\Noise Generator\\QCintevo_WC\\NEMA\\2024-03__Studies\\LEHR_BG0\\a.dcm'

image = Image(third_path)
header = image.header
iop = image.header.get("ImageOrientationPatient")

aff = image.affine

arr = image.data

plt.imshow(arr[120,:,:])


raw = image.raw


