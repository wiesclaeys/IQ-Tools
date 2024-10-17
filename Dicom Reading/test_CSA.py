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


fragment = b"x80?[R\x82?\xa0\xeb\x82?\x1c\x1b\x85?\x13\xb1\x85?\x81\x9f\x83?\xfb"


# detected = chardet.detect(fragment)
# decoded = raw.decode(detected["encoding"])

# raw.decode("utf-32")
# a = np.array(dcm[0x0029,0x1010].value, dtype = np.float32)


codecs = [
    "ascii", "big5", "big5hkscs", "cp037", "cp273", "cp424", "cp437", "cp500", "cp720", 
    "cp737", "cp775", "cp850", "cp852", "cp855", "cp856", "cp857", "cp858", "cp860",
    "cp861", "cp862", "cp863", "cp864", "cp865", "cp866", "cp869", "cp874", "cp875",
    "cp932", "cp949", "cp950", "cp1006", "cp1026", "cp1125", "cp1140", "cp1250",
    "cp1251", "cp1252", "cp1253", "cp1254", "cp1255", "cp1256", "cp1257",
    "cp1258", "cp65001", "euc_jp", "euc_jis_2004", "euc_jisx0213", "euc_kr", "gb2312",
    "gbk", "gb18030", "hz", "iso2022_jp", "iso2022_jp_1", "iso2022_jp_2",
    "iso2022_jp_2004", "iso2022_jp_3", "iso2022_jp_ext", "iso2022_kr", "latin_1",
    "iso8859_2", "iso8859_3", "iso8859_4", "iso8859_5", "iso8859_6", "iso8859_7",
    "iso8859_8", "iso8859_9", "iso8859_10", "iso8859_11", "iso8859_13", "iso8859_14",
    "iso8859_15", "iso8859_16", "johab", "koi8_r", "koi8_t", "koi8_u", "kz1048",
    "mac_cyrillic", "mac_greek", "mac_iceland", "mac_latin2", "mac_roman",
    "mac_turkish", "ptcp154", "shift_jis", "shift_jis_2004", "shift_jisx0213",
    "utf_32", "utf_32_be", "utf_32_le", "utf_16", "utf_16_be", "utf_16_le", "utf_7",
    "utf_8", "utf_8_sig",
]

data = fragment

codec = "iso8859_1"

for codec in codecs:
    try:
        print(f"{codec}, {data.decode(codec)}")
    except UnicodeDecodeError:
        continue


#%%



