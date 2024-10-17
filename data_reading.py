# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:08:37 2024

@author: wclaey6


This code is meant to provide an easy means to read DICOM files exprted by MIM
"""

import os
import pydicom

import numpy as np
import pandas as pd


specifiers = ["patient_name", "patient_ID", "modality", "date", "time", "study_description", "series_description", "r1", "r2", "r3"]

def create_database(path):
    " create a pandas dataframe to easily look up exported MIM data"
    files = []
    names = []
    folders = os.listdir(path)
    
    for folder in folders:
        series = os.listdir(path + "\\" + folder)
        
        for serie in series:
            names.append(serie)
            files.append(path + "\\" + folder + "\\" + serie)
            
    info = []   
    for name in names:
        words = name.split("_")
        words = [process(word) for word in words]
        info.append(words)
        
        # patient_name, patient_ID, modality, date, time, study_description, series_description, r1, r2, r3 = name.split("_")
    
    
    info = pd.DataFrame(np.array(info), columns = specifiers)
    paths = pd.DataFrame(np.array(files), columns = ["full_path"])
    
    database = info.join(paths)
    
    return database

def process(name):
    " recreate the way MIM processes strings before using them as a filename "
    news = ["_", "[", "]"]
    olds = [".", "(", ")"]
    for i in range(len(olds)):
        name = name.replace(olds[i], news[i])
    return name

def lookup_series(database, series_description, patient_name = None, patient_ID = None):
    # "deprecated"
    
    subset = database
    if patient_name is not None:
        patient_name = process(patient_name)
        subset = subset.loc[subset['patient_name'] == patient_name]
        if len(subset) == 0:
            print("Patient name", patient_name, " not found")
            return
        
    if patient_ID is not None:
        patient_ID = process(patient_ID)
        subset = subset[subset['patient_ID'] == patient_ID]
        if len(subset) == 0:
            print("Patient ID not found")
            return
    
    series_description = process(series_description)
    series = subset.loc[subset['series_description'].str.contains(series_description)]   
    
    print(len(series), "series found satisfying the requirements")
    return series

def find_series(patient, series_description, modality = None, date = None, time = None, study_description = None, num_images = None):
    
    tags = {"modality" : modality, "date" : date, "time" : time, "study_description" : study_description, "r1" : num_images}
    
    
    subset = patient
    
    for tag in tags:
        if tags[tag] is not None:
            val = process(tags[tag])
            subset = subset.loc[subset[tag] == val]
            if len(subset) == 0:
                print("No series found with ", tag, val)
                return

    
    series_description = process(series_description)
    series = subset.loc[subset['series_description'].str.contains(series_description)]   
    
    print(len(series), "series found satisfying the requirements")
    return series

def load_series(series):
    dcms = []
    print("Loading", len(series), "series:")
    for i in range(len(series)):
        print(i, "-", series['series_description'].values[i])
        path = series['full_path'].values[i]
        dcm = pydicom.dcmread(path + "\\" + os.listdir(path)[0])
        dcms.append(dcm)
    return dcms

def lookup_patient(database, patient_name, patient_ID = None):
    """
    Look up a specific patient based on patient name and ID

    Parameters
    ----------
    database : TYPE
        DESCRIPTION.
    patient_name : TYPE
        DESCRIPTION.
    patient_ID : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    subset : pandas dataframe
        DESCRIPTION.

    """
    
    if patient_ID is not None:
        print("Looking for patient", patient_name, "with patient ID", patient_ID)
    else:
        print("Looking for patient", patient_name)
    
    subset = database
    patient_name = process(patient_name)
    subset = subset.loc[subset['patient_name'] == patient_name]
    if len(subset) == 0:
        print("Patient name", patient_name, " not found")
        return
        
    if patient_ID is not None:
        patient_ID = process(patient_ID)
        subset = subset[subset['patient_ID'] == patient_ID]
        if len(subset) == 0:
            print("Patient ID", patient_ID, " not found")
            return
    
    
    print(len(subset), "series found corresponding to this patient")
    return subset
    
def list_patients(database):
    """
    List all unique patient names and IDs in the current database

    Parameters
    ----------
    database : pandas dataframe
        the collection of series to consider

    Returns
    -------
    None.

    """    
    patient_names = database.patient_name.unique()
    print("--- Listing all patients ---")
    for patient_name in patient_names:
        patient = database.loc[database['patient_name'] == patient_name]
        patient_IDs = patient.patient_ID.unique()
        for patient_ID in patient_IDs:
            print("Name: ", patient_name, " - ID: ",  patient_ID)
    print("-----------------------------")
    return


def get_rescaled_data(dcm):
    # reading the raw data
    vol         = dcm.pixel_array 


    # try to find a rescale factor
    try:    # MIM rescale
        field = dcm[0x0040,0x9096][0]
        rescale = field[0x0040,0x9225].value
    except:    
        try:    
            # Siemens rescale
            rescale = dcm[0x0033,0x1038].value
        except:
            rescale = 1
    
    # rescale the data
    vol = vol * rescale
    
    return vol

    
# path = "C:\\Wies Data\\Data for Python"
# # path = "C:\\Wies Data\\MIM Local Data\EARL Data"

# database = create_database(path)

# patient_name = "QCintevo_WC"
# patient_ID = "Noise Levels"
# series_description = "30 minutes"

# s = lookup_series(database, series_description, patient_name=patient_name, patient_ID=patient_ID)






