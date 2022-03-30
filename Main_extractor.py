import csv
import os
import numpy as np
from Preprocessing.decoding import return_signal, calculate_leads, extract_ecg
from Augmentation.Augmentation import create_dataframes, resample_signal, extra_Pwave_removal

og_dir = os.getcwd()
xml_dir = og_dir+'\\data\\XML_data'

# This function is the data extraction pipeline for the more standard networks
# Extract the ECGs and place each patient ECG in a csv file
extract_ecg(xml_dir, og_dir)
# csv_dir = og_dir+'\\data\\CSV_files'

# Go from csv files to one pandas dataframe
# create_dataframes(csv_dir, og_dir)

# Remove extra p and t waves from the ECG
extra_Pwave_removal()
