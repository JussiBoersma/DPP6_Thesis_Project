import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as et
import csv
import sys
import os
import warnings
import neurokit2 as nk
import re 
from pandas._libs.missing import NA
import pywt
import pywt.data
import sys
from scipy import signal
from collections import Counter
from ecgdetectors import Detectors
import re
import statistics as st
detectors = Detectors(250)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# Extract and decode the signal from the XML file
def return_signal(root, signal_type):
    signal = []
    # Define the things we want to extract
    root2 = root.findall('Waveform')
    root3 = root.find('TestDemographics')
    location = root3.find('Location')

    # extracting the ECG wave values
    root4 = root.find('RestingECGMeasurements')
    measures = []
    # Extract the meta data from the ECG
    try:
        measures.append(int(root4.find('VentricularRate').text))
        measures.append(int(root4.find('PRInterval').text))
        measures.append(int(root4.find('QRSDuration').text))
        measures.append(int(root4.find('QTInterval').text))
        measures.append(int(root4.find('QTCorrected').text))
        measures.append(int(root4.find('PAxis').text))
        measures.append(int(root4.find('RAxis').text))
        measures.append(int(root4.find('TAxis').text))
        measures.append(int(root4.find('QRSCount').text))
        measures.append(int(root4.find('QOnset').text))
        measures.append(int(root4.find('QOffset').text))
        measures.append(int(root4.find('POnset').text))
        measures.append(int(root4.find('POffset').text))
        measures.append(int(root4.find('TOffset').text))
    except:
        measures = 0

    # Define wheter extracting median or rhythm signal
    if signal_type == 'median':
        i = 0
    else:
        i = 1
    # Find all leads for the signal
    for x in root2[i].findall('LeadData'): #set this value i to 0 for median and 1 for rhythym
        try:
            amp = float(x.find('LeadAmplitudeUnitsPerBit').text)
        except:
            print('amp got broken')
            return 0, 0
        # Go over the found leads and decode them
        for y in x.findall('WaveFormData'):
            decoded_wave = base64.b64decode(y.text)
            decoded = np.frombuffer(decoded_wave, dtype='int16')
            signal.append(decoded*amp)
    # Return the signal, the location of the ECG and the metadata
    return signal, int(location.text), measures

# Function for extracting the XML data for the Male Female transfer learning dataset
def return_signal2(root, signal_type):
    signal = []
    root2 = root.findall('Waveform')
    gender = root.find('Gender')
    if signal_type == 'median':
        i = 0
    else:
        i = 1
    for x in root2[i].findall('LeadData'): #set this value i to 0 for median and 1 for rhythym
        try:
            amp = float(x.find('LeadAmplitudeUnitsPerBit').text)
        except:
            print('amp got broken')
            return 0, 0
        # for lead_name in x.findall('LeadID'):
        #     print(lead_name.text)
        for y in x.findall('WaveFormData'):
            decoded_wave = base64.b64decode(y.text)
            decoded = np.frombuffer(decoded_wave, dtype='int16')
            signal.append(decoded*amp)
    return signal, str(gender.text) 

# Calculate the 4 non given leads
def calculate_leads(signals):
    # I II V1 V2 V3 V4 V5 V6
    Three = signals[1] - signals[0]
    aVR = -1 * ((signals[0] + signals[1]) / 2)
    aVL = (signals[0] - Three) / 2
    aVF = (signals[1] + Three) / 2
    # III = II - I

    # aVR = -1 * ((I + II) / 2 )

    # aVL = (I - III) / 2

    # aVF = (II + III) / 2 
    ECGs = [Three, aVR, aVL, aVF]

    return ECGs

# For extracting the ECG's from XML to individual .csv files
def extract_ecg(dir, og_dir):
    # For getting a generalizable directory so it can run on other systems
    os.chdir(dir)  
    # This list are all the ECG rooms that we allow into the dataset. Other rooms than these can contain recusitated patients with IVF artifacts 
    loc_check = [66, 67, 69, 70, 71, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 104, 105 ,106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 123, 125, 126, 127, 128, 129, 130, 135, 139, 141, 142, 147, 149, 168, 171, 175, 180]
    # A list of patients with removed ecg's because of multiple reasons such as noise
    removed_ecgs = ['002', '003', '004', '008', '010', '014', '037', '058', '061', '093', '134', '145', '146', '158', '159', '173', '178', '204', '210', '214', '226', '230', '237', '260', '349', '366', '429', '431', '444', '498', '526', '539', '555', '587', '597', '600', '602', '611', '639', '651']
    measure_arr, measure_labels = [], []
    for classification in os.listdir(dir):
        types = ['rhythm', 'median']
        # Extract both the rhythm and the median signal
        for signal_type in types:
            number = ''
            status = 'init'
            dir2 = dir+'\\'+classification
            os.chdir(dir2)
            cnt = 0
            # Loop over all the patients XML files
            for patient in os.listdir(dir+'\\'+classification):
                # Make sure that only one ECG per patient is taken into the dataset
                if number != patient[5:8] or status == 'searching':
                    number = patient[5:8]
                    if patient.lower().endswith(".xml"):
                        tree = et.parse(patient)
                        root = tree.getroot()
                        # Run the signal extracting function which returns 12 leads
                        signal, location, measures = return_signal(root, signal_type)  # signal lead sequence: I, II, V1, V2, V3, V4, V5, V6
                        # Make sure the location is allowed and the ECG should not be removed
                        if location in loc_check and patient[5:8] not in removed_ecgs:
                            ECGs = calculate_leads(signal)
                            leads = []
                            leads.append(signal[0])
                            leads.append(signal[1])
                            leads.append(ECGs[0])
                            leads.append(ECGs[1])
                            leads.append(ECGs[2])
                            leads.append(ECGs[3])
                            leads.append(signal[2])
                            leads.append(signal[3])
                            leads.append(signal[4])
                            leads.append(signal[5])
                            leads.append(signal[6])
                            leads.append(signal[7])      
                            cnt += 1
                            # Save the individual patient csv file
                            np.savetxt(og_dir+'\\data\\CSV_files\\{0}\\{1}\\{2}.csv'.format(signal_type, classification+'_csv', patient[:-4]), leads, delimiter=',', fmt='% s')
                            status = 'found'                    # the final order I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
                            # For saving the metadata for the machine learning methods
                            if measures != 0:
                                measure_arr.append(measures)
                                if classification == 'negative':
                                    label = 0
                                else:
                                    label = 1
                                measure_labels.append(label)
                        else:
                            status = 'searching'
            print(cnt)
        # Save the metadata
        measure_tot = {'values': measure_arr, "label": measure_labels}
    df = pd.DataFrame(measure_tot)
    os.chdir(og_dir+'\data\meta_data')
    df.to_pickle('meta_data_ML')   

# Extract the ECGs for the transfer learning data
def transfer_learning_ecg_extraction():
     # For getting a generalizable directory so it can run on other systems
    dir = os.getcwd()
    dir = dir.replace('\Preprocessing', '')
    dir = (dir+'\\data\\XML_data')
    # dir = (dir+'D:\Master Thesis\Median_TransferLearning\muse\muse')
    os.chdir(dir)   
    unique_patients = []
    for patient in os.listdir(dir):
        patient_number = patient.rpartition('_')[0]
        if patient_number in unique_patients:
            continue
        else:
            unique_patients.append(patient_number)
        types = ['median']
        for signal_type in types:
            if patient.lower().endswith(".xml"):
                print(patient)
                tree = et.parse(patient)
                root = tree.getroot()
                signal, gender = return_signal2(root, signal_type)  # signal lead sequence: I, II, V1, V2, V3, V4, V5, V6

                ECGs = calculate_leads(signal)
                leads = []
                leads.append(signal[0])
                leads.append(signal[1])
                leads.append(ECGs[0])
                leads.append(ECGs[1])
                leads.append(ECGs[2])
                leads.append(ECGs[3])
                leads.append(signal[2])
                leads.append(signal[3])
                leads.append(signal[4])
                leads.append(signal[5])
                leads.append(signal[6])                 # the final order I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
                leads.append(signal[7])                        
                np.savetxt(r"D:\Master Thesis\Median_TransferLearning\{0}\{1}\{2}.csv".format(signal_type, gender+'_csv', patient[:-4]), leads, delimiter=',', fmt='% s')                

# Extract a single ECG
def single_ecg_extraction():
    dir = (r'C:\Users\jussi\Documents\Master Thesis\Data\XML_data_adjusted\positive')
    os.chdir(dir)
    arrt = ['rhythm', 'median']
    for signal_type in arrt:  
        classification = 'positive'
        for file in os.listdir(dir):
            tree = et.parse(file)
            root = tree.getroot()
            root2 = root.find('TestDemographics')
            acqdate = root2.find('AcquisitionDate')
            # print(file[5:8], acqdate.text)
            if file[5:8] == '342' and acqdate.text == '10-20-2010':
                signal, location = return_signal(root, signal_type)
                ECGs = calculate_leads(signal)
                leads = []
                leads.append(signal[0])
                leads.append(signal[1])
                leads.append(ECGs[0])
                leads.append(ECGs[1])
                leads.append(ECGs[2])
                leads.append(ECGs[3])
                leads.append(signal[2])
                leads.append(signal[3])
                leads.append(signal[4])
                leads.append(signal[5])
                leads.append(signal[6])
                leads.append(signal[7])                        
                np.savetxt(r"C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset\revised_patients\{0}\{1}\{2}.csv".format(signal_type, classification, file[:-4]), leads, delimiter=',', fmt='% s')

# Resample the 3beats for creating the Betti curve
def resample_3beats(df):
    # Declare some empty variables
    labels, patient_ID, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(14))
    for i in range(0, df.shape[0]):        
        # Declare some empty variables
        aThree, aaVR, aaVL, aaVF, aI, aII, aV1, aV2, aV3, aV4, aV5, aV6 = ([] for i in range(12))        
        for j in range(len(df.iloc[i, 0])):
            if not np.isnan(df.iloc[i, 0][j]):
                aI.append(df.iloc[i, 0][j])

        for j in range(len(df.iloc[i, 1])):
            if not np.isnan(df.iloc[i, 1][j]):
                aII.append(df.iloc[i, 1][j])

        for j in range(len(df.iloc[i, 2])):
            if not np.isnan(df.iloc[i, 2][j]):
                aThree.append(df.iloc[i, 2][j])

        for j in range(len(df.iloc[i, 3])):
            if not np.isnan(df.iloc[i, 3][j]):
                aaVR.append(df.iloc[i, 3][j])

        for j in range(len(df.iloc[i, 4])):
            if not np.isnan(df.iloc[i, 4][j]):
                aaVL.append(df.iloc[i, 4][j])

        for j in range(len(df.iloc[i, 5])):
            if not np.isnan(df.iloc[i, 5][j]):
                aaVF.append(df.iloc[i, 5][j])

        for j in range(len(df.iloc[i, 6])):
            if not np.isnan(df.iloc[i, 6][j]):
                aV1.append(df.iloc[i, 6][j])

        for j in range(len(df.iloc[i, 7])):
            if not np.isnan(df.iloc[i, 7][j]):
                aV2.append(df.iloc[i, 7][j])

        for j in range(len(df.iloc[i, 8])):
            if not np.isnan(df.iloc[i, 8][j]):
                aV3.append(df.iloc[i, 8][j])

        for j in range(len(df.iloc[i, 9])):
            if not np.isnan(df.iloc[i, 9][j]):
                aV4.append(df.iloc[i, 9][j])

        for j in range(len(df.iloc[i, 10])):
            if not np.isnan(df.iloc[i, 10][j]):
                aV5.append(df.iloc[i, 10][j])

        for j in range(len(df.iloc[i, 11])):
            if not np.isnan(df.iloc[i, 11][j]):
                aV6.append(df.iloc[i, 11][j])
        # Resample the signals to a size of 500
        aThree = signal.resample(aThree, 500)
        aaVR = signal.resample(aaVR, 500)
        aaVL = signal.resample(aaVL, 500)
        aaVF = signal.resample(aaVF, 500)
        aI = signal.resample(aI, 500)
        aII = signal.resample(aII, 500)
        aV1 = signal.resample(aV1, 500)
        aV2 = signal.resample(aV2, 500)
        aV3 = signal.resample(aV3, 500)
        aV4 = signal.resample(aV4, 500)
        aV5 = signal.resample(aV5, 500)
        aV6 = signal.resample(aV6, 500)
        Three.append(aThree)
        aVR.append(aaVR)
        aVL.append(aaVL)
        aVF.append(aaVF)
        I.append(aI)
        II.append(aII)
        V1.append(aV1)
        V2.append(aV2)
        V3.append(aV3)
        V4.append(aV4)
        V5.append(aV5)
        V6.append(aV6)
        if df.iloc[i, 12] == 0:
            labels.append(0)
        else:
            labels.append(1)
        # Save the data to a dataframe
        patient_ID.append(df.iloc[i, 13])
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels, 'patient_ID': patient_ID}
    df5 = pd.DataFrame(data)
    return(df5)

# An auxillary function for plotting 12 lead ECG from the csv files
def plot_12_leads():
    dir = os.getcwd()
    dir = dir.replace('\Preprocessing', '')
    dir = (dir+'\\data\\CSV_files\\median\\negative_csv')
    dir = dir+'\\data\\median'
    try:  
        os.chdir(dir)
    except OSError:
        print("Can't change the Current Working Directory") 
    i = 0
    for filename in os.listdir(dir):
        df = pd.read_csv(filename, header = None)  
        fig, axs = plt.subplots(nrows= 6, ncols= 2)

        axs[0,0].plot(df.iloc[0, :])
        axs[1,0].plot(df.iloc[1, :])
        axs[2,0].plot(df.iloc[2, :])
        axs[3,0].plot(df.iloc[3, :])
        axs[4,0].plot(df.iloc[4, :])
        axs[5,0].plot(df.iloc[5, :])
        axs[0,1].plot(df.iloc[6, :])
        axs[1,1].plot(df.iloc[7, :])
        axs[2,1].plot(df.iloc[8, :])
        axs[3,1].plot(df.iloc[9, :])
        axs[4,1].plot(df.iloc[10, :])
        axs[5,1].plot(df.iloc[11, :])
        axs[0,0].title.set_text('III')
        axs[1,0].title.set_text('aVR')
        axs[2,0].title.set_text('aVL')
        axs[3,0].title.set_text('aVF')
        axs[4,0].title.set_text('I')
        axs[5,0].title.set_text('II')
        axs[0,1].title.set_text('V1')
        axs[1,1].title.set_text('V2')
        axs[2,1].title.set_text('V3')
        axs[3,1].title.set_text('V4')
        axs[4,1].title.set_text('V5')
        axs[5,1].title.set_text('V6')
        name = filename[:-4]
        # Set this directory to where you want to save the plots
        plt.savefig(dir+'\simple_plots\positive\{0}_{1}.png'.format(name, i))
        i += 1

# Plot individual leads
def plot_12_300length_leads():
    dir = os.getcwd()
    dir = dir.replace('\Preprocessing', '')
    dir = (dir+'\\data\\CSV_files\\median\\negative_csv')
    dir = dir+'\\data\\median'
    os.chdir(dir)
    df = pd.read_pickle('3_beat_df')
    for i in range(int(df.shape[0])):
        for j in range(12):
            plt.plot(df.iloc[i,j])
            plt.show()

# Extract the 3 beats from the rhythm signal
def extract_3beats():
    dir = os.getcwd()
    dir = dir.replace('\Preprocessing', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)
    df = pd.read_pickle('rhythm_signals_train')
    filler_df = pd.DataFrame(index=range(len(df)),columns=range(14))
    prev = [0,0,0,0,0,0,0]
    for i in range(df.shape[0]):
        print(i)
        for j in range(12):
            # Detect the peaks
            try:
                xcoords1 = detectors.two_average_detector(df.iloc[i, j])
            except:
                print('ERROR line: ', print(i))
                print(df.iloc[i, 0])
                plt.plot(df.iloc[i, :])
                plt.show()

            # For showing the signal with the found peaks
            plt.plot(df.iloc[i,j])
            plt.scatter(xcoords1, df.iloc[i,j][xcoords1], c = 'r')
            plt.show()
            # Make sure there are enough peaks
            if(len(xcoords1) > 5):
                # Define the segmentation points
                middle = int(len(xcoords1)/2)
                point1 = xcoords1[middle-2] + ((xcoords1[middle-1] - xcoords1[middle-2])/2)
                point2 = xcoords1[middle+1] + ((xcoords1[middle+2] - xcoords1[middle+1])/2)
                # Segment the signal
                arr = df.iloc[i, j][int(point1):int(point2)]
                arr2 = arr.tolist()
                # Make sure the signal is of equal size
                if len(arr2) % 2 != 0:
                    arr2 = np.pad(arr2, (1, 0), constant_values=(arr2[0]))
                # Pad the signal
                distance = 1200-len(arr2)
                new_arr = np.pad(arr2, int(distance/2), constant_values=(arr2[0], arr2[len(arr2)-1]))
                filler_df.iloc[i, j] = new_arr
                prev = new_arr
            else:
                filler_df.iloc[i, j] = prev
                print('does this happen?')
        filler_df.iloc[i, 12] = df.iloc[i, 12]
        filler_df.iloc[i, 13] = df.iloc[i, 13]
    # Save the dataframe
    df2 = resample_3beats(filler_df)
    print(df2)
    os.chdir(dir+'\\data\\raw_dataframes')
    df2.to_pickle('3beats_train')

# Function for plotting the data from a dataframe
def plot_data_from_df():
    dir = os.getcwd()
    dir = dir.replace('\Preprocessing', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)
    df = pd.read_pickle('df_pwave_removed')
    for i in range(df.shape[0]):
        fig, axs = plt.subplots(nrows= 6, ncols= 2)
        axs[0,0].plot(df.iloc[i, 0])
        axs[1,0].plot(df.iloc[i, 1])
        axs[2,0].plot(df.iloc[i, 2])
        axs[3,0].plot(df.iloc[i, 3])
        axs[4,0].plot(df.iloc[i, 4])
        axs[5,0].plot(df.iloc[i, 5])
        axs[0,1].plot(df.iloc[i, 6])
        axs[1,1].plot(df.iloc[i, 7])
        axs[2,1].plot(df.iloc[i, 8])
        axs[3,1].plot(df.iloc[i, 9])
        axs[4,1].plot(df.iloc[i, 10])
        axs[5,1].plot(df.iloc[i, 11])
        axs[0,0].title.set_text('I')
        axs[1,0].title.set_text('II')
        axs[2,0].title.set_text('III')
        axs[3,0].title.set_text('aVR')
        axs[4,0].title.set_text('aVL')
        axs[5,0].title.set_text('aVF')
        axs[0,1].title.set_text('V1')
        axs[1,1].title.set_text('V2')
        axs[2,1].title.set_text('V3')
        axs[3,1].title.set_text('V4')
        axs[4,1].title.set_text('V5')
        axs[5,1].title.set_text('V6')
        plt.show()
        if df.iloc[i, 12] == 1:
            plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\plots_pwave_removed\positive\{0}.png'.format(i))
        else:
            plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\plots_pwave_removed\negative\{0}.png'.format(i))

if __name__ == "__main__":
    pass
    # transfer_learning_ecg_extraction()
    # extract_3beats()
    # plot_data_from_df()
    # plot_12_300length_leads()
    # extract_ecg()
    # single_ecg_extraction()
    # plot_12_leads()
    # plot_12_300length_leads()
