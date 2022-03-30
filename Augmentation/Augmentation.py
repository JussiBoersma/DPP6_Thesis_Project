import csv
from math import nan
import sys
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, kaiserord, lfilter, firwin, freqz, butter, filtfilt, lfilter_zi
from scipy.signal import savgol_filter
import heartpy as hp
import random
import shutil
import re
import random

# Create a dataframe from the CSV files
def create_dataframes(dir, og_dir):
    classification_arr = ['median', 'rhythm']
    for classification in classification_arr:
        # For getting a generalizable directory so it can run on other systems
        # og_dir = os.getcwd()
        # dir = og_dir.replace('\Augmentation', '')
        # dir = dir+'\\data\\CSV_files\\median'
        os.chdir(dir+'\\'+classification)  
        data_array = []
        file_array = []
        df_neg = pd.DataFrame()
        df_pos = pd.DataFrame()
        # Set the sizes depending on the signal types
        if 'median' in classification:
            x = 300
            name = 'median_df'
        else:
            x = 2500
            name = 'rhythm_df'
        # Loop over the negative csv's
        for file in os.listdir(dir+'\\'+classification+'\\'+'negative_csv'):
            file_array.append(file)
            os.chdir(dir+'\\'+classification+'\\'+'negative_csv')
            df = pd.read_csv(file, header = None)
            if df.shape[1] > x:
                # Some singals are double the size but the same signal, these need to be resampled
                df = resample_signal(df, x)
            df_neg = pd.concat([df_neg, df])
        # Loop over the positive csv's
        for file in os.listdir(dir+'\\'+classification+'\\'+'positive_csv'):
            file_array.append(file)
            os.chdir(dir+'\\'+classification+'\\'+'positive_csv')
            df = pd.read_csv(file, header = None)
            if df.shape[1] > x:
                df = resample_signal(df, x)
            df_pos = pd.concat([df_pos, df])
        # When its a rhythm signal also save the pos and neg dataframes seperately
        if x == 2500:
            os.chdir(og_dir+'\\data\\raw_dataframes')
            df_pos.to_csv('pos_rhythm_df.csv', index = False, header = False)
            df_neg.to_csv('neg_rhythm_df.csv', index = False, header = False)
        # Normalize the data between a range of -1 and 1
        neg_data = normalize_data2(df_neg)
        pos_data = normalize_data2(df_pos)
        # Creat a combined neg and pos dataframe
        df = create_pandas_dataframe(neg_data, pos_data, file_array) 
        os.chdir(og_dir+'\\data\\raw_dataframes')
        df.to_pickle(name)

# Create a dataframe for the male female transfer learning data
def create_dataframes_male_female(path):
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\CSV_files\\median'
    os.chdir(dir)
    data_array = []
    file_array = []
    os.chdir(path)
    df_male = pd.DataFrame()
    df_female = pd.DataFrame()

    if 'median' in path:
        x = 300
        name = 'median_10k_df'
    else:
        x = 2500
        name = 'rhythm_df'

    for file in os.listdir(path+'\\'+'MALE_csv'):
        file_array.append(file)
        os.chdir(path+'\\'+'MALE_csv')
        df = pd.read_csv(file, header = None)
        if df.shape[1] > x:
            df = resample_signal(df, x)
        df_male = pd.concat([df_male, df])
    for file in os.listdir(path+'\\'+'FEMALE_csv'):
        file_array.append(file)
        os.chdir(path+'\\'+'FEMALE_csv')
        df = pd.read_csv(file, header = None)
        if df.shape[1] > x:
            df = resample_signal(df, x)
        df_female = pd.concat([df_female, df])

    male_data = normalize_data2(df_male)
    female_data = normalize_data2(df_female)

    df = create_pandas_dataframe(male_data, female_data, file_array) 
    os.chdir(path)
    df.to_pickle(name)

# Create a subset of the transfer learning data
def subset_transferlearning_data():
    setje = ['MALE_all_csv', 'FEMALE_all_csv']
    for gender in setje:
        original = r'D:\Master Thesis\Median_TransferLearning\median\{0}'.format(gender)
        wordje = gender.rpartition('_')[0]
        wordje2 = wordje.rpartition('_')[0]
        target = r'D:\Master Thesis\Median_TransferLearning\median\{0}_csv'.format(wordje2)
        i = 0
        for file in os.listdir(original):
            i += 1
            shutil.copyfile(original+"\\"+file, target+"\\"+file)
            if i == 10000:
                break      

# Remove CSV files for transfer learning with nan values
def remove_nan_transferlearning_data():
    os.chdir(r'D:\Master Thesis\Median_TransferLearning\median\FEMALE_all_csv')
    for file in os.listdir(r'D:\Master Thesis\Median_TransferLearning\median\FEMALE_all_csv'):
        df = pd.read_csv(file, header = None)
        for i in range(12):
            if sum(df.iloc[i,:]) == 0:
                # print(file)
                try: 
                    os.remove(r'D:\Master Thesis\Median_TransferLearning\median\FEMALE_all_csv' + "\\" + file)
                except:
                    print(file)

# Resamples a signal to the desired size
def resample_signal(data, sample_size):
    df = pd.DataFrame(0, index=np.arange(12), columns=np.arange(sample_size))
    for i in range(data.shape[0]): 
        lead = data.iloc[i, :]
        resampled_lead = signal.resample(lead, sample_size)
        for j in range(sample_size):
            df.iloc[i, j] = resampled_lead[j]
    return df

# Normalize signals between -1 and 1
def normalize_data2(data):
    print('Normalizing the data')
    for i in range(data.shape[0]):
        max_Val = max(data.iloc[i,:])
        min_Val = min(data.iloc[i,:])
        # print(i)
        for j in range(data.shape[1]):
            try:
                data.iloc[i,j] = data.iloc[i,j]/(max_Val-min_Val)
            except:
                print('ERRORTJE: ', data.iloc[i,j])
    return data.to_numpy()

# Go from a pos and neg df to a single dataframe with patient IDs
def create_pandas_dataframe(neg_data, pos_data, files):  
    if files == 0:
        files = list(range(0, len(pos_data)+len(neg_data)))
    # Declare some empty variables
    labels, patient, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(14))
    j, idx = 0, 0
    data = neg_data
    for i in range(int((len(neg_data)+len(pos_data))/12)):
        if i == len(neg_data)/12:
            data = pos_data
            j = 0
        I.append(data[j].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(data[j+1].astype('float32'))
        Three.append(data[j+2].astype('float32'))
        aVR.append(data[j+3].astype('float32'))
        aVL.append(data[j+4].astype('float32'))
        aVF.append(data[j+5].astype('float32'))
        V1.append(data[j+6].astype('float32'))
        V2.append(data[j+7].astype('float32'))
        V3.append(data[j+8].astype('float32'))
        V4.append(data[j+9].astype('float32'))
        V5.append(data[j+10].astype('float32'))
        V6.append(data[j+11].astype('float32'))
        j += 12
        # Give the correct label
        if i < len(neg_data)/12:
            labels.append(0) #negative = male = 0                                                
        else:
            labels.append(1) #positive = female = 1
        patient.append(files[idx])
        idx += 1     
    # Save the dataframe   
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels, 'patient': patient}
    df = pd.DataFrame(data)
    return df

# Create a dataframe for the male female dataset
def create_pandas_dataframe2(data):  
    labels, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(13))
    j = 0
    idx = 0
    for i in range(int(len(data)/12)):
        I.append(data[j].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(data[j+1].astype('float32'))
        Three.append(data[j+2].astype('float32'))
        aVR.append(data[j+3].astype('float32'))
        aVL.append(data[j+4].astype('float32'))
        aVF.append(data[j+5].astype('float32'))
        V1.append(data[j+6].astype('float32'))
        V2.append(data[j+7].astype('float32'))
        V3.append(data[j+8].astype('float32'))
        V4.append(data[j+9].astype('float32'))
        V5.append(data[j+10].astype('float32'))
        V6.append(data[j+11].astype('float32'))
        j += 12

        if i < len(data)/2:
            labels.append(0) #negative = male = 0                                                #  III aVR aVL aVF I II V1 V2 V3 V4 V5 V6  is the correct sequence     
        else:
            labels.append(1) #positive = female = 1
        
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels}
    df = pd.DataFrame(data)
    return df

# Function to write a df to a CSV file
def write_df_to_csv():
    dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\rhythm'
    os.chdir(dir)
    df = pd.read_pickle('df_3beats')
    for i in range(df.shape[0]):
        arr = []
        for j in range(12):     
            arr.append(df.iloc[i, j])
        if df.iloc[i, 12]==0:
            os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\rhythm\3beats\negative')
        else:
            os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\rhythm\3beats\positive')
        with open((str(i)+".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(arr)

# Function needed to prep ECGs in order to plot them on the red grid with the plot_ecg.py function
def prep_ecg_plot():
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\CSV_files\\median'
    os.chdir(dir)  
    ecgs = []
    for file in os.listdir(dir):
        # print(file[5:8])
        # if file[5:8] in include: #use this loop for the condition in the above comment
        with open(file) as csv_file:
            print(csv_file.name)
            csv_reader = csv.reader(csv_file, delimiter=',')
            arr = []
            for row in csv_reader:
                arr.append(row)
            ecgs.append(arr)
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\excluded_examples')
    np.save('ecgs', ecgs)

# Define the low pass filter
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

# Functions for applying a low pass filter to a rhythm signal
def filter_signal():
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)
    df = pd.read_pickle('rhythm_signals_train')
    filler_df = pd.DataFrame(index=range(df.shape[0]),columns=range(13))
    for i in range(df.shape[0]):
        for j in range(12):
            fs = 250
            cutoff_high = 1
            cutoff_low = 15
            b,a = butter_lowpass(cutoff_low, fs, order=1)
            filter_signal = lfilter(b, a, df.iloc[i,j])
            filler_df.iloc[i,j] = filter_signal
            plt.plot(df.iloc[i,j], color = 'b')
            plt.plot(filter_signal, color = 'r')
            # plt.show()
        filler_df.iloc[i, 12] = df.iloc[i, 12]
    filler_df.to_pickle('rhythm_signals_train_filtered')           

# Find the peaks and valleys in the 3beat signals
def peak_finding():
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)  
    df = pd.read_pickle('3beat_train')
    all_xpos, all_ypos, all_type = [], [], []
    for i in range(df.shape[0]):
        for j in range(12):
            signal = []
            signal = df.iloc[i, j]
            signal = np.array(signal)     
            fs = 250
            cutoff_high = 0.1
            cutoff_low = 15
            b,a = butter_lowpass(cutoff_low, fs, order=1)
            filter_signal = lfilter(b,a,signal)
            signal = signal * 100
            filter_signal = filter_signal * 100
            signal = filter_signal
            # For plotting the filtered signal
            # plt.plot(signal)
            # plt.plot(filter_signal)
            # plt.show()
            inv_signal = signal*-1
            # Find the peaks and valleys
            peak_indices = find_peaks(signal, distance=1, prominence=3)
            valley_indices = find_peaks(inv_signal, distance=1, prominence=5)
            peaks, valleys, peak_coords, valley_coords, xpos, ypos, type = ([] for t in range(7))
            # Append the positions of the peaks or valleys
            for o in peak_indices[0]:
                peaks.append(signal[o])
                xpos.append(o)
                ypos.append(signal[o])
                type.append('peak')
            for p in valley_indices[0]:
                valleys.append(signal[p])
                xpos.append(p)
                ypos.append(signal[p])
                type.append('valley')
            # For plotting the signal with the peaks and valleys highlighted
            # fig = plt.figure()
            # plt.plot(signal)
            # plt.plot(filter_signal)
            # plt.scatter(peak_indices[0], peaks, marker='x', linewidths=0.6, c = 'r')
            # plt.scatter(valley_indices[0], valleys, marker='o', linewidths=0.6, c = 'g')
            # plt.show()
            # plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\rhythm\check_plots\{0}.png'.format(i))
            all_xpos.append(xpos)
            all_ypos.append(ypos)
            all_type.append(type)
    # Save the dataframe
    frame = {'x': all_xpos, 'y': all_ypos, 'type': all_type}
    df2 = pd.DataFrame(frame)
    df2.to_pickle('peaks_valleys_train')

# Create the persistence barcodes for the signals
def persistence_barcode():
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)
    df = pd.read_pickle('peaks_valleys_train')
    barcodes = []
    for i in range(df.shape[0]):
        xpos = np.array(df.iloc[i, 0])
        ypos = np.array(df.iloc[i, 1])
        type = np.array(df.iloc[i, 2])
        valley_stack, valley_xpos_stack, barcode, coded_peaks, coded_valleys = ([] for t in range(5))
        min_val = np.inf
        max_val = -np.inf
        # Loop over all peaks and valleys and create a barcode the length between the stack popped peak and valley
        for j in range(len(xpos)):
            idx = np.argmin(ypos)
            if type[idx] == 'valley':
                valley_stack.append(ypos[idx])
                valley_xpos_stack.append(xpos[idx])
                if ypos[idx] < min_val:
                    min_val = ypos[idx]
                ypos[idx] = np.inf
            else:
                if not valley_stack: # If there is no valley in the stack
                    ypos[idx] = np.inf
                    # continue
                else:
                    valley_val = valley_stack.pop()
                    valley_val_x = valley_xpos_stack.pop()
                    peak_val = ypos[idx]
                    ypos[idx] = np.inf
                    barcode.append([valley_val, peak_val])
                    coded_peaks.append([xpos[idx], peak_val])
                    coded_valleys.append([valley_val_x, valley_val])
                    if peak_val > max_val:
                        max_val = peak_val
        # plot_barcoded_valleys_and_peaks(coded_peaks, coded_valleys, i)
        # Normalize the barcodes between 0 and 1
        barcode_normalized = normalize_barcodes(barcode, min_val, max_val)
        barcodes.append(barcode_normalized)
        # For plotting the barcodes
        # bar_b = []
        # bar_t = []
        # for k in range(len(barcodes[i])):
        #     bar_b.append(barcodes[i][k][0])
        #     bar_t.append(barcodes[i][k][1])
        #     bar_t[k] = bar_t[k] - bar_b[k]
        # bar_x = list(range(0,len(barcodes[i])))
        # fig = plt.figure()
        # plt.bar(bar_x, bar_t, bottom=bar_b) 
        # plt.show()
        # plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\check_plots\{0}.png'.format(i))
    # Create the Betti curves from the barcodes
    betti = betti_curves(barcodes)

# Create the betti curves
def betti_curves(barcodes):
    # For getting a generalizable directory so it can run on other systems
    og_dir = os.getcwd()
    dir = og_dir.replace('\Augmentation', '')
    dir = dir+'\\data\\raw_dataframes'
    os.chdir(dir)
    all_bettis = []
    for row in barcodes:
        betti = [0]*300
        for point in row:
            for i in range(int(round(point[0])), int(round(point[1]))):
                betti[i] += 1
        plt.plot(betti)
        plt.show()
        all_bettis.append(betti)
    np.save('bettis_training', all_bettis)  

# Normalize the barcodes between values of 1 and 0
def normalize_barcodes(barcode, min, max):
    for i in range(len(barcode)):
        barcode[i][0] = (barcode[i][0]-min)*(300/(max-min))
        barcode[i][1] = (barcode[i][1]-min)*(300/(max-min))
    return(barcode)

# Plot the barcoded peaks and valleys
def plot_barcoded_valleys_and_peaks(peaks, valleys, i):
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\rhythm\DF')
    # df = pd.read_csv('pos_df.csv', header = None)
    df = pd.read_pickle('3beats_neg')
    signal = []
    for j in range(len(df.iloc[i])):
        signal.append(df.iloc[i,j])
    signal = np.array(signal)       
    # signal = savgol_filter(signal, 3, 1)

    x_peaks = []
    y_peaks = []
    # print(len(peaks), len(valleys))
    for k in range(len(peaks)):
        x_peaks.append(peaks[k][0])
        y_peaks.append(peaks[k][1])
    x_valleys = []
    y_valleys = []
    for k in range(len(valleys)):
        x_valleys.append(valleys[k][0])
        y_valleys.append(valleys[k][1])
    fig = plt.figure()
    plt.plot(signal)
    plt.scatter(x_peaks, y_peaks, marker='x', linewidths=0.6, c = 'r')
    plt.scatter(x_valleys, y_valleys, marker='o', linewidths=0.6, c = 'g')
    # plt.show()
    plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\rhythm\check_plots\{0}.png'.format(i))

# Normalize the betti curve between 0 and 1
def normalize_betti():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\rhythm')
    data = np.load('bettis_testing.npy',allow_pickle=True)  
    data = data.astype(float)
    for i,row in enumerate(data):
        max_Val = max(row)
        min_Val = min(row)
        for j in range(300):
            try:
                data[i][j] = data[i][j]/(max_Val-min_Val)
            except:
                print('ERRORTJE: ', data.iloc[i,j])
    return data

# Plot both the median signals for all 12 leads in one plot
def betti_curve_plot_12_leads():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\rhythm\DF')
    df = pd.read_pickle('pickled_df_bettis')
    df = df.sample(frac=1)
    # os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\betti_plots')
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
        # plt.show()
        if df.iloc[i, 12] == 0:
            name = 'neg'
        else:
            name = 'pos'
        plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\rhythm\betti_plots\{0}_{1}.png'.format(i, name))

# Plot both the median signal and the Betti curve in one plot, all 12 leads
def plot_median_and_bettis():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\median\DF\training')
    # df = pd.read_pickle('train_median_and_bettis_pickled_df')
    df = pd.read_pickle('testing_noisydata')
    # df = df.sample(frac=1)
    for i in range(df.shape[0]):
        fig, axs = plt.subplots(nrows= 6, ncols= 4)
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
        axs[0,2].plot(df.iloc[i, 12])
        axs[1,2].plot(df.iloc[i, 13])
        axs[2,2].plot(df.iloc[i, 14])
        axs[3,2].plot(df.iloc[i, 15])
        axs[4,2].plot(df.iloc[i, 16])
        axs[5,2].plot(df.iloc[i, 17])
        axs[0,3].plot(df.iloc[i, 18])
        axs[1,3].plot(df.iloc[i, 19])
        axs[2,3].plot(df.iloc[i, 20])
        axs[3,3].plot(df.iloc[i, 21])
        axs[4,3].plot(df.iloc[i, 22])
        axs[5,3].plot(df.iloc[i, 23])
        # plt.show()
        if df.iloc[i, 24] == 0:
            name = 'neg'
        else:
            name = 'pos'
        plt.savefig(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\median\DF\training\plots_noisydata\{0}_{1}.png'.format(i, name))

# Auxillary function that can set the area under a Betti curve to 1
def set_AUC_betti():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\DF')
    df = pd.read_pickle('pickled_df_bettis')
    arr = []
    for i in range(df.shape[0]):
        for j in range(12):
            betti =  df.iloc[i,j]
            sumBetti = sum(betti)
            for k in range(len(betti)):
                df.iloc[i,j][k] = betti[k]/sumBetti
    df.to_pickle('pickled_df_AUC1_bettis')

# Function to split the original dataframe into a train and validation set
def split_df():
    # For getting a generalizable directory so it can run on other systems
    dir1 = os.getcwd()
    og_dir = dir1.replace('\Augmentation', '')
    dir = og_dir+'\\data\\raw_dataframes'
    os.chdir(dir)  
    df = pd.read_pickle('df_pwave_removed')     

    # for splitting the df in 23 neg and pos patients for the testing and the rest for training
    # rand_neg = random.sample(range(0, 144), 24)
    # rand_pos = random.sample(range(145, 288), 24)
    # df_neg = df.iloc[rand_neg,:]
    # df_pos = df.iloc[rand_pos,:]

    # df_neg = df.iloc[24:48,:]
    # df_pos = df.iloc[168:192,:]
    # arr1 = list(range(24, 48))
    # arr2 = list(range(168, 192))

    df_neg = df.iloc[0:24,:]
    df_pos = df.iloc[144:168,:]
    arr1 = list(range(0, 24))
    arr2 = list(range(144, 168))
    arr = arr1 + arr2
    
    # arr = rand_neg + rand_pos
    df = df.drop(arr, axis=0)
    frames = [df_neg, df_pos]
    result = pd.concat(frames)
    df = df.reset_index(drop=True)
    os.chdir(og_dir+'\\data\\training')
    df.to_pickle('train_df_imp')
    os.chdir(og_dir+'\\data\\validation')
    result.to_pickle('test_df_imp')

# Function that can convert a 1 bit label df to a 2 bit label df
def convert_data_to_2bit():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\training')    # 288 in tot
    df = pd.read_pickle('train_img_df')  
    arr = []
    for i in range(len(df)):          #  0 = [1, 0] negative         1 = [0, 1] positive
        if df.iloc[i,12] == 0:     
            arr.append([1,0])
        else:
            arr.append([0,1])
    df['2bit_label'] = arr
    df.to_pickle('train_img_df_2bit')
    print(df)

# The wave stretch method for only adding the sin cos wave
def wave_Stretch_part1():
     # For getting a generalizable directory so it can run on other systems
    dir1 = os.getcwd()
    og_dir = dir1.replace('\Augmentation', '')
    dir = og_dir+'\\data\\training'
    os.chdir(dir)
    df = pd.read_pickle('median_1D_train')

    # Declare some empty variables
    labels, patient, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(14))
    for i in range(len(df)):
        print(i)
        for p in range(3):
            new_signals = []
            for j in range(12):
                if p != 0:
                    # Randomize the amplitude of the sin or cos wave
                    x = range(0, random.randint(3,16))
                    # Randomize whether it becomes a sin or cos wave
                    z = random.randint(0,1)
                    if z == 0:
                        x = np.sin(x)
                    else:
                        x = np.cos(x)
                    signal_size = 3
                    while(signal_size % 2 != 0):
                        # Randomly within a range determine what specific area of the center of the ECG to add wave form to
                        signal_size = random.randint(100, 200)
                    # Add the wave form to the original signal and resample it and pad it
                    noise = signal.resample(x, signal_size)
                    noise = (noise/18)
                    noise = np.pad(noise, int((300-signal_size)/2), constant_values=(noise[0], noise[signal_size-1]))
                    augmented_signal = df.iloc[i,j] + noise
                    new_signals.append(augmented_signal)
                else:
                    new_signals.append(df.iloc[i,j])
            I.append(new_signals[0].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            II.append(new_signals[1].astype('float32'))
            Three.append(new_signals[2].astype('float32'))
            aVR.append(new_signals[3].astype('float32'))
            aVL.append(new_signals[4].astype('float32'))
            aVF.append(new_signals[5].astype('float32'))
            V1.append(new_signals[6].astype('float32'))
            V2.append(new_signals[7].astype('float32'))
            V3.append(new_signals[8].astype('float32'))
            V4.append(new_signals[9].astype('float32'))
            V5.append(new_signals[10].astype('float32'))
            V6.append(new_signals[11].astype('float32'))
            labels.append(df.iloc[i,12])
            patient.append(df.iloc[i,13])
    # Save the augmented data
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels, 'patient': patient}
    df2 = pd.DataFrame(data)
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\training')
    df2.to_pickle('df_aug')

# The wave stretch method that stretches and squeezes the signal
def wave_Stretch_part2():
     # For getting a generalizable directory so it can run on other systems
    dir1 = os.getcwd()
    og_dir = dir1.replace('\Augmentation', '')
    dir = og_dir+'\\data\\training'
    os.chdir(dir)
    df = pd.read_pickle('median_1D_train')

    # Declare some empty variables
    labels, patient, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(14))
    for i in range(len(df)):
        print(i)
        new_signals1 = []
        new_signals2 = []
        og_signals = []
        for j in range(12):
            signal_og = df.iloc[i,j]

            # for stretching the signal
            cutoff = 3
            while(cutoff % 2 != 0):
                cutoff = random.randint(25,55)
            signal1 = signal_og[cutoff:(300-cutoff)]
            signal1 = signal.resample(signal1, 300)

            signal1 = np.pad(signal1, (int(cutoff/4), 0), constant_values=(signal1[0]))
            signal1 = signal1[0:300]

            # for compressing the signal
            shrink_size = random.randint(20, 50)
            signal2 = np.pad(signal_og, shrink_size, constant_values=(signal_og[0], signal_og[299]))
            signal2 = signal2[10:len(signal2)] # small adaptation to keep the pqrs complex on the original spot
            signal2 = signal.resample(signal2, 300)

            new_signals1.append(signal1)
            new_signals2.append(signal2)
            og_signals.append(signal_og)

            # fig, axs = plt.subplots(3)
            # axs[0].plot(signal_og, c = 'b')
            # axs[1].plot(signal1)
            # axs[2].plot(signal2)
            # plt.show()

        I.append(og_signals[0].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(og_signals[1].astype('float32'))
        Three.append(og_signals[2].astype('float32'))
        aVR.append(og_signals[3].astype('float32'))
        aVL.append(og_signals[4].astype('float32'))
        aVF.append(og_signals[5].astype('float32'))
        V1.append(og_signals[6].astype('float32'))
        V2.append(og_signals[7].astype('float32'))
        V3.append(og_signals[8].astype('float32'))
        V4.append(og_signals[9].astype('float32'))
        V5.append(og_signals[10].astype('float32'))
        V6.append(og_signals[11].astype('float32'))
        labels.append(df.iloc[i,12])
        patient.append(df.iloc[i,13])
        
        I.append(new_signals1[0].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(new_signals1[1].astype('float32'))
        Three.append(new_signals1[2].astype('float32'))
        aVR.append(new_signals1[3].astype('float32'))
        aVL.append(new_signals1[4].astype('float32'))
        aVF.append(new_signals1[5].astype('float32'))
        V1.append(new_signals1[6].astype('float32'))
        V2.append(new_signals1[7].astype('float32'))
        V3.append(new_signals1[8].astype('float32'))
        V4.append(new_signals1[9].astype('float32'))
        V5.append(new_signals1[10].astype('float32'))
        V6.append(new_signals1[11].astype('float32'))
        labels.append(df.iloc[i,12])
        patient.append(df.iloc[i,13])

        I.append(new_signals2[0].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(new_signals2[1].astype('float32'))
        Three.append(new_signals2[2].astype('float32'))
        aVR.append(new_signals2[3].astype('float32'))
        aVL.append(new_signals2[4].astype('float32'))
        aVF.append(new_signals2[5].astype('float32'))
        V1.append(new_signals2[6].astype('float32'))
        V2.append(new_signals2[7].astype('float32'))
        V3.append(new_signals2[8].astype('float32'))
        V4.append(new_signals2[9].astype('float32'))
        V5.append(new_signals2[10].astype('float32'))
        V6.append(new_signals2[11].astype('float32'))
        labels.append(df.iloc[i,12])
        patient.append(df.iloc[i,13])

    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels, 'patient': patient}
    df2 = pd.DataFrame(data)
    print(df2)
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\training')
    df2.to_pickle('df_full_augmentation')

# Remove extra p and t waves and replace by padding
def extra_Pwave_removal(): # also centers the median beat
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median') 
    df = pd.read_pickle('median_df') 

    # Declare some empty variables
    labels, patient, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(14))
    for i in range(len(df)):
        print(i)
        new_signals = []
        for j in range(12):
            og_signal = df.iloc[i,j]
            # This lead uasually has valleys instead of peaks
            if j == 3:
                og_signal = og_signal*-1
            fs = 250
            cutoff_high = 0.1
            cutoff_low = 15
            # Get a low pass filtered version of the signal to better find peaks
            b,a = butter_lowpass(cutoff_low, fs, order=1)
            filter_signal = lfilter(b,a,og_signal)
            # Find peaks in the filtered signal
            peak_indices = find_peaks(filter_signal, distance=20, prominence=0.05)
            peaks = []
            # Save the found peaks
            for o in peak_indices[0]:
                peaks.append(og_signal[o])
            peak_xvalues = peak_indices[0]
            # If there are more than 3 peaks we might have to remove
            if len(peak_xvalues) > 3:
                # If there is an extra p and t wave
                if len(peak_xvalues) == 5:
                    cut1 = int(((peak_xvalues[0]+peak_xvalues[1])/2))
                    cut2 = int(((peak_xvalues[4]+peak_xvalues[3])/2))
                    cut_signal = og_signal[cut1:cut2]
                    new_signal = np.pad(cut_signal, (cut1, 0), constant_values=og_signal[cut1])
                    new_signal = np.pad(new_signal, (0, 300-cut2), constant_values=og_signal[cut2])
                # If there is only one extra wave
                elif len(peak_xvalues) == 4:
                    if peak_xvalues[3] > 240:
                        cut1 = int(((peak_xvalues[2]+peak_xvalues[3])/2))
                        cut_signal = og_signal[0:cut1]
                        new_signal = np.pad(cut_signal, (0, 300-cut1), constant_values=og_signal[cut1])
                    else:
                        new_signal = og_signal
            else:
                new_signal = og_signal
                # For plotting the signal
                # fig, axs = plt.subplots(2)
                # axs[0].plot(og_signal, c = 'b')
                # # axs[0].scatter(peak_indices[0], peaks, marker='x', linewidths=0.6, c = 'r')
                # axs[1].plot(new_signal)
                # plt.show()
            # Reverse the signal again
            if j == 3:
                new_signal = new_signal*-1
            # For centering the ECG
            new_signal = new_signal[0:270]
            new_signal = np.pad(new_signal, (30, 0), constant_values=new_signal[0])
            new_signals.append(new_signal)
            # Plot the centered ECG
            # fig = plt.figure()
            # plt.plot(new_signal)
            # # plt.scatter(peak_indices[0], peaks, marker='x', linewidths=0.6, c = 'r')
            # plt.show()
        I.append(new_signals[0].astype('float32'))             # I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        II.append(new_signals[1].astype('float32'))
        Three.append(new_signals[2].astype('float32'))
        aVR.append(new_signals[3].astype('float32'))
        aVL.append(new_signals[4].astype('float32'))
        aVF.append(new_signals[5].astype('float32'))
        V1.append(new_signals[6].astype('float32'))
        V2.append(new_signals[7].astype('float32'))
        V3.append(new_signals[8].astype('float32'))
        V4.append(new_signals[9].astype('float32'))
        V5.append(new_signals[10].astype('float32'))
        V6.append(new_signals[11].astype('float32'))
        labels.append(df.iloc[i,12])
        patient.append(df.iloc[i,13])
    # Save the augmented data
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels, 'patient': patient}
    df2 = pd.DataFrame(data)
    print(df2)

    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median')
    df2.to_pickle('df_pwave_removed')       
    # os.chdir(r'D:\Master Thesis\Median_TransferLearning\median')   
    # df2.to_pickle('median_10k_df_imp') 

# Plot the average negative and positive signal, used for grad cam plotting
def plot_avg_pos_neg_signals():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\training')    # 288 in tot
    df = pd.read_pickle('df_full_augmentation') 

    pos_avg = []
    neg_avg = []
    
    for i in range(12):
        signal = np.zeros(300)
        signal2 = np.zeros(300)
        for j in range(int(len(df)/2)):
            signal = signal + df.iloc[j,i]
        for k in range(int(len(df)/2), len(df)):
            signal2 = signal2 + df.iloc[k,i]
        neg_avg.append(signal/int(len(df)/2))
        pos_avg.append(signal2/int(len(df)/2))

    for i in range(12):
        fig, axs = plt.subplots(2)
        axs[0].plot(neg_avg[i])
        axs[1].plot(pos_avg[i])
        plt.show()
        

if __name__ == "__main__":
    plot_avg_pos_neg_signals()
 
    # os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset\median') 
    # df = pd.read_pickle('median_df')
    # print(df) 

    # create_dataframes_male_female(r'D:\Master Thesis\Median_TransferLearning\median')
    # subset_transferlearning_data()
    # check_transferlearning_data_for_nan()
    # remove_nan_transferlearning_data()

    # go from the individual patient csv files to a median df
    create_dataframes()                                                                                                                                                                                                                                                                        

    # for the creating the betti curves
    # df_neg = df.iloc[0:144, :]
    # df_pos = df.iloc[144:288, :]
    # peak_finding()
    # persistence_barcode()
    # persistence_barcode('peaks_valleys_pos')

    # # prepare dataframe for CNN_betti
    # os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\rhythm')
    # neg_data = np.load('bettis_neg.npy',allow_pickle=True)  
    # pos_data = np.load('bettis_pos.npy',allow_pickle=True)  
    # neg_data = normalize_betti(neg_data)
    # data = normalize_betti()
    # df = create_pandas_dataframe2(data)
    # os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\rhythm')
    # df.to_pickle('df_bettis_testing')
    # print(df)

    # betti_curve_plot_12_leads()
    # set_AUC_betti()
    # prep_ecg_plot()
    # plot_median_and_bettis()
    # split_df()
    # time_series_augmentation()
    # time_series_augmentation_part2()
    # extra_Pwave_removal()
    # filter_signal()
    # test()
    # write_df_to_csv()
    # convert_data_to_2bit()

    # transfer_data()






    





