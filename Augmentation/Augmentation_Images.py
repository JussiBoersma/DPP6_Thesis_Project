import csv
import os
import numpy as np
import sys
from numpy.core.numeric import NaN
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
from PIL import Image
import PIL.ImageOps    
import cv2

all_dir = os.getcwd()+'\\'+'data\\'

# Create images from a 1D signal dataframe
def create_images(run_type):
    os.chdir(all_dir+'\\'+run_type)
    if run_type == 'training':
        adname = 'train'
    else:
        adname = 'val'
    df = pd.read_pickle('median_1D_'+adname)
    cnt = 0
    for i in range(df.shape[0]):
        if cnt == 9:
            cnt = 0
        if df.iloc[i,12] == 0:
            name = 'negative'
            os.mkdir((all_dir+r'\images\training\negative\{0}'.format(df.iloc[i,13][:-4]+"_"+str(cnt))))                #+"_"+str(cnt)
        else:
            name = 'positive'
            os.mkdir((all_dir+r'\images\training\positive\{0}'.format(df.iloc[i,13][:-4]+"_"+str(cnt))))                 #+"_"+str(cnt)
        for j in range(12):
            plt.plot(df.iloc[i,j], color = 'w')
            ax = plt.gca()
            ax.set_facecolor('k')
            plt.ylim([-1, 1])
            plt.tight_layout()
            if df.iloc[i,12] == 0:
                name = 'negative'
            else:
                name = 'positive'
            plt.savefig(all_dir+'\\images\\training\\{0}\\{1}\\{2}.png'.format(name, df.iloc[i,13][:-4]+"_"+str(cnt), j), bbox_inches='tight', cmap='gray')
            plt.clf()
        cnt = cnt+1

# Crop the images so that the plot axis are removed
def crop_images(run_type):
    dir = all_dir+'\\images\\'+run_type
    os.chdir(dir)
    for file in os.listdir(dir):
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                img = cv2.imread(name)
                crop_img = img[17:434, 86:596]
                cv2.imwrite(name, crop_img)

# Dilate the images to make signal clearer
def dilate(run_type):
    dir = all_dir+'\\images\\'+run_type
    os.chdir(dir)
    x = 0
    for file in os.listdir(dir):
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                image = cv2.imread(name)
                kernel = np.ones((2, 2), 'uint8')
                dilate_img = cv2.dilate(image, kernel, iterations=1)
                cv2.imwrite(name, dilate_img)

# Convert the image to a gray scale image
def gray_scale_images(run_type):
    dir = all_dir+'\\images\\'+run_type
    os.chdir(dir)
    for file in os.listdir(dir):
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                image = cv2.imread(name)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dir+'\\'+r'{0}\{1}\{2}.png'.format(file, file1, i), gray_image)

# Go from seperate images to one pandas dataframe that contains all the arrays of images       
def create_img_df(run_type):    
    labels, Three, aVR, aVL, aVF, I, II, V1, V2, V3, V4, V5, V6 = ([] for i in range(13))
    j = 0
    dir = all_dir+'\\images\\'+run_type
    for diagnosis in os.listdir(dir):
        for patient in os.listdir(dir+"\\"+diagnosis):
            os.chdir(dir+"\\"+diagnosis+"\\"+patient)
            print(patient)

            im0 = Image.open('0.png')
            im1 = Image.open('1.png')
            im2 = Image.open('2.png')
            im3 = Image.open('3.png')
            im4 = Image.open('4.png')
            im5 = Image.open('5.png')
            im6 = Image.open('6.png')
            im7 = Image.open('7.png')
            im8 = Image.open('8.png')
            im9 = Image.open('9.png')
            im10 = Image.open('10.png')
            im11 = Image.open('11.png')

            resized_image0 = im0.resize((128,128))
            resized_image1 = im1.resize((128,128))
            resized_image2 = im2.resize((128,128))
            resized_image3 = im3.resize((128,128))
            resized_image4 = im4.resize((128,128))
            resized_image5 = im5.resize((128,128))
            resized_image6 = im6.resize((128,128))
            resized_image7 = im7.resize((128,128))
            resized_image8 = im8.resize((128,128))
            resized_image9 = im9.resize((128,128))
            resized_image10 = im10.resize((128,128))
            resized_image11 = im11.resize((128,128))

            imgs0 = np.asarray(resized_image0)
            imgs1 = np.asarray(resized_image1)
            imgs2 = np.asarray(resized_image2)
            imgs3 = np.asarray(resized_image3)
            imgs4 = np.asarray(resized_image4)
            imgs5 = np.asarray(resized_image5)
            imgs6 = np.asarray(resized_image6)
            imgs7 = np.asarray(resized_image7)
            imgs8 = np.asarray(resized_image8)
            imgs9 = np.asarray(resized_image9)
            imgs10 = np.asarray(resized_image10)
            imgs11 = np.asarray(resized_image11)

            imgs0 = imgs0.reshape((1, 128, 128))
            imgs1 = imgs1.reshape((1, 128, 128))
            imgs2 = imgs2.reshape((1, 128, 128))
            imgs3 = imgs3.reshape((1, 128, 128))
            imgs4 = imgs4.reshape((1, 128, 128))
            imgs5 = imgs5.reshape((1, 128, 128))
            imgs6 = imgs6.reshape((1, 128, 128))
            imgs7 = imgs7.reshape((1, 128, 128))
            imgs8 = imgs8.reshape((1, 128, 128))
            imgs9 = imgs9.reshape((1, 128, 128))
            imgs10 = imgs10.reshape((1, 128, 128))
            imgs11 = imgs11.reshape((1, 128, 128))

            imgs0 = imgs0/255
            imgs1 = imgs1/255
            imgs2 = imgs2/255
            imgs3 = imgs3/255
            imgs4 = imgs4/255
            imgs5 = imgs5/255
            imgs6 = imgs6/255
            imgs7 = imgs7/255
            imgs8 = imgs8/255
            imgs9 = imgs9/255
            imgs10 = imgs10/255
            imgs11 = imgs11/255

            I.append(np.asarray(imgs0))
            II.append(np.asarray(imgs1))
            Three.append(np.asarray(imgs2))
            aVR.append(np.asarray(imgs3))
            aVL.append(np.asarray(imgs4))
            aVF.append(np.asarray(imgs5))
            V1.append(np.asarray(imgs6))
            V2.append(np.asarray(imgs7))
            V3.append(np.asarray(imgs8))
            V4.append(np.asarray(imgs9))
            V5.append(np.asarray(imgs10))
            V6.append(np.asarray(imgs11))

            if diagnosis == 'negative':
                labels.append(0)
            else:
                labels.append(1)
    data = {'I': I, 'II': II, 'Three': Three, 'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'label': labels}
    if run_type == 'training':
        adname = 'train'
    else:
        adname = 'val'
    df = pd.DataFrame(data)
    os.chdir(all_dir + '\\'+run_type)
    df.to_pickle('median_2D_'+adname)

# Convert images to ppm format for the image morph
def convert_to_ppm(dir0, dir1):
    if dir0 == 0:
        dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\image_morph\train_images_zip'
    else:
        dir = dir0
    x=0
    os.chdir(dir)
    for file in os.listdir(dir):
        # if dir0 != 0:
        #     if file == 'positive':
        #         x = 120
        #     for i in range(120):
        #         os.mkdir(dir1+'\\'+file+'\\'+str(i+x))
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                im = Image.open(name)
                new_name = str(i)+'.ppm'
                if dir0 == 0:
                    im.save(new_name)
                else:
                    im.save(dir1+'\\'+r'{0}_{1}.ppm'.format(file1, i))

# Apply a gaussian filter to images
def gaussian_filter(dir0, dir1):
    if dir0 == 0:
        # dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Clean_data_bonus\image_morph\train_images'
        dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v2\newsplit\images\training'
    else:
        dir = dir0
    os.chdir(dir)
    x = 0
    for file in os.listdir(dir):
        if dir0 != 0:
            if file == 'positive':
                x = 120
            for i in range(120):
                os.mkdir(dir1+'\\'+file+'\\'+str(i+x))
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                image = cv2.imread(name)
                dilate_img = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
                dilate_img = cv2.cvtColor(dilate_img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('img', dilate_img)
                # cv2.imshow('img2', image)
                # cv2.waitKey(0)
                # sys.exit()
                if dir0 == 0:
                    cv2.imwrite(name, dilate_img)
                else:
                    cv2.imwrite(dir1+'\\'+r'{0}\{1}\{2}.png'.format(file, file1, i), dilate_img)

# Invert images from black to white
def invert_images():
    dir = r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw_denoised\median\DF\images\train_images'
    os.chdir(dir)
    for file in os.listdir(dir):
        for file1 in os.listdir(dir+ "\\" + file):
            os.chdir(dir+"\\"+file+"\\"+file1)
            for i in range(12):
                name = str(i)+'.png'
                image = Image.open(name)
                gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
                inverted_image = PIL.ImageOps.invert(image.convert('RGB'))
                inverted_image.save(name)

def image_changer(dir0, dir1):
    # create_images()
    # crop_images()
    # dilate(0,0)
    # gray_scale_images(0, 0)
    create_img_df(0)

# Additional functions for preparing the morph images
def ppm_image_structuring():
    dir = r'D:\Master Thesis\Data\Morphing\morphed_images\morph_dir'
    outdir = r'D:\Master Thesis\Data\Morphing\images'
    os.chdir(dir)
    cnt=0
    for i in range(240): #go over each original patient basis ecg
        if i < 120:
            condition = 'negative'
        else:
            condition = 'positive'
        for j in range(4): #go over each augmented new patient
            os.mkdir(r'D:\Master Thesis\Data\Morphing\images\{0}\{1}'.format(condition, i+240+cnt)) 
            patient_nr = (i+240+cnt)
            cnt+=1
            for k in range(12): #go over each lead
                name = str(i)+'_'+str(k)+'-'+str(j)+'-'+'morphed.ppm'
                im = Image.open(name)
                im.save(outdir+'\\'+r'{0}\{1}\{2}.png'.format(condition, patient_nr, k))
        cnt-=1

# For each individual patient compile their images into one image
def combine_12_images_plot_per_patient():
    # create figure
    fig = plt.figure(figsize=(12, 10))
    
    # setting values to rows and column variables
    rows = 3
    columns = 4
    
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\activation_maps')
    for classification in os.listdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\activation_maps'):
        for patient in os.listdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\activation_maps'+"\\"+classification):
            os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\activation_maps\{0}\{1}'.format(classification, patient))
            # reading images
            Image1 = cv2.imread('I.jpg')
            Image5 = cv2.imread('II.jpg')
            Image9 = cv2.imread('III.jpg')
            Image2 = cv2.imread('aVR.jpg')
            Image6 = cv2.imread('aVL.jpg')
            Image10 = cv2.imread('aVF.jpg')
            Image3 = cv2.imread('V1.jpg')
            Image7 = cv2.imread('V2.jpg')
            Image11 = cv2.imread('V3.jpg')
            Image4 = cv2.imread('V4.jpg')
            Image8 = cv2.imread('V5.jpg')
            Image12 = cv2.imread('V6.jpg')

            arr = [Image1, Image2, Image3, Image4, Image5, Image6, Image7, Image8, Image9, Image10, Image11, Image12]
            title = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']

            i = 0
            for image in arr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Adds a subplot at the 1st position
                fig.add_subplot(rows, columns, i+1)
                
                # showing image
                plt.imshow(image)
                plt.axis('off')
                plt.title(title[i])
                i = i+1
            # plt.show()
            os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\activation_maps_2\{0}'.format(classification))
            plt.savefig(patient+'.png', bbox_inches='tight')

# Combine 12 lead images to one nice plot for in the thesis
def combine_12_images_plot():
    # create figure
    fig = plt.figure(figsize=(12, 10))
    # setting values to rows and column variables
    rows = 4
    columns = 3
    # For getting a generalizable directory so it can run on other systems
    dir1 = os.getcwd()
    og_dir = dir1.replace('\Augmentation', '')
    dir = og_dir+'\\Grad_CAM_results\\all_patients'
    os.chdir(dir)
    # reading images
    Image4 = cv2.imread('I.jpg')
    Image5 = cv2.imread('II.jpg')
    Image6 = cv2.imread('III.jpg')
    Image3 = cv2.imread('aVR.jpg')
    Image2 = cv2.imread('aVL.jpg')
    Image1 = cv2.imread('aVF.jpg')
    Image7 = cv2.imread('V1.jpg')
    Image8 = cv2.imread('V2.jpg')
    Image9 = cv2.imread('V3.jpg')
    Image10 = cv2.imread('V4.jpg')
    Image11 = cv2.imread('V5.jpg')
    Image12 = cv2.imread('V6.jpg')

    arr = [Image1, Image2, Image3, Image4, Image5, Image6, Image7, Image8, Image9, Image10, Image11, Image12]
    title = ['aVF', 'aVL', 'aVR', 'I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    i = 0
    for image in arr:
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i+1)
        
        # showing image
        plt.imshow(image)
        plt.axis('off')
        plt.title(title[i], y = -0.18)
        # plt.tight_layout(pad=0.5)
        i = i+1
    # plt.show()
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.8, top=1, bottom=0.1)
    os.chdir(og_dir+'\\Grad_CAM_results\\save_gradcams')
    plt.savefig('model4_val77_sens875_sens667.png', bbox_inches='tight')

# Create the average median signal images for the Grad-CAM sum plot
def creat_avg_images():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\testing')
    df = pd.read_pickle('test_df_imp')

    leads = []
    for i in range(12):
        sumi = np.zeros(300)
        for j in range(df.shape[0]):
            sumi = sumi + df.iloc[j,i]
        sumi = sumi/48
        # plt.plot(sumi)
        # plt.show()
        leads.append(sumi)

    data = {'I': [leads[0]], 'II': [leads[1]], 'Three': [leads[2]], 'aVR': [leads[3]], 'aVL': [leads[4]], 'aVF': [leads[5]], 'V1': [leads[6]], 'V2': [leads[7]], 'V3': [leads[8]], 'V4': [leads[9]], 'V5': [leads[10]], 'V6': [leads[11]], 'label': 0}
    df = pd.DataFrame(data)
    print(df)
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median')
    df.to_pickle('average_medians')

# Combine 12 images to one image for the 2D_combined
def combine_12_images_to_one():
    dir = r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\images\training_imp'
    for condition in os.listdir(dir):
        image_list = []
        for patient in os.listdir(dir+'//'+condition):
            x = 0
            y = 0
            new_image = Image.new('L',(3*128, 4*128))
            for i in range(12):
                os.chdir(dir+'//'+condition+'//'+patient)
                image1 = Image.open(str(i)+'.png')
                image1 = image1.resize((128,128))
                new_image.paste(image1,(x,y))
                if ((i+1) % 3 == 0):
                    x = 0
                    y = y + 128
                else:
                    x = x + 128
            # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            new_image.save(r'C:\Users\jussi\Documents\Master Thesis\Data\Final_dataset_v3\median\images\training_one_img\{0}\{1}.png'.format(condition, patient),"PNG")


if __name__ == "__main__":
    # Run this to create images and the image dataframe for the 2D CNN
    arr = ['training', 'validation']
    for run_type in arr:
        create_images(run_type)
        crop_images(run_type)
        dilate(run_type)
        gray_scale_images(run_type)
        create_img_df(run_type)

