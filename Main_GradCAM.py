import csv
import os
import numpy as np
from scipy.sparse import dia
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
# from torch._C import long
from torch.nn import functional as F
from torch.nn.modules.pooling import AvgPool2d
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from multiprocessing import cpu_count
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold
import gudhi
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import math
import cv2
from PIL import Image
import sys
import pathlib
import shutil
from torch.utils.tensorboard import SummaryWriter

m = nn.Sigmoid()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The 2D CNN gradcam model
class Two_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(c_in=1, c_out=3, dropout=0.4, kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=3, c_out=4, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.block5 = nn.Conv2d(48, 10, kernel_size=32)
        self.fcRED = nn.Linear(10, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.4)

        self.gradients0 = None
        self.activations0 = None
        self.gradients1 = None
        self.activations1 = None
        self.gradients2 = None
        self.activations2 = None
        self.gradients3 = None
        self.activations3 = None
        self.gradients4 = None
        self.activations4 = None
        self.gradients5 = None
        self.activations5 = None
        self.gradients6 = None
        self.activations6 = None
        self.gradients7 = None
        self.activations7 = None
        self.gradients8 = None
        self.activations8 = None
        self.gradients9 = None
        self.activations9 = None
        self.gradients10 = None
        self.activations10 = None
        self.gradients11 = None
        self.activations11 = None

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block

    def activations_hook0(self, grad):
        self.gradients0 = grad
    def set_activations0(self, x):
        self.activations0 = x

    def activations_hook1(self, grad):
        self.gradients1 = grad
    def set_activations1(self, x):
        self.activations1 = x

    def activations_hook2(self, grad):
        self.gradients2 = grad
    def set_activations2(self, x):
        self.activations2 = x

    def activations_hook3(self, grad):
        self.gradients3 = grad
    def set_activations3(self, x):
        self.activations3 = x

    def activations_hook4(self, grad):
        self.gradients4 = grad
    def set_activations4(self, x):
        self.activations4 = x

    def activations_hook5(self, grad):
        self.gradients5 = grad
    def set_activations5(self, x):
        self.activations5 = x

    def activations_hook6(self, grad):
        self.gradients6 = grad
    def set_activations6(self, x):
        self.activations6 = x

    def activations_hook7(self, grad):
        self.gradients7 = grad
    def set_activations7(self, x):
        self.activations7 = x

    def activations_hook8(self, grad):
        self.gradients8 = grad
    def set_activations8(self, x):
        self.activations8 = x

    def activations_hook9(self, grad):
        self.gradients9 = grad
    def set_activations9(self, x):
        self.activations9 = x

    def activations_hook10(self, grad):
        self.gradients10 = grad
    def set_activations10(self, x):
        self.activations10 = x

    def activations_hook11(self, grad):
        self.gradients11 = grad
    def set_activations11(self, x):
        self.activations11 = x
    
    def forward(self, leads):
        x0 = self.block1((leads[:,0]))
        x0 = self.maxpool(x0)
        x0 = self.block2(x0)
        self.set_activations0(x0)
        h = x0.register_hook(self.activations_hook0)
        x0 = self.maxpool(x0)

        x1 = self.block1((leads[:,1]))
        x1 = self.maxpool(x1)
        x1 = self.block2(x1) 
        self.set_activations1(x1)
        h = x1.register_hook(self.activations_hook1)
        x1 = self.maxpool(x1)  

        x2 = self.block1((leads[:,2]))
        x2 = self.maxpool(x2)
        x2 = self.block2(x2) 
        self.set_activations2(x2)
        h = x2.register_hook(self.activations_hook2)
        x2 = self.maxpool(x2) 

        x3 = self.block1((leads[:,3]))
        x3 = self.maxpool(x3)
        x3 = self.block2(x3) 
        self.set_activations3(x3)
        h = x3.register_hook(self.activations_hook3)
        x3 = self.maxpool(x3) 

        x4 = self.block1((leads[:,4]))
        x4 = self.maxpool(x4)
        x4 = self.block2(x4) 
        self.set_activations4(x4)
        h = x4.register_hook(self.activations_hook4)
        x4 = self.maxpool(x4) 

        x5 = self.block1((leads[:,5]))
        x5 = self.maxpool(x5)
        x5 = self.block2(x5) 
        self.set_activations5(x5)
        h = x5.register_hook(self.activations_hook5)
        x5 = self.maxpool(x5)

        x6 = self.block1((leads[:,6]))
        x6 = self.maxpool(x6)
        x6 = self.block2(x6) 
        self.set_activations6(x6)
        h = x6.register_hook(self.activations_hook6)
        x6 = self.maxpool(x6) 

        x7 = self.block1((leads[:,7]))
        x7 = self.maxpool(x7)
        x7 = self.block2(x7) 
        self.set_activations7(x7)
        h = x7.register_hook(self.activations_hook7)
        x7 = self.maxpool(x7)

        x8 = self.block1((leads[:,8]))
        x8 = self.maxpool(x8)
        x8 = self.block2(x8) 
        self.set_activations8(x8)
        h = x8.register_hook(self.activations_hook8)
        x8 = self.maxpool(x8) 

        x9 = self.block1((leads[:,9]))
        x9 = self.maxpool(x9)
        x9 = self.block2(x9) 
        self.set_activations9(x9)
        h = x9.register_hook(self.activations_hook9)
        x9 = self.maxpool(x9) 

        x10 = self.block1((leads[:,10]))
        x10 = self.maxpool(x10)
        x10 = self.block2(x10) 
        self.set_activations10(x10)
        h = x10.register_hook(self.activations_hook10)
        x10 = self.maxpool(x10)

        x11 = self.block1((leads[:,11]))
        x11 = self.maxpool(x11)
        x11 = self.block2(x11) 
        self.set_activations11(x11)
        h = x11.register_hook(self.activations_hook11)
        x11 = self.maxpool(x11)
        
        combined = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], dim=1)
        combined = self.block5(combined)
        combined = combined.view(combined.size(0), -1)
        combined = self.dropout(combined)
        x = self.fcRED(combined)
        output = x
        return output

    def get_activations_gradient0(self):
        return self.gradients0
    def get_activations0(self):
        return self.activations0

    def get_activations_gradient1(self):
        return self.gradients1
    def get_activations1(self):
        return self.activations1

    def get_activations_gradient2(self):
        return self.gradients2
    def get_activations2(self):
        return self.activations2

    def get_activations_gradient3(self):
        return self.gradients3
    def get_activations3(self):
        return self.activations3

    def get_activations_gradient4(self):
        return self.gradients4
    def get_activations4(self):
        return self.activations4

    def get_activations_gradient5(self):
        return self.gradients5
    def get_activations5(self):
        return self.activations5

    def get_activations_gradient6(self):
        return self.gradients6
    def get_activations6(self):
        return self.activations6

    def get_activations_gradient7(self):
        return self.gradients7
    def get_activations7(self):
        return self.activations7

    def get_activations_gradient8(self):
        return self.gradients8
    def get_activations8(self):
        return self.activations8

    def get_activations_gradient9(self):
        return self.gradients9
    def get_activations9(self):
        return self.activations9

    def get_activations_gradient10(self):
        return self.gradients10
    def get_activations10(self):
        return self.activations10

    def get_activations_gradient11(self):
        return self.gradients11
    def get_activations11(self):
        return self.activations11

# Define the model architecture for the combined image gradcam
class Two_D_single_img(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(c_in=3, c_out=12, dropout=0.2, kernel_size=3, stride=1, padding=1)     # 0.4 dropout
        self.block2 = self.conv_block(c_in=12, c_out=8, dropout=0.2, kernel_size=3, stride=1, padding=1)     # 0.5 dropout
        self.block3 = self.conv_block(c_in=8, c_out=4, dropout=0.2, kernel_size=3, stride=1, padding=1)     
        self.fcRED = nn.Linear(3072, 1)
        self.dropout = nn.Dropout2d(p=0.1)   # 0.4 dropout
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gradients = None
        self.activations = None

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block

    # Grad-CAM gradient hook functions
    def activations_hook(self, grad):
        self.gradients = grad
    def set_activations(self, x):
        self.activations = x
    
    def forward(self, leads):
        x = self.block1((leads))
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        # Register the hooks for Grad-CAM
        self.set_activations(x)
        h = x.register_hook(self.activations_hook)
        x = self.maxpool(x)
        combined = x.view(x.size(0), -1)
        combined = self.dropout(combined)
        x = self.fcRED(combined)
        output = x
        return output
    
    # Grad-CAM gradient hook functions
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self):
        return self.activations

# The 2D network that takes in an image as 12 layered channels
class Two_D_12channel(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = self.conv_block(c_in=12, c_out=8, dropout=0.2, kernel_size=3, stride=1, padding=0)     # 0.4 dropout
        self.block2 = self.conv_block(c_in=8, c_out=4, dropout=0.3, kernel_size=3, stride=1, padding=0)     # 0.5 dropout
        self.block5 = nn.Conv2d(4, 10, kernel_size=30) # 96
        self.fcRED = nn.Linear(10, 1)
        self.dropout = nn.Dropout2d(p=0.2)   # 0.4 dropout
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block

    # Grad-CAM gradient hook functions
    def activations_hook(self, grad):
        self.gradients = grad
    def set_activations(self, x):
        self.activations = x
    
    def forward(self, leads):
        print(leads.shape)
        x = self.block1(leads)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        # Register the hooks for Grad-CAM
        self.set_activations(x)
        h = x.register_hook(self.activations_hook)
        x = self.maxpool(x)
        print(x.shape)
        x = self.block5(x)
        print(x.shape)
        flattend = x.view(x.size(0), -1)
        print(x.shape)
        # flattend = self.dropout(flattend)
        x = self.fcRED(flattend)
        print(x.shape)
        sys.exit()
        output = x
        return output

        # Grad-CAM gradient hook functions
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self):
        return self.activations


def calc_acc(output, target, batch_size): #calculates the accuracy of 1 batch for the sake of printing the training accuracy per epoch
    correct = 0
    for i in range(batch_size):  
        if m(output[i]) < 0.5:
            if target[i].item() == 0:
                correct += 1
        else:
            if target[i].item() == 1:
                correct += 1
    return (correct/batch_size)*100

def create_tensorDataset(df):
    X_valid = df.iloc[:, 0:12]
    y_valid = df.iloc[:, 12]

    X_valid = X_valid.to_numpy()
    y_valid = y_valid.to_numpy()

    X_valid = X_valid.tolist()
    y_valid = y_valid.tolist()

    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    valid_ds = TensorDataset(X_valid, y_valid)

    return valid_ds

# For the 2D_12channel model the structure of the data has to be reshaped to layer the leads as 12 channels
def transform_df_to_12channel(df):
    end_arr = []
    for j in range(len(df)):
        arr = df.iloc[j]
        new_arr = []
        for i in range(len(arr)):
            if i < 12:
                holder = arr[i]
                holder1 = holder[0]
                new_arr.append(holder1)
            else:
                new_arr.append(arr[i])
        end_arr.append(new_arr)
    df_new = pd.DataFrame(end_arr)
    return df_new

if __name__ == "__main__":
    dir = os.getcwd()
    os.chdir(dir+'\\data\\validation')
    
    ans = input('Which model would you like to run for GradCAM?\n (A) \t 2D CNN median \n (B) \t 2D CNN single image\n (C) \t 2D CNN 12 channel\n')
    model_nr = input('Which of the 5 trained models would you like to use?\n0, 1, 2, 3 or 4\n')
    nr = input('Would you like have individual patient Grad-CAMs (0) or a single summed Grad-CAM (1) ?\n')
    if nr == '0':
        # Set this to false for summing every patients gradcam
        single_patient = True
    else:
        single_patient = False

    if ans == 'A':
        model = Two_D()
        model.load_state_dict(torch.load(dir+'\\models\\Grad_2D_median_{0}'.format(model_nr)), strict=False)
        model.eval()
        df = pd.read_pickle('median_2D_val')
        model_name = '2D_median'
        # Create tensor datasets from the data
        valid_ds = create_tensorDataset(df)
    elif ans == 'B':
        model = Two_D_single_img()
        model.load_state_dict(torch.load(dir+'\\models\\Grad_2D_one_img_{0}'.format(model_nr)), strict=False)
        model.eval()
        model_name = '2D_one_img'
        image_transforms = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()]) #(256, 256)
        # Create a dataset from the images
        ECG_dataset = datasets.ImageFolder(root = dir+'\\data\\validation\\testing_one_img', transform = image_transforms)
        # Define the  data loader
        valid_ds = torch.utils.data.DataLoader(ECG_dataset, batch_size=1)
    elif ans == 'C':
        model = Two_D_12channel()
        model.load_state_dict(torch.load(dir+'\\models\\Grad_2D_12channel_{0}'.format(model_nr)), strict=False)
        model.eval()
        df = pd.read_pickle('median_2D_val')
        model_name = '2D_12channel'
        df = transform_df_to_12channel(df)
        # Create tensor datasets from the data
        valid_ds = create_tensorDataset(df)

    if model_name == '2D_median':
        # Declare an empty heatmap with zeros for later accumulation of heatmaps
        accum_heatmap = []
        for e in range(12):
            accum_heatmap.append(np.zeros((417,510)))
        # The loop over the validation data
        for batch_idx, (data, target) in enumerate(valid_ds):
            # torch.cuda.set_device(0)
            # Set the data to cuda device for faster performance
            # data = data.to(device)
            # target = target.to(device)
            # Run the patient ECG through the model
            output = model(data.unsqueeze(0))
            output = torch.squeeze(output, -1)
            # Run the output backward through the network to get the gradients
            output[0].backward()

            # Get the gradients of the last concolutional layers per channel
            gradients0 = model.get_activations_gradient0()
            gradients1 = model.get_activations_gradient1()
            gradients2 = model.get_activations_gradient2()
            gradients3 = model.get_activations_gradient3()
            gradients4 = model.get_activations_gradient4()
            gradients5 = model.get_activations_gradient5()
            gradients6 = model.get_activations_gradient6()
            gradients7 = model.get_activations_gradient7()
            gradients8 = model.get_activations_gradient8()
            gradients9 = model.get_activations_gradient9()
            gradients10 = model.get_activations_gradient10()
            gradients11 = model.get_activations_gradient11()

            # Get the activations of the last convolutional layer per channel
            activations0 = model.get_activations0().detach()
            activations1 = model.get_activations1().detach()
            activations2 = model.get_activations2().detach()
            activations3 = model.get_activations3().detach()
            activations4 = model.get_activations4().detach()
            activations5 = model.get_activations5().detach()
            activations6 = model.get_activations6().detach()
            activations7 = model.get_activations7().detach()
            activations8 = model.get_activations8().detach()
            activations9 = model.get_activations9().detach()
            activations10 = model.get_activations10().detach()
            activations11 = model.get_activations11().detach()

            # Loop over the gradcam functions for each of the 12 channels/leads
            for channel in range(12):
                # Pool the gradients across the channels      
                vars()['pooled_gradients' + str(channel)] = torch.mean(vars()['gradients' + str(channel)], dim=[0, 2, 3])
                # Weight the channels by corresponding gradients
                for i in range(4):
                    vars()['activations' + str(channel)][:, i, :, :] *= vars()['pooled_gradients' + str(channel)][i]  
                # Average the channels of the activations
                vars()['heatmap' + str(channel)] = torch.mean(vars()['activations' + str(channel)], dim=1).squeeze()
                # Set the heatmaps to CPU
                vars()['heatmap' + str(channel)] = vars()['heatmap' + str(channel)].cpu()
                # Get the max of the heatmaps
                vars()['heatmap' + str(channel)] = np.maximum(vars()['heatmap' + str(channel)], 0)
                # Normalize the heatmaps
                vars()['heatmap' + str(channel)] /= torch.max(vars()['heatmap' + str(channel)])
                # Convert the heatmaps to an array
                vars()['heatmap' + str(channel)] = np.array(vars()['heatmap' + str(channel)])
                # Resize the heatmaps
                vars()['heatmap' + str(channel)] = cv2.resize( vars()['heatmap' + str(channel)], (510, 417))
                # Convert the heatmaps to int
                vars()['heatmap' + str(channel)] = np.uint8(255 * vars()['heatmap' + str(channel)])

            # Get the target classification
            if int(target.item()) == 0:
                diagnosis = 'negative'
            else:
                diagnosis = 'positive'

            # Dont plot a gradcam for each individual patient
            if single_patient == False:
                # Loop to accumulate each heatmap per patient for each lead
                for i in range(12):
                    accum_heatmap[i] = accum_heatmap[i] + vars()['heatmap' + str(i)]
                # If all heatmaps per all patients are accumulated
                if batch_idx == len(valid_ds)-1:
                    for i in range(12):
                        for j in range(417):
                            for k in range(510):
                                # Normalize the accum heatmap over all patients
                                accum_heatmap[i][j,k] = accum_heatmap[i][j,k]/(len(valid_ds)/2)
                        accum_heatmap[i] = accum_heatmap[i].astype(np.uint8)
                    # Read the average ECG median images
                    os.chdir(dir+'\\data\\images\\avg_images')
                    img0 = cv2.imread('0.png')
                    img1 = cv2.imread('1.png')
                    img2 = cv2.imread('2.png')
                    img3 = cv2.imread('3.png')
                    img4 = cv2.imread('4.png')
                    img5 = cv2.imread('5.png')
                    img6 = cv2.imread('6.png')
                    img7 = cv2.imread('7.png')
                    img8 = cv2.imread('8.png')
                    img9 = cv2.imread('9.png')
                    img10 = cv2.imread('10.png')
                    img11 = cv2.imread('11.png')

                    fin = []
                    # Loop to plot the heatmap ontop of the average median ECG
                    for i in range(12):
                        accum_heatmap[i] = cv2.applyColorMap(accum_heatmap[i], cv2.COLORMAP_JET)
                        accum_heatmap[i] = accum_heatmap[i] * 0.4
                        img =  vars()['img' + str(i)]
                        img = np.asarray(img, np.float64)
                        mapski = np.asarray(accum_heatmap[i], np.float64)
                        fin.append(cv2.addWeighted(mapski, 2, img, 0.6, 0))
                    # Write the images to a folder                            
                    os.chdir(dir+'\\Grad_CAM_results')
                    cv2.imwrite('./all_patients/I.jpg', fin[0])
                    cv2.imwrite('./all_patients/II.jpg', fin[1])
                    cv2.imwrite('./all_patients/III.jpg', fin[2])
                    cv2.imwrite('./all_patients/aVR.jpg', fin[3])
                    cv2.imwrite('./all_patients/aVL.jpg', fin[4])
                    cv2.imwrite('./all_patients/aVF.jpg', fin[5])
                    cv2.imwrite('./all_patients/V1.jpg', fin[6])
                    cv2.imwrite('./all_patients/V2.jpg', fin[7])
                    cv2.imwrite('./all_patients/V3.jpg', fin[8])
                    cv2.imwrite('./all_patients/V4.jpg', fin[9])
                    cv2.imwrite('./all_patients/V5.jpg', fin[10])
                    cv2.imwrite('./all_patients/V6.jpg', fin[11])

            else: # Plot every patient activation map individually
                # For getting the correct patient ID
                if int(target.item()) == 0:
                    diagnosis = 'negative'
                    pp = 0
                    for ppp in os.listdir(dir+'\\data\\images\\validation\\negative'): #these 2 loops are needed to get the correct patient file name into this code
                        if pp == batch_idx:
                            patient_ID = ppp
                        pp += 1
                else:
                    diagnosis = 'positive'
                    pp = len(valid_ds)/2
                    for ppp in os.listdir(dir+'\\data\\images\\validation\\positive'):
                        if pp == batch_idx: # Get the patient ID of the patient currently in the GradCAM loop
                            patient_ID = ppp
                        pp += 1

                # Use the acquired patient ID to load in the correct ECG images as underlay of the gradCAM
                os.chdir(dir+'\\data\\images\\validation\\{0}\\{1}'.format(diagnosis, patient_ID))
                img0 = cv2.imread('0.png')
                img1 = cv2.imread('1.png')
                img2 = cv2.imread('2.png')
                img3 = cv2.imread('3.png')
                img4 = cv2.imread('4.png')
                img5 = cv2.imread('5.png')
                img6 = cv2.imread('6.png')
                img7 = cv2.imread('7.png')
                img8 = cv2.imread('8.png')
                img9 = cv2.imread('9.png')
                img10 = cv2.imread('10.png')
                img11 = cv2.imread('11.png')

                fin = []
                # Loop to plot the heatmap ontop of the average median ECG
                for i in range(12):
                    vars()['heatmap' + str(i)] = cv2.applyColorMap(vars()['heatmap' + str(i)], cv2.COLORMAP_JET)
                    vars()['heatmap' + str(i)] = vars()['heatmap' + str(i)] * 0.4
                    img =  vars()['img' + str(i)]
                    img = np.asarray(img, np.float64)
                    mapski = np.asarray(vars()['heatmap' + str(i)], np.float64)
                    fin.append(cv2.addWeighted(mapski, 2, img, 0.6, 0))

                try:
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients\\{0}\\{1}'.format(diagnosis, patient_ID))
                except:
                    if batch_idx == 0:
                        print('NOTICE: overwriting previous individual patient plots')
                    shutil.rmtree(dir+'\\individual_patients\\{0}\\{1}'.format(diagnosis, patient_ID))
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients\\{0}\\{1}'.format(diagnosis, patient_ID))
                os.chdir(dir+'\\Grad_CAM_results\\individual_patients')
                cv2.imwrite('./{0}/{1}/I.jpg'.format(diagnosis, patient_ID), fin[0])
                cv2.imwrite('./{0}/{1}/II.jpg'.format(diagnosis, patient_ID), fin[1])
                cv2.imwrite('./{0}/{1}/III.jpg'.format(diagnosis, patient_ID), fin[2])
                cv2.imwrite('./{0}/{1}/aVR.jpg'.format(diagnosis, patient_ID), fin[3])
                cv2.imwrite('./{0}/{1}/aVL.jpg'.format(diagnosis, patient_ID), fin[4])
                cv2.imwrite('./{0}/{1}/aVF.jpg'.format(diagnosis, patient_ID), fin[5])
                cv2.imwrite('./{0}/{1}/V1.jpg'.format(diagnosis, patient_ID), fin[6])
                cv2.imwrite('./{0}/{1}/V2.jpg'.format(diagnosis, patient_ID), fin[7])
                cv2.imwrite('./{0}/{1}/V3.jpg'.format(diagnosis, patient_ID), fin[8])
                cv2.imwrite('./{0}/{1}/V4.jpg'.format(diagnosis, patient_ID), fin[9])
                cv2.imwrite('./{0}/{1}/V5.jpg'.format(diagnosis, patient_ID), fin[10])
                cv2.imwrite('./{0}/{1}/V6.jpg'.format(diagnosis, patient_ID), fin[11])

    elif model_name == '2D_one_img': # Run the one image gradcam
        accum_heatmap = np.zeros((512,384))
        # If you want Grad-CAM activation (Does not work for VGG16)
        # Loop over the validation set
        for batch_idx, data in enumerate(valid_ds, 0):
            input, target = data
            # Run the model
            output = model(input)
            target = target.float()
            target = torch.reshape(target, (1, 1))

            output[0].backward()
            gradients = model.get_activations_gradient()
            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            # get the activations of the last convolutional layer
            activations = model.get_activations().detach()
            for i in range(4):
                activations[:, i, :, :] *= pooled_gradients[i]
            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = heatmap.cpu()
            heatmap = np.maximum(heatmap, 0)
            # normalize the heatmap
            heatmap /= torch.max(heatmap)
            heatmap = np.array(heatmap)
            # Gets the patient ID's
            if int(target.item()) == 0:
                diagnosis = 'negative'
                pp = 0
                for ppp in os.listdir(dir+'\\data\\images\\validation\\negative'): #these 2 loops are needed to get the correct patient file name into this code
                    if pp == batch_idx:
                        patient_ID = ppp
                    pp += 1
            else:
                diagnosis = 'positive'
                pp = 24
                for ppp in os.listdir(dir+'\\data\\images\\validation\\negative'):
                    if pp == batch_idx:
                        patient_ID = ppp
                    pp += 1
            # Resize the heatmap to the original image
            heatmap = cv2.resize(heatmap, (384, 512))
            heatmap = np.uint8(255 * heatmap)


            # Runs only the summed activation maps
            if single_patient == False:  
                # Sum the heatmaps for each patient
                accum_heatmap = accum_heatmap + heatmap 

                # If all negative patients have been run
                if batch_idx == (len(valid_ds)-1):
                    for j in range(512):
                        for k in range(384):
                            accum_heatmap[j,k] = accum_heatmap[j,k]/(len(valid_ds)/2) #normalize each pixel value
                    accum_heatmap = accum_heatmap.astype(np.uint8)
                    # Read in the original image
                    os.chdir(dir+'\\data\\training\\training_one_img\\negative')
                    img = cv2.imread('DPP6_258_6d40feb38f.png')
                    fin = []
                    # Apply the interpolation of the heatmap
                    accum_heatmap = cv2.applyColorMap(accum_heatmap, cv2.COLORMAP_JET)
                    accum_heatmap = accum_heatmap * 0.4
                    img = np.asarray(img, np.float64)
                    mapski = np.asarray(accum_heatmap, np.float64)
                    # Add the mean heartbeat as background to interpolate the heatmap on
                    fin = cv2.addWeighted(mapski, 2, img, 0.6, 0) 
                    os.chdir(dir+'\\Grad_CAM_results')
                    cv2.imwrite('./one_img_results/GRADCAM.jpg', fin)                       
                            
            else: # Plot every patients individually
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + img
                try:
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                except:
                    if batch_idx == 0:
                        print('NOTICE: overwriting previous individual patient plots')
                    shutil.rmtree(dir+'\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                cv2.imwrite('./activation_maps/{0}/{1}/I.jpg'.format(diagnosis, patient_ID), superimposed_img)

    elif model_name == '2D_12channel':
         # Declare an empty heatmap with zeros for later accumulation of heatmaps
        accum_heatmap = np.zeros((417,510))
        # Loop over the validation set
        for batch_idx, (data, target) in enumerate(valid_ds):
            # Run the patient ECG through the model
            output = model(data.unsqueeze(0))
            output = torch.squeeze(output, -1)
            # Run the output backward through the network to get the gradients
            output[0].backward()

            gradients = model.get_activations_gradient()
            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            # get the activations of the last convolutional layer
            activations = model.get_activations().detach()
            for i in range(4):
                activations[:, i, :, :] *= pooled_gradients[i]
            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = heatmap.cpu()
            heatmap = np.maximum(heatmap, 0)
            # normalize the heatmap
            heatmap /= torch.max(heatmap)
            heatmap = np.array(heatmap)
            # Gets the patient ID's
            if int(target.item()) == 0:
                diagnosis = 'negative'
                pp = 0
                for ppp in os.listdir(dir+'\\data\\images\\validation\\negative'): #these 2 loops are needed to get the correct patient file name into this code
                    if pp == batch_idx:
                        patient_ID = ppp
                    pp += 1
            else:
                diagnosis = 'positive'
                pp = 24
                for ppp in os.listdir(dir+'\\data\\images\\validation\\negative'):
                    if pp == batch_idx:
                        patient_ID = ppp
                    pp += 1
            # Resize the heatmap to the original image
            heatmap = cv2.resize(heatmap, (510, 417))
            heatmap = np.uint8(255 * heatmap)


            # Runs only the summed activation maps
            if single_patient == False:  
                # Sum the heatmaps for each patient
                accum_heatmap = accum_heatmap + heatmap 

                # If all negative patients have been run
                if batch_idx == (len(valid_ds)-1):
                    for j in range(417):
                        for k in range(510):
                            accum_heatmap[j,k] = accum_heatmap[j,k]/(len(valid_ds)/2) #normalize each pixel value
                    accum_heatmap = accum_heatmap.astype(np.uint8)
                    # Read in the original image
                    os.chdir(dir+'\\data\\images\\avg_images')
                    img = cv2.imread('0.png')
                    fin = []
                    # Apply the interpolation of the heatmap
                    accum_heatmap = cv2.applyColorMap(accum_heatmap, cv2.COLORMAP_JET)
                    accum_heatmap = accum_heatmap * 0.4
                    img = np.asarray(img, np.float64)
                    mapski = np.asarray(accum_heatmap, np.float64)
                    # Add the mean heartbeat as background to interpolate the heatmap on
                    fin = cv2.addWeighted(mapski, 2, img, 1.5, 0) 
                    os.chdir(dir+'\\Grad_CAM_results')
                    cv2.imwrite('./12channel_results/GRADCAM.jpg', fin)                       
                            
            else: # Plot every patients individually
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + img
                try:
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                except:
                    if batch_idx == 0:
                        print('NOTICE: overwriting previous individual patient plots')
                    shutil.rmtree(dir+'\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                    os.mkdir(dir+'\\Grad_CAM_results\\individual_patients_one_img\\{0}\\{1}'.format(diagnosis, patient_ID))
                cv2.imwrite('./activation_maps/{0}/{1}/I.jpg'.format(diagnosis, patient_ID), superimposed_img)

    # else:
    correct = 0
    incorrect = 0
    FP = 0
    FN = 0      
    TP = 0
    TN = 0
    y_true = []
    y_score = []
    for batch_idx, (data, target) in enumerate(valid_ds):
        # data = data.to(device)
        # target = target.to(device)

        if model_name == '2D_median' or model_name == '2D_12channel':
            data = data.unsqueeze(0)
        else:
            target = target.float()
            target = torch.reshape(target, (1, 1))

        output = model(data)

        if model_name == '2D_one_img':
            output = torch.squeeze(output, -1)
        y_true.append(target.item())
        y_score.append(m(output).item())
        test_acc = calc_acc(output, target.unsqueeze(0), 1)
        if m(output) < 0.5:  #model output says negative  No dpp6
            if test_acc == 100:  #D # it is negative  
                correct += 1
                TN += 1
            else:        #C            # it is actually positive
                incorrect += 1
                FN += 1
        else:                           # yes dpp6
            if test_acc == 100: 
                correct += 1
                TP += 1
            else:   #B
                incorrect += 1
                FP += 1
    
    y_score = [round(num, 2) for num in y_score]
    print('validation: ', round(correct/len(list(valid_ds)),2), '  Spec: ', round(TN/(TN+FP)*100,2), 'Sens: ', round(TP/(TP+FN)*100,2))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_2D_12channel validation - model 4')
    plt.legend(loc="lower right")
    # plt.show()



    

