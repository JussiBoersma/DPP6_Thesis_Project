import csv
import os
from cv2 import mean
import scipy
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import lstm, nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from multiprocessing import cpu_count
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from matplotlib.collections import PolyCollection
from sklearn import metrics
from scipy import interp
import sys
import statistics as st
import torchvision.models as models

model_vgg = models.vgg16(pretrained=True)

# The definition of the 1D CNN model architecture
class One_D_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_block(12, 24, 0.4, kernel_size=3)
        self.conv2 = self.conv_block(24, 12, 0.5, kernel_size=3)      # network for the median beat
        self.conv3 = self.conv_block(12, 6, 0.4, kernel_size=3)  
        self.fc1 = nn.Linear(210, 20) # 210 for median, 360 for 3beat rhythm
        self.fc2 = nn.Linear(20,1)
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )        
        return seq_block

    def forward(self, median):  # type: ignore
        x = self.conv1(median)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        fused = self.fc2(x)
        output = fused
        return output

# The definition of the 3beats model architecture
class One_D_3beats_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_block(12, 24, 0.4, kernel_size=3)
        self.conv2 = self.conv_block(24, 12, 0.5, kernel_size=3)      # network for the median beat
        self.conv3 = self.conv_block(12, 6, 0.4, kernel_size=3)  
        self.fc1 = nn.Linear(360, 20) # 210 for median, 360 for 3beat rhythm
        self.fc2 = nn.Linear(20,1)
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )        
        return seq_block

    def forward(self, median):  # type: ignore
        x = self.conv1(median)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        fused = self.fc2(x)
        output = fused
        return output

# The definition of the LSTM architecture
class One_D_LSTM(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden = 25, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
            dropout = 0.55
        )
        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)

# The definition of the Male/Female model architecture
class One_D_MF_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_block(12, 12, kernel_size=7)
        self.conv2 = self.conv_block(12, 12, kernel_size=5)   
        self.conv3 = self.conv_block(12, 16, kernel_size=3)  
        self.conv4 = self.conv_block(16, 12, kernel_size=3)  
        self.conv5 = self.conv_block(12, 12, kernel_size=3) 
        self.conv6 = self.conv_block(12, 6, kernel_size=3) 
        self.dens1 = self.dense_block(12, 12, 0.1)
        self.dens2 = self.dense_block(12, 1, 0.1)
    
    def conv_block(self, c_in, c_out,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )        
        return seq_block

    def dense_block(self, c_in, c_out, dropout,  **kwargs):
        dense_block = nn.Sequential(
            nn.Linear(in_features=c_in, out_features=c_out),
            nn.BatchNorm1d(num_features=c_out),
            # nn.ReLU(),
            nn.Dropout(dropout)
        )        
        return dense_block

    def forward(self, median):  # type: ignore
        x = self.conv1(median)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.dens1(x)
        x = self.dens2(x)
        output = x
        return output

# The definition of the median betti model architecture
class One_D_Med_Betti(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_block(24, 48, 0.4, kernel_size=3)
        self.conv2 = self.conv_block(48, 18, 0.5, kernel_size=3)      # network for the median beat and betti curve
        self.conv3 = self.conv_block(18, 10, 0.4, kernel_size=3)  
        self.fc1 = nn.Linear(350, 20) # 438
        self.fc2 = nn.Linear(20, 1)
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )        
        return seq_block

    def forward(self, median): 
        x = self.conv1(median)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        fused = self.fc1(x)
        fused = self.fc2(fused)
        output = fused
        return output

# The network architecture definition for 2D CNN combined image
class Two_D_single_img(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.conv_block(c_in=3, c_out=12, dropout=0.2, kernel_size=3, stride=1, padding=1)     
        self.block2 = self.conv_block(c_in=12, c_out=8, dropout=0.2, kernel_size=3, stride=1, padding=1)     
        self.block3 = self.conv_block(c_in=8, c_out=4, dropout=0.2, kernel_size=3, stride=1, padding=1)    
        self.fcRED = nn.Linear(3072, 1)  #768  1920
        self.dropout = nn.Dropout2d(p=0.1)   # 0.4 dropout
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block
    
    def forward(self, leads):
        x = self.block1((leads))
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.maxpool(x)
        combined = x.view(x.size(0), -1)
        combined = self.dropout(combined)
        x = self.fcRED(combined)
        output = x
        return output

    def get_final_activations(self):
        return self.final_activations

# The network architecture definition for the 2D CNN 
class Two_D(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = self.conv_block(c_in=1, c_out=3, dropout=0.4, kernel_size=3, stride=1, padding=1)     # 0.4 dropout
        self.block2 = self.conv_block(c_in=3, c_out=4, dropout=0.5, kernel_size=3, stride=1, padding=1)     # 0.5 dropout
        self.block5 = nn.Conv2d(48, 10, kernel_size=32) # 96
        self.fcRED = nn.Linear(10, 1)
        self.dropout = nn.Dropout2d(p=0.4)   # 0.4 dropout
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )        
        return seq_block
        
   # Function can be used for TSNE visualization of the final layer activations
    # def set_final_activations(self, x):
    #     self.final_activations = x
    
    def forward(self, leads):
        x0 = self.block1((leads[:,0]))
        x0 = self.maxpool(x0)
        x0 = self.block2(x0)
        x0 = self.maxpool(x0)

        x1 = self.block1((leads[:,1]))
        x1 = self.maxpool(x1)
        x1 = self.block2(x1) 
        x1 = self.maxpool(x1)

        x2 = self.block1((leads[:,2]))
        x2 = self.maxpool(x2)
        x2 = self.block2(x2) 
        x2 = self.maxpool(x2)

        x3 = self.block1((leads[:,3]))
        x3 = self.maxpool(x3)
        x3 = self.block2(x3) 
        x3 = self.maxpool(x3)

        x4 = self.block1((leads[:,4]))
        x4 = self.maxpool(x4)
        x4 = self.block2(x4) 
        x4 = self.maxpool(x4)

        x5 = self.block1((leads[:,5]))
        x5 = self.maxpool(x5)
        x5 = self.block2(x5) 
        x5 = self.maxpool(x5)

        x6 = self.block1((leads[:,6]))
        x6 = self.maxpool(x6)
        x6 = self.block2(x6)
        x6 = self.maxpool(x6) 

        x7 = self.block1((leads[:,7]))
        x7 = self.maxpool(x7)
        x7 = self.block2(x7) 
        x7 = self.maxpool(x7)

        x8 = self.block1((leads[:,8]))
        x8 = self.maxpool(x8)
        x8 = self.block2(x8) 
        x8 = self.maxpool(x8)

        x9 = self.block1((leads[:,9]))
        x9 = self.maxpool(x9)
        x9 = self.block2(x9) 
        x9 = self.maxpool(x9)

        x10 = self.block1((leads[:,10]))
        x10 = self.maxpool(x10)
        x10 = self.block2(x10) 
        x10 = self.maxpool(x10)

        x11 = self.block1((leads[:,11]))
        x11 = self.maxpool(x11)
        x11 = self.block2(x11) 
        x11 = self.maxpool(x11)
        
        combined = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], dim=1)
        combined = self.block5(combined)
        combined = combined.view(combined.size(0), -1)
        # self.set_final_activations(combined)
        combined = self.dropout(combined)
        x = self.fcRED(combined)
        output = x
        return output

# The definition of the VGG16 architecture
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.layer1 = nn.Linear(1000,1)
        self.net = model_vgg
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        x1 = self.net(x)
        y = self.layer1(x1)
        return y

if __name__ == "__main__":
    pass










