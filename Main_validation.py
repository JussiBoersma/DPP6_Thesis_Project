import csv
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from multiprocessing import cpu_count
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold
import gudhi
import statistics as st
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import sys
import math
from Model_classes import One_D_CNN, One_D_3beats_CNN, One_D_LSTM, One_D_Med_Betti, One_D_MF_CNN, Two_D, Two_D_single_img, VGG16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = nn.Sigmoid()

# Go from validation data to tensordataset
def create_tensorDataset(df):
    # The betti median combined has more columns
    if df.shape[1] > 15:
        num_columns = 24
    else:
        num_columns = 12
    X_valid = df.iloc[:, 0:num_columns]
    y_valid = df.iloc[:, num_columns]
    X_valid = X_valid.to_numpy()
    y_valid = y_valid.to_numpy()
    X_valid = X_valid.tolist()
    y_valid = y_valid.tolist()
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    valid_ds = TensorDataset(X_valid, y_valid)
    return valid_ds


if __name__ == "__main__":
    dir = os.getcwd()
    os.chdir(dir+'\\data\\validation')

    model_type = input("What model type would you like to validate? \n (A) \t 1D_median \n (B) \t 1D_lstm \n (C) \t 1D_transfer_median \n (D) \t 1D_3beats \n (E) \t 1D_Median_Betti \n (F) \t 1D_Wave_Stretch \n (G) \t 2D_median \n (H) \t 2D_median_one_img \n (I) \t 2D_morphed \n (J) \t VGG16 \n")
    
    acc_arr, spec_arr, sens_arr, auc_arr = ([] for i in range(4))
    for cnt in range(5):
        # Get the model type the user wants to run
        if model_type == 'A':
            model = torch.load(dir+'\\models\\1D_median_{0}'.format(cnt))
            df = pd.read_pickle('median_1D_val')
            model_name = '1D_median'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'B':
            model = torch.load(dir+'\\models\\1D_lstm_{0}'.format(cnt))
            df = pd.read_pickle('median_1D_val')
            model_name = '1D_lstm'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'C':
            model = torch.load(dir+'\\models\\1D_transfer_median_{0}'.format(cnt))
            df = pd.read_pickle('median_1D_val')
            model_name = '1D_transfer_median'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'D':
            model = torch.load(dir+'\\models\\1D_3beats_{0}'.format(cnt))
            df = pd.read_pickle('3beat_1D_val')
            model_name = '1D_3beats'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'E':
            model = torch.load(dir+'\\models\\1D_median_betti_{0}'.format(cnt))
            df_median = pd.read_pickle('median_1D_val')
            df_betti = pd.read_pickle('betti_1D_val')
            model_name = '1D_median_betti'
            df_median = df_median.reset_index(drop=True)
            df_median = df_median.drop(df_median.columns[13], axis = 1)
            frames = [df_betti, df_median]
            df = pd.concat(frames, axis = 1)
            df.columns = ['Three_b', 'aVR_b', 'aVL_b', 'aVF_b', 'I_b', 'II_b', 'V1_b', 'V2_b', 'V3_b', 'V4_b', 'V5_b', 'V6_b', 'remove_label','Three', 'aVR', 'aVL', 'aVF', 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'label']
            df = df.drop(df.columns[12], axis=1)
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'F':
            model = torch.load(dir+'\\models\\1D_wave_stretch_{0}'.format(cnt))
            df = pd.read_pickle('median_1D_val')
            model_name = '1D_wave_stretch'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 8
        elif model_type == 'G':
            model = torch.load(dir+'\\models\\2D_median_{0}'.format(cnt))
            df = pd.read_pickle('median_2D_val')
            model_name = '2D_median'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 10
        elif model_type == 'H':
            model = torch.load(dir+'\\models\\2D_one_img_{0}'.format(cnt))
            model_name = '2D_one_img'
            n_epochs = 10
            image_transforms = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()]) #(256, 256)
            # Create a dataset from the images
            ECG_dataset = datasets.ImageFolder(root = dir+'\\data\\validation\\testing_one_img', transform = image_transforms)
            # Define the  data loader
            valid_ds = torch.utils.data.DataLoader(ECG_dataset, batch_size=1)
        elif model_type == 'I':
            model = torch.load(dir+'\\models\\2D_morphed_{0}'.format(cnt))
            df = pd.read_pickle('medianMorph_2D val')
            model_name = '2D_morphed'
            # Create tensor datasets from the data
            valid_ds = create_tensorDataset(df)
            n_epochs = 10
        elif model_type == 'J':
            model = torch.load(dir+'\\models\\2D_VGG16_{0}'.format(cnt))
            model_name = '2D_VGG16'
            n_epochs = 10
            image_transforms = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()]) #(256, 256)
            # Create a dataset from the images
            ECG_dataset = datasets.ImageFolder(root = dir+'\\data\\validation\\testing_one_img', transform = image_transforms)
            # Define the  data loader
            valid_ds = torch.utils.data.DataLoader(ECG_dataset, batch_size=1)

        # For the Grad-CAM able models ask if you want to run Grad-CAM
        # if model_name == '2D_median' or model_name == '2D_one_img':
        #     ans = input('Do you want to run Grad-CAM?\n (A) \t Yes \n (B) \t No \n')
        #     if ans == 'A':
        #         run_GradCAM = True
        #     else:
        #         run_GradCAM = False
        
        correct, incorrect, FP, FN, TP, TN = (0 for i in range(6))
        y_true, y_score = [], []
        patient_hist = np.zeros(48)
        with torch.no_grad():
            model.eval()
            # Loop over the validation set
            for batch_idx, (data, target) in enumerate(valid_ds):
                data = data.to(device)
                target = target.to(device)
                if model_name != '2D_VGG16' and model_name != '2D_one_img':
                    data = data.unsqueeze(0)
                else:
                    target = target.float()
                    target = torch.reshape(target, (1, 1))
                # Run the model
                output = model(data)
                # Save the score and target for plotting the AUC
                y_true.append(target.item())
                y_score.append(m(output).item())
                if m(output).item() < 0.5:  #model output says negative  No dpp6
                    if target.item() == 0:  #D # it is negative  
                        correct += 1
                        TN += 1 
                    else:        #C            # it is actually positive
                        incorrect += 1
                        patient_hist[batch_idx] += 1
                        FN += 1
                else:                           # yes dpp6
                    if target.item() == 1:  # true positive
                        correct += 1
                        TP += 1
                    else:   
                        incorrect += 1
                        patient_hist[batch_idx] += 1
                        FP += 1
            np.save(r'C:\Users\jussi\Documents\Master Thesis\Code_Refactored\data\model_{0}'.format(cnt), patient_hist)
            # Prepare the AUC metrics
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
            roc_auc = metrics.auc(fpr, tpr)
            # Plot the AUC
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Roc {0} validation - model {1}'.format(model_name, cnt))
            plt.legend(loc="lower right")
            # plt.show()
            # Print the validation metrics
            print('model_{0}: \t'.format(cnt), 'validation: ', round(correct/len(list(valid_ds))*100,2),  ' \t Sens: ', round(TP/(TP+FN)*100,2), ' \t Spec: ', round(TN/(TN+FP)*100,2), '\t AUC: ', round(roc_auc,2))

            acc_arr.append(round(correct/len(list(valid_ds))*100,2))
            spec_arr.append(round(TN/(TN+FP)*100,2))
            sens_arr.append(round(TP/(TP+FN)*100,2))
            auc_arr.append(roc_auc)      

    acctot, spectot, senstot, auctot = (0 for i in range(4))
    # Sum the training metrics over the 5 folds
    for u in range(5):
        acctot += acc_arr[u]
        spectot += spec_arr[u]
        senstot += sens_arr[u]
        auctot += auc_arr[u]    
    # Print the mean training metrics
    print('\naccuracy: ', round(acctot/5, 2), '\t Sens: ', round(senstot/5, 2), ' \t\t Spec: ', round(spectot/5, 2), '\t\t AUC: ', round(auctot/5, 2))  #63 for equal, 78 for unequal
    print('acc std: ', round(st.stdev(acc_arr),2), '\t\tSens std: ', round(st.stdev(sens_arr),2), '\t Spec std: ', round(st.stdev(spec_arr),2), '\t AUC std: ', round(st.stdev(auc_arr),2))
    
    
