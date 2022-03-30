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
from torch._C import dtype
from torch.nn import functional as F
from torch.nn.modules.pooling import AvgPool2d, MaxPool2d
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
import statistics as st
from matplotlib.collections import PolyCollection
from scipy import interp
from Model_classes import One_D_CNN, One_D_3beats_CNN, One_D_LSTM, One_D_Med_Betti, One_D_MF_CNN, Two_D, Two_D_12channel, Two_D_single_img, VGG16, Two_D_12channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The sigmoid function to run the model output through
m = nn.Sigmoid()

# Reset the weights of the model to prevent training on validation data within the 5 folds
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.LSTM) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Go from the data to tensor datasets for the model input
def create_tensors_folds(data, train_ids, valid_ids):
    if data.shape[1] == 25:
        size = 24
    else:
        size = 12

    X_train = data.iloc[train_ids, 0:size]
    X_valid = data.iloc[valid_ids, 0:size]
    y_train = data.iloc[train_ids, size]
    y_valid = data.iloc[valid_ids, size]

    X_train = X_train.to_numpy()
    X_valid = X_valid.to_numpy()
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()

    X_train = X_train.tolist()
    X_valid = X_valid.tolist()
    y_train = y_train.tolist()
    y_valid = y_valid.tolist()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)

    return train_ds, valid_ds

# Calculates the accuracy of 1 batch for the sake of printing the training accuracy per epoch
def calc_acc(output, target, batch_size):
    correct = 0
    for i in range(batch_size):  
        if m(output[i]) < 0.5:
            if target[i].item() == 0:
                correct += 1
        else:
            if target[i].item() == 1:
                correct += 1
    return (correct/batch_size)*100

# The training loop
def train_model(model, train_ds, valid_ds, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

    # for printing the number of parameters in the network
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters]) 
    # print(params)
    # sys.exit()

    model.cuda()
    cross_entropy_loss = nn.BCEWithLogitsLoss()
    loss = {'train': [], "val": []}
    accuracy = {'train': [], "val": []}
    # The loop over the amount of epochs
    for i in range(n_epochs):
        train_loss_tot = 0
        train_acc_tot = 0
        batch_size = 8
        model.train()
        # Prepare mini batch training
        permutation = torch.randperm(len(train_ds))
        # Loop over the mini batches
        for j in range(0,len(train_ds), batch_size):
            # Select a batch from the data
            indices = permutation[j:j+batch_size]
            batch = train_ds[indices]
            data = batch[0]
            target = batch[1]
            # Reshape the target
            target = torch.reshape(target, (batch_size, 1))
            # Set the data and target to device for CUDA running
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()   
            # Run the model
            output = model(data)

            #------------------------------------------------------------------------------------------------
            # 1D CNN Reshape the output
            if 'Two' in model.__class__.__name__:
                # 2D CNN Reshape the output
                output = output.squeeze(-1)
                target = target.squeeze(-1)    
            #------------------------------------------------------------------------------------------------
            
            # Calculate the loss
            train_loss = cross_entropy_loss(output, target)
            # Get the training acc over the batch
            train_acc = calc_acc(output, target, batch_size)
            train_loss.backward()
            optimizer.step()
            # Save the training metrics for later plotting
            train_loss_tot += train_loss.item()
            train_acc_tot += train_acc
        # For the validation part
        test_loss_tot, test_acc_tot = 0, 0
        y_true, y_score = [], []
        with torch.no_grad():
            model.eval()
            # Loop over the validation set
            for batch_idx, (data_v, target_v) in enumerate(valid_ds):
                # Set the data and target to device for CUDA running
                data_v = data_v.to(device)
                target_v = target_v.to(device)

                #------------------------------------------------------------------------------------------------
                # 1D CNN
                if 'One' in model.__class__.__name__:
                    target_v = torch.reshape(target_v, (1, 1))
                #------------------------------------------------------------------------------------------------
                
                data_v = data_v.unsqueeze(0)
                # Run the model
                output_v = model(data_v)
                if 'One' not in model.__class__.__name__:
                    output_v = torch.squeeze(output_v, -1)
                # Save the output and target for the AUC plotting
                y_true.append(target_v.item())
                y_score.append(m(output_v).item())

                # Calculate the loss
                if 'One' not in model.__class__.__name__:
                    loss_v = cross_entropy_loss(output_v, target_v.unsqueeze(0))
                    # Calculate the accuracy over the batch
                    test_acc = calc_acc(output_v, target_v.unsqueeze(0), 1)
                else:
                    loss_v = cross_entropy_loss(output_v, target_v)
                    # Calculate the accuracy over the batch
                    test_acc = calc_acc(output_v, target_v, 1)
                
                # Save the training metrics for later plotting
                test_loss_tot += loss_v.item()
                test_acc_tot += test_acc
        # Save the mean training metrics per epoch
        loss['train'].append(train_loss_tot/(len(train_ds)/batch_size))
        loss['val'].append(test_loss_tot/len(valid_ds))
        accuracy['train'].append(train_acc_tot/(len(train_ds)/batch_size))
        accuracy['val'].append(test_acc_tot/len(valid_ds))
    return model, loss, accuracy

# The training loop
def train_one_img_model(model, trainloader, testloader, n_epochs):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

    # For printing the number of parameters in the network
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters]) 
    # print(params)
    # sys.exit()

    model.cuda()
    cross_entropy_loss = nn.BCEWithLogitsLoss()
    loss = {'train': [], "val": []}
    accuracy = {'train': [], "val": []}
    for i in range(n_epochs):
        train_loss_tot = 0
        train_acc_tot = 0
        model.train()
        # Loop over the trainloader
        for j, data in enumerate(trainloader, 0):
            input, target = data
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # Run the model
            output = model(input)
            target = target.float()
            # Reshape the output
            target = torch.reshape(target, (8, 1))
            # Caculate the loss
            train_loss = cross_entropy_loss(output, target)
            train_loss.backward()
            optimizer.step()       
            # Calculate the accuracy over the batch
            train_acc = calc_acc(output, target, 8)
            # Save the training metrics
            train_loss_tot += train_loss.item()
            train_acc_tot += train_acc
        # for the validation part
        test_loss_tot, test_acc_tot = 0, 0 
        y_true, y_score = [], []
        with torch.no_grad():
            model.eval()
            # The validation loop
            for j, data_v in enumerate(testloader, 0):
                input_v, target_v = data_v
                input_v = input_v.to(device)
                target_v = target_v.to(device)
                # Run the model
                output_v = model(input_v)
                target_v = target_v.float()
                target_v = torch.reshape(target_v, (1, 1))
                # Calculate the loss
                loss_v = cross_entropy_loss(output_v, target_v)
                # Save the output and target for the AUC plot
                y_true.append(target_v.item())
                y_score.append(m(output_v).item())
                # Calculate the accuracy over the batch
                test_acc = calc_acc(output_v, target_v, 1)
                # Save the validation metrics for plotting
                test_loss_tot += loss_v.item()
                test_acc_tot += test_acc
        # Average the training metrics over all patients
        loss['train'].append(train_loss_tot/len(trainloader))
        loss['val'].append(test_loss_tot/len(testloader))
        accuracy['train'].append(train_acc_tot/len(trainloader))
        accuracy['val'].append(test_acc_tot/len(testloader))
    return model, loss, accuracy

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
    os.chdir(dir+'\\data\\training')

    model_type = input("What model type would you like to train? \n (A) \t 1D_median \n (B) \t 1D_lstm \n (C) \t 1D_Male_Female \n (D) \t 1D_transfer_median \n (E) \t 1D_3beats \n (F) \t 1D_Median_Betti \n (G) \t 1D_Wave_Stretch \n (H) \t 2D_median \n (I) \t 2D_median_one_img \n (J) \t 2D_12channel \n (K) \t 2D_morphed \n (L) \t VGG16 \n")
    # Get the model type the user wants to run
    if model_type == 'A':
        model = One_D_CNN()
        df = pd.read_pickle('median_1D_train')
        model_name = '1D_median'
        n_epochs = 8
    elif model_type == 'B':
        model = One_D_LSTM(300, 1)
        df = pd.read_pickle('median_1D_train')
        model_name = '1D_lstm'
        n_epochs = 8
    elif model_type == 'C':
        model = One_D_MF_CNN()
        df = pd.read_pickle('male_female_1D_train')
        model_name = '1D_MF'
        n_epochs = 8
    elif model_type == 'D':
        # Load in one of the 5 transfer learning M/F trained models
        model = torch.load(dir+'\\models\\1D_MF_0')
        df = pd.read_pickle('median_1D_train')
        model_name = '1D_transfer_median'
        n_epochs = 8
    elif model_type == 'E':
        model = One_D_3beats_CNN()
        df = pd.read_pickle('3beat_1D_train')
        model_name = '1D_3beats'
        n_epochs = 8
    elif model_type == 'F':
        model = One_D_Med_Betti()
        df_median = pd.read_pickle('median_1D_train')
        df_betti = pd.read_pickle('betti_1D_train')
        model_name = '1D_median_betti'
        df_median = df_median.drop(df_median.columns[13], axis = 1)
        frames = [df_betti, df_median]
        df = pd.concat(frames, axis = 1)
        df.columns = ['Three_b', 'aVR_b', 'aVL_b', 'aVF_b', 'I_b', 'II_b', 'V1_b', 'V2_b', 'V3_b', 'V4_b', 'V5_b', 'V6_b', 'remove_label','Three', 'aVR', 'aVL', 'aVF', 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'label']
        df = df.drop(df.columns[12], axis=1)
        n_epochs = 8
    elif model_type == 'G':
        model = One_D_CNN()
        df = pd.read_pickle('wave_stretch_1D_train')
        model_name = '1D_wave_stretch'
        n_epochs = 8
    elif model_type == 'H':
        model = Two_D()
        df = pd.read_pickle('median_2D_train')
        model_name = '2D_median'
        n_epochs = 10
    elif model_type == 'I':
        model = Two_D_single_img()
        model_name = '2D_one_img'
        n_epochs = 10
        image_transforms = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()]) #(256, 256)
        # Create a dataset from the images
        ECG_dataset = datasets.ImageFolder(root = dir+'\\data\\training\\training_one_img', transform = image_transforms)
    elif model_type == 'J':
        model = Two_D_12channel()
        df = pd.read_pickle('median_2D_train')
        model_name = '2D_12channel'
        n_epochs = 10
        df = transform_df_to_12channel(df)
    elif model_type == 'K':
        model = Two_D()
        df = pd.read_pickle('morphed_2D_train')
        model_name = '2D_morphed'
        n_epochs = 10
    elif model_type == 'L':
        model = VGG16()
        model_name = '2D_VGG16'
        n_epochs = 50
        image_transforms = transforms.Compose([transforms.Resize((256, 192)), transforms.ToTensor()]) #(256, 256)
        # Create a dataset from the images
        ECG_dataset = datasets.ImageFolder(root = dir+'\\data\\training\\training_one_img', transform = image_transforms)

    # Set this to True for plotting the training AUCs, set it to false for plotting the training metrics
    plot_AUCs = False
    # Set to true for 5 seperate AUCs
    per_Model_Plot = False

    # Different initialization for the one image version
    if model_type == 'I' or model_type =='K':
        x = ECG_dataset
        # Declares an array similair to the labels chronologically
        y = [0 for i in range(int(len(ECG_dataset)/2))]
        y.extend([1]*int(len(ECG_dataset)/2))
    else:
        df = df.sample(frac=1)
        # Prepare the 5 fold stratified split
        if model_name == '1D_median_betti':
            column_length = 24
        else:
            column_length = 12
        x = df.iloc[:, 0:column_length].copy()
        y = df.iloc[:, column_length].copy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    # Declare some empty variables for later
    acc, sens, spec, tprs, aucs, loss_arr, acc_arr = ([] for i in range(7))
    plot0, plot1, plot2, plot3, plot4 = range(5)
    base_fpr = np.linspace(0, 1, 100)

    # The loop over the 5 folds
    for cnt, (train_ids, valid_ids) in enumerate(kfold.split(x, y)): 
        torch.cuda.set_device(0)
        if model_type == 'I' or model_type == 'K': # One image model is different
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
            # Create the training and validation set
            train_ds = torch.utils.data.DataLoader(
                        ECG_dataset, 
                        batch_size=8, sampler=train_subsampler)
            valid_ds = torch.utils.data.DataLoader(
                        ECG_dataset,
                        batch_size=1, sampler=test_subsampler)
            # Train the model
            model_out, xx, yy = train_one_img_model(model, train_ds, valid_ds, n_epochs)
        else: # The standard models
            train_ds, valid_ds = create_tensors_folds(df, train_ids, valid_ids)
            # Train the model
            model_out, xx, yy = train_model(model, train_ds, valid_ds, n_epochs)
            
        # Save the training metrics for later plotting
        loss_arr.append(xx)
        acc_arr.append(yy)

        # Save the model in a different way for the gradcam
        if model_name == '2D_median' or model_name == '2D_one_img' or model_name == '2D_12channel':
            torch.save(model_out.state_dict(), dir+'\models\Grad_{0}_{1}'.format(model_name, cnt))
        # Save the model in the standard way
        torch.save(model_out,  dir+'\models\{0}_{1}'.format(model_name, cnt))
        model_out.eval()
        # Declare some empty variables
        FP, FN, TP, TN, correct, incorrect = (0 for i in range(6))
        y_true, y_score = [], []
        # One image model is different
        if model_type == 'I' or model_type == 'K': 
            for j, data_v in enumerate(valid_ds, 0):
                input_v, target_v = data_v
                input_v = input_v.to(device)
                target_v = target_v.to(device)
                target_v = target_v.float()
                # Reshape the target
                target_v = torch.reshape(target_v, (1, 1))
                # Run the model
                output_v = model_out(input_v)
                # Save the target and output for plotting the AUC
                y_true.append(target_v.item())
                y_score.append(m(output_v).item())
                if m(output_v) < 0.5:  #model output says negative  No dpp6
                    if target_v.item() == 0:  #D # it is negative  
                        correct += 1
                        TN += 1 
                    else:        #C            # it is actually positive
                        incorrect += 1
                        FN += 1
                else:                           # yes dpp6
                    if target_v.item() == 1: #A
                        correct += 1
                        TP += 1
                    else:   #B
                        incorrect += 1
                        FP += 1
        else:
            # Loop over the validation data
            for batch_idx, (data, target) in enumerate(valid_ds):
                data = data.to(device)
                target = target.to(device)
                data = data.unsqueeze(0)
                # Run the model
                output = model_out(data)
                # Save the target and output for plotting the AUC
                y_true.append(target.item())
                y_score.append(m(output).item())
                if m(output) < 0.5:  #model output says negative  No dpp6
                    if target.item() == 0:  #D # it is negative  
                        correct += 1
                        TN += 1 
                    else:        #C            # it is actually positive
                        incorrect += 1
                        FN += 1
                else:                           # yes dpp6
                    if target.item() == 1: #A
                        correct += 1
                        TP += 1
                    else:   #B
                        incorrect += 1
                        FP += 1
        # Prepare the AUC plot
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
        roc_auc = metrics.auc(fpr, tpr)
        AUC = round(roc_auc,2)

        if per_Model_Plot == True and plot_AUCs == False:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                    lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
        print('model_{0}'.format(cnt), '\tvalidation accuracy: ', round((correct/len(list(valid_ds)))*100, 2), '\tSpec: ', round(TN/(TN+FP)*100,2), '\tSens: ', round(TP/(TP+FN)*100,2), '\tAUC: ', round(roc_auc, 2))

        # Plot each of the 5 folds their AUC
        if plot_AUCs == True:
            plt.plot(fpr, tpr, cnt, alpha=0.3)
            vars()['plot'+str(cnt)], _ = plt.plot(fpr, tpr, cnt, alpha=0.3, label = f'ROC fold {cnt} (AUC: {AUC})')
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        # Save the training metrics for later printing
        aucs.append(AUC)
        acc.append(correct/len(list(valid_ds))*100)
        sens.append(TP/(TP+FN)*100)
        spec.append(TN/(TN+FP)*100)
        model.apply(weight_reset)
    print('\n')

    # Plot the AUCs over the 5 folds
    if plot_AUCs == True:
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        std_auc = round(np.std(aucs),2)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        roc_auc = metrics.auc(base_fpr, mean_tprs)
        AUC2 = round(roc_auc,2)
        plt.plot(base_fpr, mean_tprs, 'b')
        # Plot the mean AUC
        plot5 = plt.plot(base_fpr, mean_tprs, 'b', alpha = 1.0, label = f'mean ROC (AUC: {AUC2})')
        plot6 = plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label = f'std. dev. $\pm$ {std_auc}')
        handle_fill = PolyCollection([])
        handle_fill.update_from(plot6)
        plt.plot([0, 1], [0, 1],'r--')
        plot7 = plt.plot([0, 1], [0, 1],'r--', label = f'chance')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('2D CNN - Median - 5-fold training AUCs')
        plt.legend(loc="lower right", handles=[plot0, plot1, plot2, plot3, plot4, plot5[0], handle_fill, plot7[0]])
        plt.show()
    # Plot the training metrics over the epochs
    else: 
        base = np.linspace(0, n_epochs-1, num = n_epochs)
        acc_train = []
        acc_val = []
        for i in acc_arr:
            acc_train.append(i['train'])
            acc_val.append(i['val'])

        loss_train = []
        loss_val = []
        for i in loss_arr:
            loss_train.append(i['train'])
            loss_val.append(i['val'])
        
        acc_train = np.array(acc_train)
        mean_acc_train = acc_train.mean(axis=0)
        std_acc_train = acc_train.std(axis=0)
        acc_train_upper = np.maximum(mean_acc_train + std_acc_train, 1)
        acc_train_lower = mean_acc_train - std_acc_train

        acc_val = np.array(acc_val)
        mean_acc_val = acc_val.mean(axis=0)
        std_acc_val = acc_val.std(axis=0)
        acc_val_upper = np.maximum(mean_acc_val + std_acc_val, 1)
        acc_val_lower = mean_acc_val - std_acc_val

        loss_train = np.array(loss_train)
        mean_loss_train = loss_train.mean(axis=0)
        std_loss_train = loss_train.std(axis=0)
        loss_train_upper = np.minimum(mean_loss_train + std_loss_train, 1)
        loss_train_lower = mean_loss_train - std_loss_train

        loss_val = np.array(loss_val)
        mean_loss_val = loss_val.mean(axis=0)
        std_loss_val = loss_val.std(axis=0)
        loss_val_upper = np.minimum(mean_loss_val + std_loss_val, 1)
        loss_val_lower = mean_loss_val - std_loss_val

        # For saving the training metrics for plotting all model metrics together
        os.chdir(dir+'\\Visualization\\combined_plots')
        save_data = [n_epochs, mean_loss_train, base, loss_train_lower, loss_train_upper, mean_acc_train, mean_acc_val, acc_val_lower, acc_val_upper, acc_train_lower, acc_train_upper]
        save_data = np.array(save_data)
        np.save(model_name, save_data)

        fig, axs = plt.subplots(2) 
        listje = list(range(1,n_epochs+1))
        plt.setp(axs[0], xticks=list(range(0,n_epochs)), xticklabels= map(str, listje))
        plt.setp(axs[1], xticks=list(range(0,n_epochs)), xticklabels= map(str, listje))
        axs[0].plot(mean_loss_train, 'r', label = 'mean train loss')
        axs[0].plot(mean_loss_val, 'b', label = 'mean val loss')
        axs[0].fill_between(base, loss_val_lower, loss_val_upper, color='cornflowerblue', alpha=0.3, label = 'std. dev. val')
        axs[0].fill_between(base, loss_train_lower, loss_train_upper, color='salmon', alpha=0.3, label = 'std. dev train.')
        axs[0].set_title('Training Loss per Epoch')
        axs[0].set_xlabel('# of epochs')
        axs[0].set_ylabel('loss')
        axs[0].legend(framealpha = 0.5)

        axs[1].plot(mean_acc_train, 'r', label = 'mean train acc')
        axs[1].plot(mean_acc_val, 'b', label = 'mean val acc')
        axs[1].fill_between(base, acc_val_lower, acc_val_upper, color='cornflowerblue', alpha=0.3, label = 'std. dev. val')
        axs[1].fill_between(base, acc_train_lower, acc_train_upper, color='salmon', alpha=0.3, label = 'std. dev. train')
        axs[1].set_title('Training Accuracy per Epoch')
        axs[1].set_xlabel('# of epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(framealpha = 0.5)
        fig.tight_layout(pad = 1.0)
    acctot, spectot, senstot, auctot = (0 for i in range(4))
    # Sum the training metrics over the 5 folds
    for u in range(5):
        acctot += acc[u]
        spectot += spec[u]
        senstot += sens[u]
        auctot += aucs[u]
    # Print the mean training metrics
    print('Val accuracy: ', round(acctot/5, 2), '\tSens: ', round(senstot/5, 2), '\t\tSpec: ', round(spectot/5, 2),  '\t\tAUC: ', round(auctot/5, 2))  #63 for equal, 78 for unequal
    print('acc std: ', round(st.stdev(acc),2), '\t\tSens std: ', round(st.stdev(sens),2), '\tSpec std: ', round(st.stdev(spec),2), '\tAUC std: ', round(st.stdev(aucs),2))
    plt.show()




































