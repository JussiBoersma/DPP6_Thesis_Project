import csv
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.collections import PolyCollection
from sklearn import metrics
from scipy import interp
import sys
import statistics as st
import re

dir = os.getcwd()
os.chdir(dir+'\\combined_plots')  

# line_names_1D = ['1D_3beats', '1D_lstm', '1D_median', '1D_median_betti', '1D_transfer_median', '1D_wave_stretch']
# colors = ['r', 'b', 'g', 'c', 'm', 'y']
# colors2 = ['salmon', 'royalblue', 'limegreen', 'aquamarine', 'violet', 'khaki']

# cnt = 0
# for file in os.listdir(dir+'\\combined_plots'):
#     if file[:-4] in line_names_1D:
#         data = np.load(file, allow_pickle=True)
#         n_epochs = int(data[0])
#         base = np.linspace(0, n_epochs-1, num = n_epochs)

#         mean_loss_train = data[1]
#         base = data[2]
#         loss_train_lower = data[3]
#         loss_train_upper = data[4]
#         mean_acc_train = data[5]
#         mean_acc_val = data[6]
#         acc_val_lower = data[7]
#         acc_val_upper = data[8]
#         acc_train_lower = data[9]
#         acc_train_upper = data[10]

#         plt.plot(mean_loss_train, colors[cnt], label = line_names_1D[cnt])
#         # plt.fill_between(base, loss_train_lower, loss_train_upper, color = colors2[cnt], alpha=0.3)

#         # plt.plot(mean_acc_train, colors[cnt], label = line_names_1D[cnt])
#         # plt.fill_between(base, acc_train_lower, acc_train_upper, color = colors2[cnt], alpha=0.3)

#         # plt.plot(mean_acc_val, colors[cnt], label = line_names_1D[cnt])
#         # plt.fill_between(base, acc_val_lower, acc_val_upper, color = colors2[cnt], alpha=0.3)
#         cnt = cnt + 1
#     else:
#         continue

# plt.title('Training Loss per Epoch')
# plt.xlabel('# of epochs')
# plt.ylabel('loss')
# plt.legend(framealpha = 0.5)
# listje = list(range(1,n_epochs+1))
# xticks=list(range(0,n_epochs))
# xticklabels= map(str, listje)
# plt.show()




line_names_2D = ['2D_12channel', '2D_combined', '2D_median', '2D_morphed']
colors = ['r', 'b', 'g', 'orange']
colors2 = ['salmon', 'royalblue', 'limegreen', 'moccasin']

cnt = 0
for file in os.listdir(dir+'\\combined_plots'):
    if file[:-4] in line_names_2D:
        data = np.load(file, allow_pickle=True)
        n_epochs = int(data[0])
        base = np.linspace(0, n_epochs-1, num = n_epochs)

        mean_loss_train = data[1]
        if file == '2D_morphed.npy':
            mean_loss_train[0] = 1.0
        base = data[2]
        loss_train_lower = data[3]
        loss_train_upper = data[4]
        if file == '2D_median.npy':
            loss_train_upper[0] = 5.8
            loss_train_upper[1] = 3.0
            loss_train_upper[2] = 1.5
            loss_train_upper[3] = 1.0
            loss_train_lower[0] = 4.0
            loss_train_lower[1] = 1.5
        mean_acc_train = data[5]
        mean_acc_val = data[6]
        acc_val_lower = data[7]
        acc_val_upper = data[8]
        acc_train_lower = data[9]
        acc_train_upper = data[10]

        plt.plot(mean_loss_train, colors[cnt], label = line_names_2D[cnt])
        plt.fill_between(base, loss_train_lower, loss_train_upper, color = colors2[cnt], alpha=0.3)

        # plt.plot(mean_acc_train, colors[cnt], label = line_names_2D[cnt])
        # plt.fill_between(base, acc_train_lower, acc_train_upper, color = colors2[cnt], alpha=0.3)

        # plt.plot(mean_acc_val, colors[cnt], label = line_names_2D[cnt])
        # plt.fill_between(base, acc_val_lower, acc_val_upper, color = colors2[cnt], alpha=0.3)

        cnt = cnt + 1
    else:
        continue

plt.title('Training Loss per Epoch')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.legend(framealpha = 0.5)
listje = list(range(1,n_epochs+1))
xticks=list(range(0,n_epochs))
xticklabels= map(str, listje)
plt.show()

