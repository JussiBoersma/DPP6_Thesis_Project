import csv
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
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import operator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def load_data():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\DF')
    df = pd.read_pickle('pickled_df_bettis')

    negs = []
    poss = []
    label = []
    for i in range(df.shape[0]):
        if df.iloc[i,12] == 0:
            arr_neg = []
            for j in range(12):
                for k in range(1000):
                    arr_neg.append(df.iloc[i, j][k])
            negs.append(arr_neg)
            label.append(0)
        else:
            arr_pos = []
            for j in range(12):
                for k in range(1000):
                    arr_pos.append(df.iloc[i, j][k])
            poss.append(arr_pos)
            label.append(1)

    print(len(label), len(poss), len(negs))

    data = {'array': negs+poss, 'label': label}
    datafr = pd.DataFrame(data)
    datafr.to_pickle('chisq_bettis')

def prep_data():
    os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\DF')
    df = pd.read_pickle('chisq_bettis')

    sum_neg = np.zeros(12000)
    sum_pos = np.zeros(12000)
    for y in range(df.shape[0]):
        if df.iloc[y,1] == 0:
            for i in range(12000):
                sum_neg[i] = sum_neg[i] + df.iloc[y,0][i]
        else:
            for i in range(12000):
                sum_pos[i] = sum_pos[i] + df.iloc[y,0][i]

    avg_neg = [i/234 for i in sum_neg]
    avg_pos = [i/156 for i in sum_pos]

    negnp = np.array(avg_neg)
    posnp = np.array(avg_pos)

    np.save('avg_neg_bettis', negnp) 
    np.save('avg_pos_bettis', posnp) 

def chi2_distance(A, B):
  
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) 
                      for (a, b) in zip(A, B)])
  
    return chi

def svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = SVC()
    clf.fit(list(X_train),y_train)

    y_pred_train = clf.predict(list(X_train))

    y_pred_test = clf.predict(list(X_test))

    print('Accuracy traindata')
    print(accuracy_score(y_train, y_pred_train))
    print('')
    print('Accuracy testdata')
    print(accuracy_score(y_test, y_pred_test))

    print("Precision:",metrics.precision_score(y_test, y_pred_test))
    print("Recall:",metrics.recall_score(y_test, y_pred_test))

def knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\Machine_learning')
df = pd.read_pickle('df')

arr = []
for i in range(len(df)):
    arr2 = []
    for j in range(14):
        arr2.append(df.iloc[i,0][j])
    arr.append(arr2)

X = arr
y = df['label'].values
knn(X,y)
# svm(X,y)







# load_data()   156 234
# prep_data()

# os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Rhythm\DF')
# x = np.load('avg_neg_bettis.npy')
# y = np.load('avg_pos_bettis.npy')
# df = pd.read_pickle('chisq_bettis')
# X = df['array']
# y = df['label']

# os.chdir(r'C:\Users\jussi\Documents\Master Thesis\Data\12_leads_raw\Median\DF')
# df = pd.read_pickle('pickled_df')
# df2 = pd.read_csv('median_data_KNN.csv', header = None)
# print(df2)

# arr2 = []
# for i in range(df2.shape[0]):
#     arr = []
#     for j in range(1,df2.shape[1]):
#         arr.append(df2.iloc[i,j])
#     arr2.append(arr)

# data = {'array': arr2}
# frampje = pd.DataFrame(data)
# x = frampje['array']
# y = df['label']

# svm(x,y)

# correct = 0
# false = 0
# for i in range(df.shape[0]):
#     dist1 = chi2_distance(x, df.iloc[i, 0])
#     dist2 = chi2_distance(y, df.iloc[i, 0])

#     if dist1 < dist2:
#         # classification = 'negative'
#         if df.iloc[i,1] == 0:
#             correct += 1
#         else:
#             false += 1
#     else:
#         # classification = 'positive'
#         if df.iloc[i,1] == 1:
#             correct += 1
#         else:
#             false += 1
# print(correct/390)


















    