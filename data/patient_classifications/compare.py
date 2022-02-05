import os
import numpy as np

dir = os.getcwd()

arr2 = np.zeros(48)
for file in os.listdir(dir):
    if file != 'compare.py':
        arr = np.load(file)
        arr2 += arr
print(arr2)