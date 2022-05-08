import cv2
import numpy as np
import copy
import os
from scipy.linalg import svd, norm

def normalize(datas):
    
    new_datas = []

    # subtract mean
    means = []
    for i in range(len(datas)):
        mean = np.mean(datas[i])
        means.append(mean)
        #datas[i] - mean
    print(means)
    # get standard deviation
    stds = []
    for i in range(len(datas)):
        sub_sum = 0
        for j in range(len(datas[i])):
            sub_sum += (datas[i][j] - means[i])**2
        stds.append((sub_sum/(len(datas[0])-1))**0.5)
    print(stds)
    # normalize
    for i in range(len(datas)):
        row = []
        for j in range(len(datas[i])):
            #print((datas[i][j] - means[i])/stds[i])
            row.append((datas[i][j] - means[i])/stds[i])
        new_datas.append(row)
    
    return np.array(new_datas)


path_dir = './faces_training'
file_list = os.listdir(path_dir)
#print(file_list)

datas = []
for file_name in file_list:
    file_path = path_dir + '/' + file_name
    original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    arr = original.reshape(-1)
    datas.append(arr)
datas = np.array(datas)
print(datas.shape)

# step 1) normalize data
datas = normalize(datas)

# step 2) compute covariance matrix
datas = datas.transpose()
print(datas)

U,S,V_t = svd(datas)
print(S)



'''
datas = datas.transpose()
print(datas.shape)
'''
