import numpy as np
import pandas as pd
import h5py
import random
import os

# Read info from first file
data_path = "../TCIR-ATLN_EPAC_WPAC.h5"
info1 = pd.read_hdf(data_path, key="info", mode='r')
assert info1.shape[0] == 47381
trainIdx1 = random.sample(range(47381), int(0.9*47381))
valIdx1 = []
for ele in range(47381):
    if ele not in trainIdx1:
        valIdx1.append(ele)
trainInfo1 = info1.iloc[trainIdx1]
valInfo1 = info1.iloc[valIdx1]

# Read info from second file
data_path = "../TCIR-CPAC_IO_SH.h5"
info2 = pd.read_hdf(data_path, key="info", mode='r')
assert info2.shape[0] == 23118
trainIdx2 = random.sample(range(23118), int(0.9*23118))
valIdx2 = []
for ele in range(23118):
    if ele not in trainIdx2:
        valIdx2.append(ele)
trainInfo2 = info2.iloc[trainIdx2]
valInfo2 = info2.iloc[valIdx2]

# write out info
trainInfo = trainInfo1.append(trainInfo2)
valInfo = valInfo1.append(valInfo2)

trainInfo['category'] = pd.cut(trainInfo['Vmax'], bins=[0, 20, 33, 63, 82, 95, 112, 136, 1000], 
       include_lowest=True, labels=[0, 1, 2, 3, 4, 5, 6, 7])
valInfo['category'] = pd.cut(valInfo['Vmax'], bins=[0, 20, 33, 63, 82, 95, 112, 136, 1000], 
       include_lowest=True, labels=[0, 1, 2, 3, 4, 5, 6, 7])

assert trainInfo.shape[0] == len(trainIdx1) + len(trainIdx2)
assert valInfo.shape[0] == len(valIdx1) + len(valIdx2)
assert trainInfo1.shape[0] + trainInfo2.shape[0] == trainInfo.shape[0]
assert valInfo1.shape[0] + valInfo2.shape[0] == valInfo.shape[0]

np.save('trainInfo', trainInfo['category'])
np.save('valInfo', valInfo['category'])

# total training number
print('Total training num = {}'.format(int(0.9*47381) + int(0.9*23118)))
print('Total val num = {}'.format(47381 + 23118 - int(0.9*47381) - int(0.9*23118)))

# initialize
total = np.zeros(4)
sqd = np.zeros(4)
invalidCount = 0

# write and count first file
with h5py.File("../TCIR-ATLN_EPAC_WPAC.h5", 'r') as hf:
    data_matrix = hf['matrix'][:]
    
    # index of train start from zero from ATLN_EPAC_WPAC
    for idx in range(len(trainIdx1)):
        assert os.path.isfile('./data/train_{}'.format(idx)) == False
        np.save('./data/train_{}'.format(idx), data_matrix[trainIdx1[idx], :, :, :])
        
        # local = data_matrix[trainIdx1[idx], :, :, :]
        # invalidCount += np.sum(np.isnan(local)) / (63448 * 201 * 201)
        # local = np.nan_to_num(local)
        # invalidCount +=  np.sum(local > 1000) / (63448 * 201 * 201)
        # local[local > 1000] = 0
        # total += np.sum(local, axis = (0,1)) / (63448 * 201 * 201)
        # sqd +=  np.sum(local ** 2, axis = (0,1)) / (63448 * 201 * 201)
        # assert local.shape == (201, 201, 4)

        if idx % 1000 == 0:
            print("Reached: {}".format(idx))
            print("invalid ratio: {}".format(invalidCount))
    
    # val of train start from zero from ATLN_EPAC_WPAC
    for idx in range(len(valIdx1)):
        assert os.path.isfile('./data/val_{}'.format(idx)) == False
        np.save('./data/val_{}'.format(idx), data_matrix[valIdx1[idx], :, :, :])

# write and count second file
with h5py.File("../TCIR-CPAC_IO_SH.h5", 'r') as hf:
    data_matrix = hf['matrix'][:]
    
    # index of train start from zero from ATLN_EPAC_WPAC
    # so here we need to add something
    for idx in range(len(trainIdx2)):
        assert os.path.isfile('./data/train_{}'.format(idx + len(trainIdx1))) == False
        np.save('./data/train_{}'.format(idx + len(trainIdx1)), data_matrix[trainIdx2[idx], :, :, :])
        
        # local = data_matrix[trainIdx2[idx], :, :, :]
        # invalidCount += np.sum(np.isnan(local)) / (63448 * 201 * 201)
        # local = np.nan_to_num(local)
        # invalidCount +=  np.sum(local > 1000) / (63448 * 201 * 201)
        # local[local > 1000] = 0
        # total += np.sum(local, axis = (0,1)) / (63448 * 201 * 201)
        # sqd +=  np.sum(local ** 2, axis = (0,1)) / (63448 * 201 * 201)
        # assert local.shape == (201, 201, 4)

        if (idx + len(trainIdx1)) % 1000 == 0:
            print("Reached: {}".format(idx + len(trainIdx1)))
            print("invalid ratio: {}".format(invalidCount))
    
    # val of train start from zero from ATLN_EPAC_WPAC
    for idx in range(len(valIdx2)):
        assert os.path.isfile('./data/val_{}'.format(idx + len(trainIdx1))) == False
        np.save('./data/val_{}'.format(idx + len(valIdx1)), data_matrix[valIdx2[idx], :, :, :])

# calculate statistics
# print(total) # [265.64917688 234.43013362   0.29799191   0.48401537]
# print(sqd) # [7.18736948e+04 5.54778039e+04 4.56178067e-01 2.40000602e+00]

# validCount = 1 - invalidCount
# print(validCount) # 0.5875800449754618

# make assertion about total number files written
assert len([f for f in os.listdir('./data/') if os.path.isfile(os.path.join('./data/', f)) and f.split('_')[0] == 'train']) == 63448
assert len([f for f in os.listdir('./data/') if os.path.isfile(os.path.join('./data/', f)) and f.split('_')[0] == 'val']) == 7051

# overfit max dataset: 
# EX: (207.02715, 0.53710043)
# std: (58.167034,  1.552934)

# EX = total / validCount
# EX2 = sqd / validCount

# var = EX2 - EX ** 2
# std = var ** 0.5
# print(EX, std)