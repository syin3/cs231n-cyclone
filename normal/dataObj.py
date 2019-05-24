import numpy as np
import torch
import os
import torch.utils.data as data

class resnetTCData(data.Dataset):

    def __init__(self, dataPath, infoPath, transform=None):
        super(resnetTCData, self).__init__()
        # load data from npy file
        # classical way of loading data: read in every thing and then index
        # 238s for one epoch on mbp
        # self.data = np.load(dataPath)
        # new way: no self.data pre-read, load in file in the fly
        # instead storing data, we look up by label in train or val
        self.dataLabel = dataPath.split('.')[0]
        # 236s for one epoch on mbp
        # load label from npy file with pandas dataframe
        self.target = np.load(infoPath)
        # transform as defined outside
        self.transform = transform
        self.targetDic = {
            'no':0, 'td':1, 'ts':2, 'one':3, 'two':4, 'three':5, 'four':6, 'five':7
        }

    def __getitem__(self, index):
        # middle crop and permute axis
        # we found that in the later cell, there is a "center crop" transform
        # if the transforms.centercrop() works, then no need for 68:(68+64)
        
        # classical way of loading data: read in every thing and then index
        # 238s for one epoch on mbp
        # datMatrix = torch.from_numpy(np.moveaxis(self.data[index, :, :, [0, 3]], 0, -1)).float()
        
        # new way: no self.data pre-read, load in file in the fly according to the label
        # 236s to train one epoch on mbp
        dat = np.load('./data/' + self.dataLabel + '_{}.npy'.format(index))
        datMatrix = torch.from_numpy(dat[:, :, [0, 3]]).float()
        
        datMatrix = np.nan_to_num(datMatrix)
        datMatrix[datMatrix > 1000] = 0
#         print(index)
        # print(datMatrix.shape)
        if self.transform:
            datMatrix = self.transform(datMatrix)
        
        labMatrix = self.target[index]
        
        return (datMatrix, labMatrix)

    def __len__(self):
        # classical way of returning length of dataset
        # return self.data.shape[0]
    
        # new way: count how many files in the current dir with same prefix
        return len([f for f in os.listdir('./data/') 
                    if os.path.isfile(os.path.join('./data/', f)) and f.split('_')[0] == self.dataLabel])