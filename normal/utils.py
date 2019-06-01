import numpy as np
import torch
import os
import torch.utils.data as data
import random
# import torchvision.transforms.functional as F

class tcTrainData(data.Dataset):

    def __init__(self, dataLabel, infoPath, crop, transform=None):
        super(tcTrainData, self).__init__()
        # load data from npy file
        # classical way of loading data: read in every thing and then index
        # 238s for one epoch on mbp
        # self.data = np.load(dataPath)
        # new way: no self.data pre-read, load in file in the fly
        # instead storing data, we look up by label in train or val
        self.dataLabel = dataLabel
        # 236s for one epoch on mbp
        # load label from npy file with pandas dataframe
        self.target = np.load(infoPath)
        # transform as defined outside
        self.transform = transform
        self.centerCrop = crop
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
        # rotation for augmentation
        dat = np.rot90(dat, random.choice([0,1,2,3]), axes=(0,1))
        if self.centerCrop:
            dat = dat[68:(68+64), 68:(68+64), :]
        
        datMatrix = torch.from_numpy(dat[:, :, [0, 3]]).float()
        
        datMatrix = np.nan_to_num(datMatrix)
        datMatrix[datMatrix > 1000] = 0

        if self.transform:
            datMatrix = self.transform(datMatrix)
        
        labMatrix = self.target[index]
        # print(datMatrix.size(), labMatrix.size())
        
        return (datMatrix, labMatrix)

    def __len__(self):
        # classical way of returning length of dataset
        # return self.data.shape[0]
    
        # new way: count how many files in the current dir with same prefix
        return len([f for f in os.listdir('./data/') 
                    if os.path.isfile(os.path.join('./data/', f)) and f.split('_')[0] == self.dataLabel])

class tcValData(data.Dataset):

    def __init__(self, dataLabel, infoPath, crop, transform=None):
        super(tcValData, self).__init__()
        # load data from npy file
        # classical way of loading data: read in every thing and then index
        # 238s for one epoch on mbp
        # self.data = np.load(dataPath)
        # new way: no self.data pre-read, load in file in the fly
        # instead storing data, we look up by label in train or val
        self.dataLabel = dataLabel
        # 236s for one epoch on mbp
        # load label from npy file with pandas dataframe
        self.target = np.load(infoPath)
        # transform as defined outside
        self.transform = transform
        self.centerCrop = crop
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
        # no data augmentation in validation set
        # dat = np.rot90(dat, random.choice([0,1,2,3]), axes=(0,1))
        if self.centerCrop:
            dat = dat[68:(68+64), 68:(68+64), :]
        
        datMatrix = torch.from_numpy(dat[:, :, [0, 3]]).float()
        # print(datMatrix.size())
        
        datMatrix = np.nan_to_num(datMatrix)
        datMatrix[datMatrix > 1000] = 0
        
        if self.transform:
            datMatrix = self.transform(datMatrix)
        
        labMatrix = self.target[index]
        # print(datMatrix.size(), labMatrix.size())
        
        return (datMatrix, labMatrix)

    def __len__(self):
        # classical way of returning length of dataset
        # return self.data.shape[0]
    
        # new way: count how many files in the current dir with same prefix
        return len([f for f in os.listdir('./data/') 
                    if os.path.isfile(os.path.join('./data/', f)) and f.split('_')[0] == self.dataLabel])

class discRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

