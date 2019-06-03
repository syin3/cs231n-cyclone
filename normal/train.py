from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import time
import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from utils import *
from model import *

random.seed(0)

# training function
def train_model(model, device, criterion, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs, folder):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    trainLoss = 0

    for epoch in range(num_epochs):
        epochStart = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if folder.split("_")[-2] == 'step':
                    scheduler.step()
                else:
                    scheduler.step(trainLoss)
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #print(outputs, loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                trainLoss = epoch_loss
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
                if epoch == 0:
                    print(type(epoch_acc))
                    print(type(epoch_loss))
                # save training and validation accuracies and losses
                with open('{}/train_acc.txt'.format(folder), "a+") as file:
                    file.write(str(epoch_acc.item()) + ',')
                with open('{}/train_loss.txt'.format(folder), "a+") as file:
                    file.write(str(epoch_loss) + ',')
            else:
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)
                with open('{}/val_acc.txt'.format(folder), "a+") as file:
                    file.write(str(epoch_acc.item()) + ',')
                with open('{}/val_loss.txt'.format(folder), "a+") as file:
                    file.write(str(epoch_loss) + ',')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        epochEnd = time.time()
        print('Epoch elapsed: {:.4f}\n'.format(epochEnd - epochStart))
        print()

        # save checkpoint, see these two webpages
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, '{}/{}'.format(folder, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss

def main(args):
    transf = {
        'mean': (267.81773236, 0.48700156),
        'std': (27.09388376,  1.47568378)
    }
    data_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=transf['mean'], std=transf['std'])
    ])
    datasets = {
        'train': tcTrainData('train', 'trainInfo.npy', args.centerCrop, data_transform), 
        'val': tcValData('val', 'valInfo.npy', args.centerCrop, data_transform)
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batchSize, num_workers=16, shuffle=True, pin_memory=True) for x in ['train', 'val']}
                
    # dataloaders = {x: DataLoader(datasets[x], batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    #                for x in ['train', 'val']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # create directory for the specified model and facilitates later output
    dirName = "./results/resnet_{}_{}_{}_{}_{}_{}".format(args.modelSize, args.epochTrain, args.regStrength, args.lr, args.scheduler, args.centerCrop)
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
        for f in os.listdir(dirName):
            os.unlink(os.path.join(dirName, f))

    # training with specified hyper-parameter
    labels = ['no', 'td', 'ts', 'one', 'two', 'three', 'four', 'five']
    models = {
        '18': resnet18,
        '34': resnet34,
        '50': resnet50,
        '101': resnet101,
        '152': resnet152
    }
    # model = nn.Sequential(models[modelSize](pretrained=True), nn.Linear(1000, 8)) 
    model = models[str(args.modelSize)](num_classes=8, pretrained=False)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regStrength)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, threshold=0.01, factor=1/3, threshold_mode='rel')
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else: 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=0.01, factor=1/3, threshold_mode='rel')

    # learn
    learned_model, train_acc, val_acc, train_loss, val_loss = train_model(model, device, criterion, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs=args.epochTrain, folder=dirName)

    # save model
    torch.save(learned_model.state_dict(), "{}/model".format(dirName))

    plt.figure(1)
    plt.plot(train_acc, label = 'train')
    plt.plot(val_acc, label = 'val')
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Accuracy")
    plt.legend()
    plt.savefig("{}/acc".format(dirName))
    plt.close()

    plt.figure(2)
    plt.plot(train_loss, label = 'train')
    plt.plot(val_loss, label = 'val')
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Loss")
    plt.legend()
    plt.savefig("{}/loss".format(dirName))
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelSize', default=18, choices=[18,34,50,101,152], type=int, help='Size of Resnet model')
    parser.add_argument('--epochTrain', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batchSize', default=64, choices=[32,64,128,256,512], type=int, help='Batch size')
    parser.add_argument('--regStrength', default=1e-3, type=float, help='Regularization strength, usually 1e-3')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate, usually 1e-3')
    parser.add_argument('--scheduler', default='step', type=str, help='Scheduler in learning rate')
    parser.add_argument('--centerCrop', default=False, type=bool, help='Boolean for center crop or not')
    args = parser.parse_args()

    main(args)


