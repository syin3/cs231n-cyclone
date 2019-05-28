from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision
from torchvision import transforms, datasets, models
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from dataObj import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# preparation for learning, define all helpers and criterion
# trainData = np.load('train.npy')
transf = {
    'mean': (267.81773236, 0.48700156),
    'std': (27.09388376,  1.47568378)
}
data_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=transf['mean'], std=transf['std']) 
    # transforms.CenterCrop(64)
])
datasets = {x: resnetTCData('{}.npy'.format(x), '{}Info.npy'.format(x), data_transform) 
            for x in ['train', 'val']}
dataloaders = {x: DataLoader(datasets[x], batch_size=256, shuffle=True, num_workers=8)
               for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# training function
def train_model(model, criterion, optimizer, scheduler, num_epochs, folder):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    training_acc = []
    training_loss = []
    val_acc = []
    val_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
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
                training_acc.append(epoch_acc)
                training_loss.append(epoch_loss)
            else:
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Elapsed: {:.4f}'.format(time.time() - since))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

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
                }, './results/{}/{}'.format(folder, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc, val_acc, training_loss, val_loss

# choose model and training epochs based on user input
modelSize = input("Enter the number of layers you want in the Resnet model:\n Choose one from: 18, 34, 50, 101, 152\n")
epochTrain = input("Enter the number of epochs to train:\n I usually do 50 or 100\n")

# create directory for the specified model and facilitates later output
dirName = "./results/resnet_{}_{}".format(modelSize, epochTrain)
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")


print("Note, your trained model will be saved at this location:\n ./results/resnet_{}_{}/model".format(modelSize, epochTrain))

# training with specified hyper-parameter
labels = ['no', 'td', 'ts', 'one', 'two', 'three', 'four', 'five']
models = {
    '18': resnet18,
    '34': resnet34,
    '50': resnet50,
    '101': resnet101,
    '152': resnet152
}
resnetTC = models[modelSize](num_classes = 8, pretrained=False)
resnetTC = resnetTC.to(device)

optimizer = optim.Adam(resnetTC.parameters(), lr=0.001, weight_decay=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
#                                                  patience = 5, threshold = 0.01, 
#                                                  factor = 1/3, threshold_mode  = 'rel')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)

# learn
learned_model, train_acc, val_acc, training_loss, val_loss = train_model(resnetTC, criterion, optimizer, scheduler, \
    num_epochs=int(epochTrain), folder="resnet_{}_{}".format(modelSize, epochTrain))

# save model
torch.save(learned_model.state_dict(), "./results/resnet_{}_{}/model".format(modelSize, epochTrain))

# save training and validation accuracies and losses
with open("/results/resnet_{}_{}/train_acc.txt".format(modelSize, epochTrain), "w+") as file:
    file.write(str(train_acc))
with open("/results/resnet_{}_{}/train_loss.txt".format(modelSize, epochTrain), "w+") as file:
    file.write(str(train_loss))
with open("/results/resnet_{}_{}/val_acc.txt".format(modelSize, epochTrain), "w+") as file:
    file.write(str(val_acc))
with open("/results/resnet_{}_{}/val_loss.txt".format(modelSize, epochTrain), "w+") as file:
    file.write(str(val_loss))

# to load saved models, please follow this page:
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3

# to load accuracies and losses from files, please refer to this webpage
# https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
# with open("test.txt", "r") as file:
#     data2 = eval(file.readline())


# monitor CPU usage
# top -F -R -o cpu

plt.plot(train_acc, label = 'train')
plt.plot(val_acc, label = 'val')
plt.xlabel(r"Epochs")
plt.ylabel(r"Training Acc")
plt.legend()
plt.savefig("/results/resnet_{}_{}/acc".format(modelSize, epochTrain))
plt.close()

