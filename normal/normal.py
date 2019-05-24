from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import time

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
dataloaders = {x: DataLoader(datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    training_acc = []
    val_acc = []

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
            else:
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Elapsed: {:.4f}'.format(time.time() - since))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc, val_acc

# choose model based on user input
modelSize = input("Enter the number of layers you want in the Resnet model: 18, 34, 50, 101, 152?\n")
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.33)

# learn and save plot
_, train_acc, val_acc = train_model(resnetTC, criterion, optimizer, scheduler, num_epochs=50)

# monitor CPU usage
# top -F -R -o cpu

plt.plot(train_acc, label = 'train')
plt.plot(val_acc, label = 'val')
plt.xlabel(r'Epochs')
plt.ylabel(r'Training Acc')
plt.legend()
plt.savefig('train acc')
plt.close()