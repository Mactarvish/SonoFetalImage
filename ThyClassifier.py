import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import MaUtilities as mu
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
from torchvision.models import vgg
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import time
import cv2
from torchvision import datasets, models, transforms
import sqlite3

from models.resnet_th import resnet_th
from datasets.ThyDataset import ThyDataset

use_gpu = torch.cuda.is_available()
image_location = "/home/hdl2/Desktop/SonoFetalImage/ThyImage/"
label_map = {"hashimoto_thyroiditis1": 0, "hyperthyreosis1": 1, "normal1": 2, "postoperative1": 3, "subacute_thyroiditis1": 4, "subhyperthyreosis1": 5}

connection = sqlite3.connect("ThyDataset_Shuffled")
cu = connection.cursor()

torch.manual_seed(123)
torch.cuda.manual_seed(222)


transformer = transforms.Compose([
    mu.ResizeImage((255, 255)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.333, 0.0586, 0.023], std=[0.265, 0.138, 0.0224])
    ]
)
# [0.33381739584119885, 0.05862841105989082, 0.023407234558809612], [0.26509894104564447, 0.13794070296714034, 0.022363285181095156]
train_loader = DataLoader(ThyDataset(train=True, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=6, num_workers=6)
val_loader   = DataLoader(ThyDataset(train=False, image_transform=transformer, pre_transform=None), shuffle=True, batch_size=6, num_workers=6)
dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(ThyDataset(train=True)), "val": len(ThyDataset(train=False))}


model_ft = resnet_th(pretrained=True)
#model_ft = models.resnet18(pretrained=False)

# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 6)

if use_gpu:
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

# torch.save(model_ft, 'Thynet.pkl')  # 保存整个神经网络的结构和模型参数
# torch.save(model_ft.state_dict(), 'Thynet.pkl')  # 只保存神经网络的模型参数

# model_ft.load_state_dict(torch.load("Thynet.pkl"))
# print(torch.max(model_ft(image2modelinput("ThyImage/hyperthyreosis1_251701.jpg", (255, 255))), 1)[1])