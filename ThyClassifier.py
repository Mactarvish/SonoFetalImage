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

from datasets.ThyDataset import ThyDataset
from models.resnet_th import resnet_th

use_gpu = torch.cuda.is_available()

class SubTrainDataset(ThyDataset):
    def __init__(self, subset, image_transform=None, pre_transform=None):
        super(SubTrainDataset, self).__init__(train=True, image_transform=image_transform, pre_transform=pre_transform)
        self.subset = subset

    def __len__(self):
        return 4150//2

    def __getitem__(self, item):
        if self.subset == 1:
            return super(SubTrainDataset, self).__getitem__(item=item)
        elif self.subset == 2:
            return super(SubTrainDataset, self).__getitem__(item=item + 4150/2)


transformer = transforms.Compose([
    mu.ResizeImage((255, 255)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.333, 0.0586, 0.023], std=[0.265, 0.138, 0.0224])
    ]
)
# [0.33381739584119885, 0.05862841105989082, 0.023407234558809612], [0.26509894104564447, 0.13794070296714034, 0.022363285181095156]
train_loader = DataLoader(ThyDataset(train=True, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=6, num_workers=6)
subset_loader1 = DataLoader(SubTrainDataset(subset=1, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=6, num_workers=6)
subset_loader2 = DataLoader(SubTrainDataset(subset=2, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=6, num_workers=6)
val_loader = DataLoader(ThyDataset(train=False, image_transform=transformer, pre_transform=None), shuffle=True, batch_size=6, num_workers=6)
dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(ThyDataset(train=True)), "val": len(ThyDataset(train=False))}

model_ft = resnet_th(pretrained=True)
#model_ft = models.resnet18(pretrained=False)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.MSELoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train step
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for lam in [0, 1]:
            # for data in train_loader:
            # for i in range(len(train_loader) // 2):
            for (x1, y1), (x2, y2) in zip(subset_loader1, subset_loader2):
                # x1 is torch.FloatTensor, y1 is torch.LongTensor, transform to float for mul
                y1 = y1.float()
                y2 = y2.float()
                # get the inputs
                inputs = lam * x1 + (1. - lam) * x2
                labels = lam * y1 + (1. - lam) * y2

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                # _, preds = torch.max(outputs, 1)
                # preds = preds.float()
                # loss = criterion(preds, labels)

                labels = mu.to_categorical(labels.data.cpu().numpy(), 6)
                labels = Variable(torch.from_numpy(labels)).float().cuda()
                # output:[torch.cuda.FloatTensor of size 6x6 (GPU 0)] labels:[torch.cuda.FloatTensor of size 6x6 (GPU 0)]
                # loss:[torch.cuda.FloatTensor of size 1 (GPU 0)]
                nn.CrossEntropyLoss
                loss = nn.MSELoss()(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.data[0]
                # running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / (dataset_sizes['train'] * 1.5)
        # epoch_acc = running_corrects / (dataset_sizes['train'] * 1.5)

        print('train Loss: {:.4f}'.format(
            epoch_loss))

        # validation step
        model.train(False)
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for data in val_loader:
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
            loss = nn.CrossEntropyLoss()(outputs, labels)
            # loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects / dataset_sizes['val']

        print('val Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

        # # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        #     if phase == 'train':
        #         scheduler.step()
        #         model.train(True)  # Set model to training mode
        #     else:
        #         model.train(False)  # Set model to evaluate mode
        #
        #     running_loss = 0.0
        #     running_corrects = 0
        #     # Iterate over data.
        #     for data in dataloaders[phase]:
        #         # get the inputs
        #         inputs, labels = data
        #
        #         # wrap them in Variable
        #         if use_gpu:
        #             inputs = Variable(inputs.cuda())
        #             labels = Variable(labels.cuda())
        #         else:
        #             inputs, labels = Variable(inputs), Variable(labels)
        #
        #         # zero the parameter gradients
        #         optimizer.zero_grad()
        #
        #         # forward
        #         outputs = model(inputs)
        #         _, preds = torch.max(outputs.data, 1)
        #         loss = criterion(outputs, labels)
        #
        #         # backward + optimize only if in training phase
        #         if phase == 'train':
        #             loss.backward()
        #             optimizer.step()
        #
        #         # statistics
        #         running_loss += loss.data[0]
        #         running_corrects += torch.sum(preds == labels.data)
        #
        #     epoch_loss = running_loss / dataset_sizes[phase]
        #     epoch_acc = running_corrects / dataset_sizes[phase]
        #
        #     print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #         phase, epoch_loss, epoch_acc))
        #
        #     # deep copy the model
        #     if phase == 'val' and epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)

# torch.save(model_ft, 'Thynet.pkl')  # 保存整个神经网络的结构和模型参数
# torch.save(model_ft.state_dict(), 'Thynet.pkl')  # 只保存神经网络的模型参数

# model_ft.load_state_dict(torch.load("Thynet.pkl"))
# print(torch.max(model_ft(image2modelinput("ThyImage/hyperthyreosis1_251701.jpg", (255, 255))), 1)[1])