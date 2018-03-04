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
# model_ft = models.resnet18(pretrained=False)

# model_ft = models.resnet152(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 6), nn.Softmax())

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.MSELoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


NUM_CLASSES = 6
IMAGE_SIDE_LENGTH = 255
# Training
def shuffle_minibatch(inputs, targets, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.

    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """

    # mess up the batch order
    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, NUM_CLASSES)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, NUM_CLASSES)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = np.tile(a, [1, NUM_CLASSES])
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle


# def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
#     since = time.time()
#
#     best_model_wts = model.state_dict()
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # train step
#         scheduler.step()
#         model.train(True)  # Set model to training mode
#         running_loss = 0.0
#         running_corrects = 0
#         # Iterate over data.
#
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             # execute mixup
#             inputs, labels = shuffle_minibatch(
#                 inputs, targets, mixup=False)
#             # wrap them in Variable
#             inputs = Variable(inputs.cuda())
#             labels = Variable(labels.cuda())
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward
#             outputs = model(inputs)
#             # _, preds = torch.max(outputs, 1)
#             # print(preds)
#             # preds = preds.float()
#             # loss = criterion(preds, labels)
#
#             # output:[torch.cuda.FloatTensor of size 6x6 (GPU 0)] labels:[torch.cuda.FloatTensor of size 6x6 (GPU 0)]
#             # loss:[torch.cuda.FloatTensor of size 1 (GPU 0)]
#             m = nn.LogSoftmax()
#
#             loss = -m(outputs) * labels
#             loss = torch.sum(loss)
#             # backward + optimize only if in training phase
#             loss.backward()
#             optimizer.step()
#
#             # statistics
#             running_loss += loss.data[0]
#             # running_corrects += torch.sum(preds == labels.data)
#
#         epoch_loss = running_loss / (dataset_sizes['train'] * 1.5)
#         # epoch_acc = running_corrects / (dataset_sizes['train'] * 1.5)
#
#         print('train Loss: {:.4f}'.format(
#             epoch_loss))
#
#         # validation step
#         y_pred = []
#         y_true = []
#         losses = []
#
#         model.train(False)
#         running_loss = 0.0
#         running_corrects = 0
#         # Iterate over data.
#         for data in val_loader:
#             # get the inputs
#             inputs, labels = data
#
#             # wrap them in Variable
#             if use_gpu:
#                 inputs = Variable(inputs.cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 inputs, labels = Variable(inputs), Variable(labels)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward
#             outputs = model(inputs)
#             _, preds = torch.max(outputs.data, 1)
#             loss = nn.CrossEntropyLoss()(outputs, labels)
#
#             y_pred_np = preds.cpu().numpy()
#             y_true_np = labels.data.cpu().numpy()
#             losses.append(loss.data[0])
#
#             for y in y_pred_np:
#                 y_pred.append(y)
#             for y in y_true_np:
#                 y_true.append(y)
#
#             # statistics
#             running_loss += loss.data[0]
#             running_corrects += torch.sum(preds == labels.data)
#
#         mu.save_matrics(y_true, y_pred, losses, net_name="resnet_th_mymixup")
#         epoch_loss = running_loss / dataset_sizes['val']
#         epoch_acc = running_corrects / dataset_sizes['val']
#
#         print('val Loss: {:.4f} Acc: {:.4f}'.format(
#             epoch_loss, epoch_acc))
#
#         # deep copy the model
#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = model.state_dict()
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

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
        # for data in train_loader:
        # for i in range(len(train_loader) // 2):
        for lam in [0, 1]:
            for (x1, y1), (x2, y2) in zip(subset_loader1, subset_loader2):
                # x1 is torch.FloatTensor, y1 is torch.LongTensor, transform to float for mul
                y1 = y1.float()
                y2 = y2.float()
                # get the inputs
                # lam = np.random.beta(1, 1)
                # lam = 1
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
                # print(preds)
                # preds = preds.float()
                # loss = criterion(preds, labels)

                labels = mu.to_categorical(labels.data.cpu().numpy(), 6)
                labels = Variable(torch.from_numpy(labels)).float().cuda()
                # output:[torch.cuda.FloatTensor of size 6x6 (GPU 0)] labels:[torch.cuda.FloatTensor of size 6x6 (GPU 0)]
                # loss:[torch.cuda.FloatTensor of size 1 (GPU 0)]
                m = nn.LogSoftmax()

                loss = -m(outputs) * labels
                loss = torch.sum(loss)
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
        y_pred = []
        y_true = []
        losses = []

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

            y_pred_np = preds.cpu().numpy()
            y_true_np = labels.data.cpu().numpy()
            losses.append(loss.data[0])

            for y in y_pred_np:
                y_pred.append(y)
            for y in y_true_np:
                y_true.append(y)

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

        mu.save_matrics(y_true, y_pred, losses, net_name="resnet_th_mymixup")
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects / dataset_sizes['val']

        print('val Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
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
                       num_epochs=150)

# torch.save(model_ft, 'Thynet.pkl')  # 保存整个神经网络的结构和模型参数
# torch.save(model_ft.state_dict(), 'Thynet.pkl')  # 只保存神经网络的模型参数

# model_ft.load_state_dict(torch.load("Thynet.pkl"))
# print(torch.max(model_ft(image2modelinput("ThyImage/hyperthyreosis1_251701.jpg", (255, 255))), 1)[1])


'''
confusion_matrix : 
 [[50  0  0  0  0  0]
 [ 0 50  0  0  0  0]
 [ 0  0 50  0  0  0]
 [ 0  0  0 50  0  0]
 [ 0  0  0  0 49  1]
 [ 0  0  1  0 23 26]]
'''