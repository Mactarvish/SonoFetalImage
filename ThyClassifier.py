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
from models.my_densenet import mydensenet121
from datasets.ThyDataset import ThyDataset

use_gpu = torch.cuda.is_available()
torch.cuda.set_device(0)
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
train_loader = DataLoader(ThyDataset(train=True, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=5, num_workers=6)
val_loader   = DataLoader(ThyDataset(train=False, image_transform=transformer, pre_transform=None), shuffle=True, batch_size=6, num_workers=6)

# model = resnet_th(pretrained=True)
# model = models.resnet18(pretrained=True)
# model = models.vgg16(pretrained=True)
model = mydensenet121(pretrained=True)


if use_gpu:
    # model = nn.DataParallel(model)
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

def Tensor2Variable_CEL(input, label):
    """
    :param input: Torch.Tensor
    :param label: Torch.Tensor
    :return: Torch.Tensor.cuda() obey CEL type
    """
    input = Variable(input).cuda().float()
    label = Variable(label).cuda().long()
    return input, label

def t2t_fft(tensor_nchw):
    """
    :param tensor_nchw: Torch.Tensor
    :return: Torch.Tensor
    """
    np_image_nchw = tensor_nchw.numpy()
    f_shift = np.fft.fftshift(np.fft.fft2(np_image_nchw, axes=(-1, -2)), axes=(-1, -2))
    f_amplify = 20 * np.log10(np.abs(f_shift))
    f_amplify = f_amplify - np.min(f_amplify)
    f_amplify = f_amplify / 255
    # mu.display(f_amplify[0, 0, ...], f_amplify[0, 1, ...], f_amplify[0, 2, ...],
    #            f_amplify[1, 0, ...], f_amplify[1, 1, ...], f_amplify[1, 2, ...],
    #            ion=False)
    f_amplify = torch.from_numpy(f_amplify).float()
    return f_amplify

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    model.train(True)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since = time.time()

        scheduler.step()
        for (input, label) in train_loader:
            optimizer.zero_grad()
            # prepare datas
            input_fft = t2t_fft(input)
            # print(torch.cat([input_fft, input], 1).shape)
            input, label = Tensor2Variable_CEL(input_fft, label)
            # run the model
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        test_model(model, criterion)

        time_diff = time.time() - since
        print('epoch complete in {:0.2f} seconds'.format(time_diff))
        print()

# only 1 epoch
def test_model(model, criterion):
    model.train(False)
    log_y_predictions = []
    log_y_trues = []
    log_losses = []
    epoch_loss = 0
    for (input, label) in val_loader:
        input, label = Tensor2Variable_CEL(input, label)
        output = model(input)
        _, prediction = torch.max(output.data, 1)
        loss = criterion(output, label)
        # storage minibatch's prediction and label to log container
        for e in prediction:
            log_y_predictions.append(e)
        for e in label.data:
            log_y_trues.append(e)
        epoch_loss += loss.cpu().data.numpy()[0]
    log_losses.append(epoch_loss)

    mu.save_matrics_model(log_y_trues, log_y_predictions, log_losses, 'resnet18', model=model)

train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=40)

# torch.save(model, 'Thynet.pkl')  # 保存整个神经网络的结构和模型参数
# torch.save(model.state_dict(), 'Thynet.pkl')  # 只保存神经网络的模型参数

# model.load_state_dict(torch.load("Thynet.pkl"))
# print(torch.max(model(image2modelinput("ThyImage/hyperthyreosis1_251701.jpg", (255, 255))), 1)[1])