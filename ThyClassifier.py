import os
import torch
from skimage import io, transform
import MaUtilities as mu
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.models import vgg
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import time
import cv2
from torchvision import datasets, models, transforms, utils
import sqlite3
import datetime
import copy

from models.resnet_th import resnet_th
from models.my_densenet import mydensenet121
from models.verify_net import VerifyNet
from models.preact_resnet import PreActResNet18
from datasets.ThyDataset import ThyDataset


use_gpu = torch.cuda.is_available()
image_location = "/home/hdl2/Desktop/SonoFetalImage/ThyImage/"
# label_map = {"hashimoto_thyroiditis1": 0, "hyperthyreosis1": 1, "normal1": 2, "postoperative1": 3, "subacute_thyroiditis1": 4, "subhyperthyreosis1": 5}
NUM_CLASSES = 6
DATA_AUGUMENTATION = True
# torch.manual_seed(55)
# torch.cuda.manual_seed(59)

transformer = transforms.Compose([
    mu.ResizeImage((255, 255)),
    # mu.ResizeImage((299, 299)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.333, 0.0586, 0.023], std=[0.265, 0.138, 0.0224])
    ]
)
# [0.33381739584119885, 0.05862841105989082, 0.023407234558809612], [0.26509894104564447, 0.13794070296714034, 0.022363285181095156]
train_loader = DataLoader(ThyDataset(train=True, image_transform=transformer, pre_transform=None),  shuffle=True, batch_size=5, num_workers=5)
val_loader   = DataLoader(ThyDataset(train=False, image_transform=transformer, pre_transform=None), shuffle=True, batch_size=5, num_workers=5)

# model = resnet_th(pretrained=True)
# model = VerifyNet(input_shape=(255, 255, 3), num_classes=6)
# model = models.resnet18(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.inception_v3(pretrained=False, num_classes=6)
# model = PreActResNet18(num_classes=6)
# model = mydensenet121(pretrained=True)
# model = models.densenet121(pretrained=True)

# if use_gpu:
#     # model = nn.DataParallel(model)
#     model = model.cuda()
# l = list(model.named_parameters())
# list_pet = []
# list_t = []
# # layer1.1.threshold_2
# for e in l:
#     if 'threshold' not in e[0]:
#         list_pet.append(e[1])
#     else:
#         list_t.append(e[1])
#
# def para_exc_threshold():
#     for e in list_pet:
#         yield e
# def para_threshold():
#     for e in list_t:
#         yield e

def Tensor2Variable(input, label, loss_type):
    """
    :param input: Torch.Tensor
    :param label: Torch.Tensor
    :return: Torch.Tensor.cuda() obey CrossEntropyLoss type
    """
    assert loss_type == 'CEL' or loss_type == 'LSM'
    input = Variable(input).cuda().float()
    if loss_type == 'CEL':
        label = Variable(label).cuda().long()
    if loss_type == 'LSM':
        label = Variable(label).cuda().float()
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

def log(save, show_detail, calculate_metrics=True):
    '''
    :param save: 是否把指标保存成为本地文件
    :param show_detail: 是否在程序运行过程中print指标
    :param calculate_metrics: 是否计算指标，如果是false，上边两个参数就不起作用
    :return: 
    '''
    def decorator(gen):
        # @functools.wraps(gen)
        def wrapper(*args, **kwargs):
            f = gen(*args, **kwargs)
            log_loss = None
            log_y_predictions = []
            log_y_trues = []
            epoch_loss = 0
            # 执行一个epoch运算

            for (prediction, label, loss) in f:
                for e in prediction:
                    log_y_predictions.append(e)
                for e in label:  # label: LongTensor
                    log_y_trues.append(e)
                epoch_loss += loss

            log_loss = epoch_loss
            if calculate_metrics:
                mu.log_metrics(log_y_trues, log_y_predictions, log_loss, model=model, save=save,
                               show_detail=show_detail, save_path=current_save_folder, note=note)
        return wrapper
    return decorator

def calculate_loss(model_output, target, task):
    '''
    对于分类任务，直接CrossEntropyLoss计算loss
    对于回归任务，先把target每个都one-shot向量化再用LogSoftmax计算loss，由于增广之后labels都是one-hot-like
    :param model_output: 
    :param target: [0,2,5,2,1,3]
    :param task: 
    :return: 
    '''
    assert task == 'c' or task == 'r'
    if task == 'r':
        # print('model_output:')
        # print(model_output)
        # print('target:')
        # print(target)
        # print('target(numpy):')
        # print(target.cpu().data.numpy())
        target_np = target.cpu().data.numpy()
        if not DATA_AUGUMENTATION:
            target_np = mu.to_categorical(target_np, num_classes=NUM_CLASSES)
        target = Variable(torch.from_numpy(target_np)).cuda()
        # print(target)
        m = nn.LogSoftmax()
        loss = -m(model_output) * target
        loss = torch.sum(loss) / 128
    else:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(model_output, target)

    return loss

def nums2onehots(labels):
    '''
    把[0,2,1] 转换为[[1,0,0],
                    [0,0,1],
                    [0,1,0]]
    :param labels: LongTensor of size n
    :return: 
    '''
    for i in labels:
        mu.to_categorical(i, NUM_CLASSES)
    oh_np = mu.to_categorical(labels.numpy(), num_classes=NUM_CLASSES)
    return torch.from_numpy(oh_np).float()

def mixup(p, image1_np, image2_np):
    '''
    给定线性因子p，新图像=image1_np*p + image2_np*(1-p)
    :param p: 
    :param image1_np: 
    :param image2_np: 
    :return: 
    '''
    assert image1_np.shape == image2_np.shape
    new = image1_np * p + image2_np * (1-p)
    new = np.uint8(new)
    return new


def random_mix(p, image1_np, image2_np, scale=1):
    '''
    给定概率p，混合得到的图像有p的区域来源于image1_np，有1-p的区域来源于image2_np
    :param p: 
    :param image1_np: cwh NOT whc
    :param image2_np: 
    :return: cwh
    '''
    # cwh -> whc
    image1_np = mu.np_channels2c_last(image1_np)
    image2_np = mu.np_channels2c_last(image2_np)
    assert image1_np.shape == image2_np.shape
    def roundingplus1(x, d):
        # f(x) = [x] if x is integer, otherwise [x] + 1
        assert d > 0 and x > d
        if x / d == x // d:
            return x // d
        else:
            return x // d + 1
    pm = np.random.binomial(1, p, (roundingplus1(image1_np.shape[0], scale), roundingplus1(image1_np.shape[1], scale)))
    patches1 = scatter_image(image1_np, scale=scale)
    patches2 = scatter_image(image2_np, scale=scale)
    new_patches = copy.deepcopy(patches2)

    for i in range(len(pm)):
        for j in range(len(pm[i])):
            if pm[i][j] == 1:
                new_patches[i][j] = patches1[i][j]
            else:
                new_patches[i][j] = patches2[i][j]
    new_sample = unscatter_images(new_patches)
    # whc -> cwh
    new_sample = mu.np_channels2c_first(new_sample)

    return new_sample


def scatter_image(image_np, scale):
    '''
    把image_np分解为若干scale*scale的小图块
    :param image_np: whc
    :param scale: 
    :return: 
    '''
    patches = []
    for i in range(0, image_np.shape[0], scale):
        row = []
        for j in range(0, image_np.shape[1], scale):
            patch = image_np[i: i + scale, j: j + scale]
            row.append(patch)
        patches.append(row)
    return patches


def unscatter_images(patches):
    '''
    把打碎的图片碎片拼合为一张图片
    example:
    [[i1, i2, i3],      [- - -
     [i4, i5, i6],  ->   - - -
     [i7, i8, i9]]       - - -] 
    :param patches: list of lists of image_nps
    :return: np
    '''
    integeral_rows = []
    for row in patches:
        integeral_row = np.column_stack(row)
        integeral_rows.append(integeral_row)
    integeral_image_np = np.row_stack(integeral_rows)
    return integeral_image_np


def augument_data(inputs, labels_oh, strategy, num_augumented=5):
    '''
    增广前后，数据类型应当保持不变（Done）
    对一个batch的数据进行增广。（增广num_augumented个样本）
    :param inputs: torch.Tensor
    :param labels: torch.Tensor  labels是one-hot
    :param aug_factor: 对于mixup，该参数是线性组合因子，对于random_mix，该参数是概率组合因子
    :return: 
    '''
    assert strategy == 'mixup' or strategy == 'random_mix'
    # 先把输入都转换为numpy类型
    inputs = inputs.numpy()
    labels_oh = labels_oh.numpy()
    # 随机给样本配对
    combinations = np.random.randint(0, inputs.shape[0], size=(num_augumented, 2))
    new_inputs = []
    new_labels = []

    aug_factor = np.random.uniform(0.5, 1, inputs.shape[0])
    for i, cp in enumerate(combinations):
        input1, label1_oh = inputs[cp[0]], labels_oh[cp[0]]
        input2, label2_oh = inputs[cp[1]], labels_oh[cp[1]]

        # 随机混合
        mixed_input = None
        if strategy == 'random_mix':
            mixed_input = random_mix(aug_factor[i], input1, input2, scale=current_random_mix_scale)
        elif strategy == 'mixup':
            mixed_input = mixup(aug_factor[i], input1, input2)
        mixed_label = aug_factor[i] * label1_oh + (1-aug_factor[i]) * label2_oh

        new_inputs.append(mixed_input)
        new_labels.append(mixed_label)
    # 原始样本和新样本整合成一个minibatch
    new_inputs = np.stack(new_inputs)
    new_labels = np.stack(new_labels)
    all_inputs = np.row_stack([new_inputs, inputs])
    all_labels = np.row_stack([new_labels, labels_oh])
    # 打乱这个minibatch
    index = np.arange(len(all_inputs))
    np.random.shuffle(index)
    all_inputs = all_inputs[index]
    all_labels = all_labels[index]
    # numpy -> torch.Tensor
    all_inputs = torch.Tensor(all_inputs)
    all_labels = torch.Tensor(all_labels)
    # print(all_labels)

    return all_inputs, all_labels

@log(save=False, show_detail=False, calculate_metrics=False)
def train(model, task, criterion, optimizer, scheduler, epoch, augmentation_strategy):
    model.train(True)
    if scheduler != None:
        scheduler.step()
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        labels = nums2onehots(labels)
        # 增广之后，labels都是one-hot-like
        if augmentation_strategy != None:
            inputs, labels = augument_data(inputs, labels, strategy=augmentation_strategy)
        inputs, labels = Tensor2Variable(inputs, labels, loss_type='LSM')
        # run the model
        output = model(inputs)

        loss = calculate_loss(output, labels, task=task)
        prediction = None
        if task == 'c':
            _, prediction = torch.max(output.data, dim=1) # float
        else:
            prediction = output.data # LongTensor
        loss.backward()
        optimizer.step()

        prediction_cpu = prediction.cpu()
        label_cpu = labels.cpu().data
        loss_cpu = loss.cpu().data.numpy()[0]
        yield prediction_cpu, label_cpu, loss_cpu



@log(save=True, show_detail=True)
def test(model, criterion, epoch):
    model.train(False)
    for (input, label) in val_loader:
        input, label = Tensor2Variable(input, label, loss_type='CEL')
        output = model(input)
        _, prediction = torch.max(output.data, 1)
        loss = criterion(output, label)
        loss_cpu = loss.cpu().data.numpy()[0]
        yield prediction, label.data, loss_cpu


# vgg=models.vgg16(pretrained=True)
# vgg.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, NUM_CLASSES),
#         )
# group0 = [resnet_th(pretrained=True), mydensenet121(pretrained=False)]
# group1 = [models.resnet18(pretrained=True), models.densenet121(pretrained=True), vgg]
def GetFileName(scales):
    filenames = []
    for s in scales:
        name = 'rm_' + str(s)
        filenames.append(name)
    return filenames

current_save_folder = 'metrics'
if not os.path.exists(current_save_folder):
    os.makedirs(current_save_folder)

note = None
random_mix_scales_global = [70, 90, 110]
filenames = GetFileName(random_mix_scales_global)

for model in [nn.Sequential(models.resnet18(pretrained=True), nn.Linear(1000, NUM_CLASSES))]:
    for current_random_mix_scale, note in zip(random_mix_scales_global, filenames):
        torch.cuda.set_device(1)
        epochs = 50
        criterion = nn.CrossEntropyLoss()
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.8)
        # optimizer_2 = optim.SGD(para_threshold(), lr=0.0005, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(epochs):
            print('{} Epoch {}/{}'.format(model.__class__.__name__, epoch, epochs))
            print('-' * 10)
            since = time.time()

            train(model, 'r', criterion, optimizer, exp_lr_scheduler, epoch=epoch, augmentation_strategy='mixup')
            test(model, criterion, epoch=epoch)

            time_diff = time.time() - since
            print('epoch complete in {:0.2f} seconds'.format(time_diff))
            print()
print('Model execution completed on ' + datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S'))
