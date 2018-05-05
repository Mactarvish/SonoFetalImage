import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import MaUtilities as mu
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import torchvision
from torchvision import datasets, models, transforms, utils
from visdom import Visdom
viz = Visdom()

def GetBestPerformance(metrics_path):
    '''
    :param metrics_path: 指标文件的路径
    :return: 最佳一次epoch的指标
    '''
    metrics = torch.load(metrics_path)
    best = None
    ts = 0
    for i in metrics:
        if i['score'] > ts:
            ts = i['score']
            best = i
    return i

def GetEpochPerformance(metrics_path, epoch):
    '''
    :param metrics_path: 指标文件的路径
    :return: 指定epoch的指标
    '''
    assert isinstance(epoch, int)
    metrics = torch.load(metrics_path)
    return metrics[epoch]

def GetAcc(metrics_path, train_saved=False):
    '''
    :param metrics_path: 指标文件的路径
    :return: 获取每个epoch的精度，list类型返回
    '''
    metrics = torch.load(metrics_path)
    mix_acc = []
    for i in metrics:
        mix_acc.append(i['average_precision'])
    # mix_acc.append(i['average_accuracy'])
    train_acc = []
    test_acc = []
    if train_saved:
        # half train accuracy and half test accuracy
        for i in range(len(mix_acc)):
            if i % 2 == 0:
                train_acc.append(mix_acc[i])
            else:
                test_acc.append(mix_acc[i])
    else:
        # all test accuracy
        for i in range(len(mix_acc)):
            test_acc.append(mix_acc[i])

    return train_acc, test_acc

def metrics2npmaxtrix(metrics):
    '''
    例子：
    rt_m = metrics2npmaxtrix(GetEpochPerformance('metrics/ResNet_th', 161)['classify_report'])
    :param metrics: 指标矩阵（str）
    :return: 不带表头的指标矩阵（numpy）
    '''
    rows = metrics.split('\n')
    # 删除前两行（标题和空白行）
    rows = rows[2:]
    rows.pop(6)
    rows.pop(7)
    # rows[6].pop(0)
    nums = []
    for i, r in enumerate(rows):
        # # print(r)
        # # print(r.replace(' ', ''))
        cs = r.split()
        # print(cs)
        valid = None
        if i != 6:
            valid = cs[1:-1]
            # print(cs[1:-1])
        else:
            valid = cs[3:-1]
            # print(cs[3:-1])
        valid = [float(i) for i in valid]
        nums.append(valid)
    # print(np.asarray(nums))
    return np.asarray(nums)

_, GI = GetAcc('metrics/Sequential_GI_weight_decay')
_, normal = GetAcc('metrics/Sequential_normal')

length = min(len(GI), len(normal))
print(len(GI), len(normal))
mu.VisdomDrawLines(GI[:length], normal[:length], legends=['GI', 'normal'], title='diffent learning algorithm')