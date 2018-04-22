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

def GetAcc(metrics_path, train_saved):
    '''
    :param metrics_path: 指标文件的路径
    :return: 获取每个epoch的精度，list类型返回
    '''
    metrics = torch.load(metrics_path)
    mix_acc = []
    for i in metrics:
        mix_acc.append(i['average_precision'])
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

    # mu.VisdomDrawLines(train_acc, test_acc)
    return train_acc, test_acc

_, rm_0      = GetAcc('metrics/Sequential_rm_0')
_, rm_05     = GetAcc('metrics/Sequential_rm_0.5')
_, rm_08     = GetAcc('metrics/Sequential_rm_0.8')
_, rm_random = GetAcc('metrics/Sequential_rm_random')

_, mu_0      = GetAcc('metrics/Sequential_mu_0')
_, mu_05     = GetAcc('metrics/Sequential_mu_0.5')
_, mu_08     = GetAcc('metrics/Sequential_mu_0.8')
_, mu_random = GetAcc('metrics/Sequential_mu_random')

mu.VisdomDrawLines(rm_0, rm_05, rm_08, rm_random,
                   mu_0, mu_05, mu_08, mu_random,
                   legends=['rm0', 'rm0.5', 'rm0.8', 'rm_random',
                            'mu0', 'mu0.5', 'mu0.8', 'mu_random'],
                   title='random mix and mixup')

assert 0
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

        # print('#'*100)

rt_m = metrics2npmaxtrix(GetEpochPerformance('metrics/ResNet_th', 141)['classify_report'])
r_m = metrics2npmaxtrix(GetEpochPerformance('metrics/ResNet', 141)['classify_report'])
dt_m = metrics2npmaxtrix(GetEpochPerformance('metrics/MyDenseNet', 141)['classify_report'])
d_m = metrics2npmaxtrix(GetEpochPerformance('metrics/DenseNet', 141)['classify_report'])
v_m = metrics2npmaxtrix(GetEpochPerformance('metrics/VGG', 25)['classify_report'])

metrics = [dt_m, d_m, r_m, rt_m, v_m]

hashimoto_thyroiditis1 = []
hyperthyreosis1 = []
normal1 = []
postoperative1 = []
subacute_thyroiditis1 = []
subhyperthyreosis1 = []
avg = []
for m in metrics:
    hashimoto_thyroiditis1.append(m[0])
    hyperthyreosis1.append(m[1])
    normal1.append(m[2])
    postoperative1.append(m[3])
    subacute_thyroiditis1.append(m[4])
    subhyperthyreosis1.append(m[5])
    avg.append(m[6])

print(avg)
assert 0
import numpy as np
import pandas as pd

# prepare for data
data = avg
# data = dt_test_acc
data_df = pd.DataFrame(data)

# change the index and column name
l = ["hashimoto_thyroiditis1", "hyperthyreosis1", "normal1", "postoperative1", "subacute_thyroiditis1", "subhyperthyreosis1"]
data_df.columns = ['precision','recall rate','f1-score']
# data_df.columns=['Proposed']
data_df.index = ['Ours', 'Densenet121', 'ResNet101', 'InceptionV3', 'VGG19']

# create and writer pd.DataFrame to excel
writer = pd.ExcelWriter('metrics excel/7.xlsx')
data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()