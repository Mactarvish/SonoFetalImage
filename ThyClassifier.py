import os
import torch
from skimage import io, transform
import MaUtilities as mu
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
from torchvision import datasets, models, transforms, utils
import datetime
import copy
from FPO import FPO

from models.resnet_th import resnet_th
from models.my_densenet import mydensenet121
from models.verify_net import VerifyNet
from models.preact_resnet import PreActResNet18
import datasets.ThyDataset as ThyDataset
import datasets.cifar10 as cifar10
# from datasets.ThyDataset import ThyDataset

NUM_CLASSES = 10
AUGMENTATION_STRATEGY = None
# torch.manual_seed(55)
# torch.cuda.manual_seed(59)

print(torch.cuda.is_available())
train_loader = DataLoader(cifar10.Cifar10(train=True,  dataset_size=1/5, image_transform=cifar10.transformer), shuffle=False, batch_size=10, num_workers=10)
val_loader   = DataLoader(cifar10.Cifar10(train=False, dataset_size=1/5, image_transform=cifar10.transformer), shuffle=False, batch_size=10, num_workers=10)

# train_loader = DataLoader(ThyDataset.ThyDataset(train=True, image_transform=ThyDataset.transformer, pre_transform=None),  shuffle=True, batch_size=5, num_workers=5)
# val_loader   = DataLoader(ThyDataset.ThyDataset(train=False, image_transform=ThyDataset.transformer, pre_transform=None), shuffle=True, batch_size=5, num_workers=5)

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
    :return: 
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

def log(train, save, show_detail, calculate_metrics=True):
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
                mu.log_metrics(train, log_y_trues, log_y_predictions, log_loss, model=model, save=save,
                               show_detail=show_detail, save_path=current_save_folder, note=NOTE)
        return wrapper
    return decorator

def calculate_loss(model_output, target):
    '''
    由于2018.5.6把任务统一作为了回归任务处理，所以target都是one-hot-like，用LogSoftmax计算loss
    :param model_output: 
    :param target: [0,2,5,2,1,3]的one-hot Variable/Tensor
    :return: 
    '''
    assert isinstance(target, Variable) or isinstance(target, torch.Tensor)
    # target_np = None
    # if isinstance(target, Variable):
    #     target_np = target.cpu().data.numpy()
    # if isinstance(target, torch.Tensor):
    #     target_np = target.cpu().numpy()
    # # if AUGMENTATION_STRATEGY == None:
    # #     target_np = mu.to_categorical(target_np, num_classes=NUM_CLASSES)
    # target = Variable(torch.from_numpy(target_np)).cuda()

    m = nn.LogSoftmax()
    t = -m(model_output)
    if not isinstance(target, Variable):
        target = Variable(target).cuda()
    loss = t * target
    loss = torch.sum(loss) / 128

    return loss

def nums2onehots(labels):
    '''
    把[0,2,1] 转换为[[1,0,0],
                    [0,0,1],
                    [0,1,0]]
    :param labels: LongTensor of size n
    :return: FloatTensor
    '''
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
    assert scale == 1, 'Experience shown that 1 is the best.'
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
            mixed_input = random_mix(aug_factor[i], input1, input2, scale=1)
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

class MaSGD(optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(MaSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                 weight_decay=weight_decay, nesterov=nesterov)
        self.last_dps = []

    def step_back(self):
        '''
        反更新参数，抵消上一次的优化效果。
        :return: 
        '''
        d_p_g = iter(self.last_dps)
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = next(d_p_g)
                p.data.add_(group['lr'], d_p)
                self.last_dps = []

        return loss


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.last_dps = []
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                P_GRAD = p.grad.data.clone()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.grad.data = P_GRAD
                self.last_dps.append(d_p)
                p.data.add_(-group['lr'], d_p)

        return loss

class GILR():
    def __init__(self, optimizer, model, last_epoch=-1):
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.optimizer = optimizer
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.model = model

    def step(self, inputs, labels, epoch):
        self.inputs = inputs
        self.labels = labels
        # if epoch < 5:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = 0.005
        # elif 5 <= epoch < 10:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = 0.0005
        # else:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        def test_fun(lrs):
            '''
            Loss = test_fun(lrs)
            :param lrs: 
            :return: 
            '''
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                # WARNED
                param_group['lr'] = lr

            # 用当前学习率更新一波参数
            self.optimizer.step()
            output = self.model(self.inputs)
            # 试验完毕，回退原先状态
            self.optimizer.step_back()

            loss = calculate_loss(output, self.labels)
            loss_float = loss.cpu().data.numpy()
            return loss_float
        fpo = FPO(test_fun, iterations_num=1, stop_loss_min=0.0001, pop_size=3, p=0.8, dim=len(self.base_lrs), val_min=0.00001, val_max=0.5)
        loss, lrs = fpo.run()

        return lrs

@log(train=True, save=True, show_detail=True, calculate_metrics=True)
def train(model, criterion, optimizer, scheduler, epoch, augmentation_strategy):
    model.train(True)
    if not GI:
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

        loss = calculate_loss(output, labels)
        prediction = output.data # LongTensor
        loss.backward()

        # FODPSO ICS FPA
        if GI:
            scheduler.step(inputs, labels, epoch)
        optimizer.step()

        prediction_cpu = prediction.cpu()
        label_cpu = labels.cpu().data
        loss_cpu = loss.cpu().data.numpy()[0]
        yield prediction_cpu, label_cpu, loss_cpu

@log(train=False, save=True, show_detail=True)
def test(model, criterion, epoch):
    model.train(False)
    for (input, label) in val_loader:
        input, label = Tensor2Variable(input, label, loss_type='CEL')
        output = model(input)
        _, prediction = torch.max(output.data, 1)
        loss = criterion(output, label)
        regression_loss = calculate_loss(output, nums2onehots(label.cpu().data))
        loss = regression_loss
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

current_save_folder = 'metrics'
if not os.path.exists(current_save_folder):
    os.makedirs(current_save_folder)

GI = True
NOTE = None

if GI:
    NOTE = 'GI'
else:
    NOTE = 'normal'

resnet18 = nn.Sequential(models.resnet18(pretrained=True), nn.Linear(1000, NUM_CLASSES))
# densenet121 = nn.Sequential(models.densenet121(pretrained=True), nn.Linear(1000, NUM_CLASSES))

for model in [resnet18]:
    NOTE = NOTE #+ model[0].__class__.__name__
    freer_gpu = mu.get_freer_gpu()
    print('using gpu %d' % freer_gpu)
    torch.cuda.set_device(freer_gpu)
    epochs = 50
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()

    def divide_model_params(model):
        assert isinstance(model, nn.Sequential)
        if isinstance(model[0], models.ResNet):
            trainable_models = [model[0].conv1, model[0].bn1, model[0].layer1, model[0].layer2, model[0].layer3, model[0].layer4, model[0].fc]  # len = 7
            trainable_models.append(model[1])
            params = [x.parameters() for x in trainable_models]
            params_dict = [{'params': p} for p in params]

            return params_dict
        elif isinstance(model[0], models.DenseNet):
            trainable_models = [model[0].features.conv0, model[0].features.norm0, model[0].features.norm5, model[0].features.transition1,
                                model[0].features.transition2, model[0].features.transition3, model[0].features.denseblock1,
                                model[0].features.denseblock2, model[0].features.denseblock3, model[0].features.denseblock4, model[0].classifier]
            trainable_models.append(model[1])
            params = [x.parameters() for x in trainable_models]
            params_dict = [{'params': p} for p in params]

            return params_dict

    optimizer = None
    scheduler = None
    if GI:
        optimizer = MaSGD(divide_model_params(model), lr=0.005, momentum=0, weight_decay=0)
        scheduler = GILR(optimizer, model=model, last_epoch=-1)
    else:
        optimizer = optim.SGD(divide_model_params(model), lr=0.005, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        print('{} Epoch {}/{}'.format(model.__class__.__name__, epoch, epochs))
        print('-' * 10)
        since = time.time()

        train(model, criterion, optimizer, scheduler, epoch=epoch, augmentation_strategy=AUGMENTATION_STRATEGY)
        test(model, criterion, epoch=epoch)

        time_diff = time.time() - since
        print('epoch complete in {:0.2f} seconds on gpu {}'.format(time_diff, torch.cuda.current_device()))
        print()

print('Model execution completed on ' + datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S'))
