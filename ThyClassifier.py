import os
import torch
import MaUtilities as mu
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.models import vgg
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import time
from torchvision import datasets, models, transforms, utils
import datetime
import copy
from colorama import Fore
from FPA_optimizer import FPA

from models.resnet_th import resnet_th
from models.my_densenet import mydensenet121
from models.verify_net import VerifyNet
from models.preact_resnet import PreActResNet18
from models.mutibranch_resnet import branch_resnet18, MultibranchResNet
from models.multiway_resnet import MultiwayResnet_AllTrained, MultiwayResnet_FcTrained
import datasets.ThyDataset as ThyDataset
import datasets.cifar10 as cifar10
# from datasets.ThyDataset import ThyDataset

NUM_CLASSES = 10
BATCH_SIZE = 20
DATASET_SIZE = 1/5

FPA_TESTSET_LOSS = False

AUGMENTATION_STRATEGY = None
DEBUG_MODE = False
if DEBUG_MODE:
    DATASET_SIZE =1/500

torch.manual_seed(60)
torch.cuda.manual_seed(60)
# torch.backends.cudnn.deterministic = True

train_loader        = DataLoader(cifar10.Cifar10(mode='train',  dataset_size=DATASET_SIZE, binclassify=None, image_transform=cifar10.transformer), shuffle=False, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE)
test_loader         = DataLoader(cifar10.Cifar10(mode='test', dataset_size=DATASET_SIZE, binclassify=None, image_transform=cifar10.transformer), shuffle=False, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE)
validation_loader   = DataLoader(cifar10.Cifar10(mode='validation', dataset_size=DATASET_SIZE, binclassify=None, image_transform=cifar10.transformer), shuffle=False, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE)

# train_loader = DataLoader(ThyDataset.ThyDataset(train=True, image_transform=ThyDataset.transformer, pre_transform=None),  shuffle=True, batch_size=5, num_workers=5)
# test_loader   = DataLoader(ThyDataset.ThyDataset(train=False, image_transform=ThyDataset.transformer, pre_transform=None), shuffle=True, batch_size=5, num_workers=5)


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

def calculate_loss(model_output, target):
    '''
    :param model_output: 
    :param target: [0,2,5,2,1,3]的one-hot Variable/Tensor
    :return: 
    '''
    assert isinstance(target, Variable) or isinstance(target, torch.Tensor)

    def get_max_second_max(output):
        '''
        返回给定二维tensor的每一行的最大值索引和次大值索引
        :param output_tensor: 
        :return:
        '''
        output_tensor = None
        if isinstance(output, Variable):
            output_tensor = output.cpu().data
        else:
            output_tensor = output
        assert len(output_tensor.size()) == 2, 'tensor must be 2d, but got %dd' % len(output_tensor.size())
        output_np = output_tensor.clone().numpy()
        m = output_np.argmax(axis=1)

        for i in range(output_np.shape[0]):
            output_np[i, m[i]] = -np.inf
        sm = output_np.argmax(axis=1)

        return m, sm

    outstandings = []
    m, sm = get_max_second_max(model_output)
    for i in range(model_output.size()[0]):
        outstanding = model_output[i, m[i]] - model_output[i, sm[i]]
        outstanding = outstanding.view(1, 1)
        outstandings.append(outstanding)
    ol = 0
    for o in outstandings:
        ol += o
    ol = 1/ol

    m = nn.LogSoftmax()
    t = -m(model_output)
    if not isinstance(target, Variable):
        target = Variable(target).cuda()
    loss = t * target
    loss = torch.sum(loss) / BATCH_SIZE
    loss = loss + ol.view(1)

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
    '''
    ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！如果启用GI weight_decay，这里的weight_decay可能就没什么卵用了
    '''
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(MaSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                 weight_decay=weight_decay, nesterov=nesterov)
        self.current_dps = []
        self.last_dps = []

    def step_back(self):
        '''
        反更新参数，抵消上一次的优化效果。
        :return: 
        '''
        d_p_g = iter(self.current_dps)
        momentum_buffers = iter(self.last_dps)

        loss = None
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                if momentum != 0:
                    param_state = self.state[p]
                    last_dp = next(momentum_buffers)
                    if last_dp is None:
                        param_state.pop('momentum_buffer')
                    else:
                        param_state['momentum_buffer'] = last_dp
                d_p = next(d_p_g)
                p.data.add_(group['lr'], d_p)
                self.current_dps = []
                self.last_dps = []

        return loss

    def step(self, FPA_mode, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.current_dps = []
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
                d_p = p.grad.data.clone()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        self.last_dps.append(None)
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        self.last_dps.append(param_state['momentum_buffer'].clone())
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                self.current_dps.append(d_p)
                p.data.add_(-group['lr'], d_p)
        if not FPA_mode:
            # 非FPA调优状态，那么本次step后就是下一个mini-batch的初始值
            self.current_dps = []
            self.last_dps = []
        return loss

class GILR():
    '''
    为optimizer服务，更新optimizer的param_groups里的每个group里的各种超参数
    '''
    def __init__(self, optimizer, model, update_hyparams_names, update_weight_decays, update_momentums, last_epoch=-1):
        '''
        :param optimizer: 
        :param model: 
        :param update_hyparams_names:  ['lr', 'weight_decay', 'momentum']
        :param update_weight_decays: 
        :param update_momentums: 
        :param last_epoch: 
        '''
        self.update_hyparams_names = None
        if update_hyparams_names == None:
            raise NotImplementedError
        elif isinstance(update_hyparams_names, list):
            self.update_hyparams_names = update_hyparams_names
            for group in optimizer.param_groups:
                for p in self.update_hyparams_names:
                    assert p in group
        else:
            assert 0

        self.optimizer = optimizer
        self.num_groups = len(self.optimizer.param_groups)
        self.model = model
        self.update_weight_decays = update_weight_decays
        self.update_momentums = update_momentums


    def step(self, inputs, labels, epoch):
        self.inputs = inputs
        self.labels = labels

        # 从测试集中随机抽取一个batch
        randint = int(np.random.random_integers(0, 30))
        self.test_inputs, self.test_labels = val_list[randint]
        self.test_inputs = self.test_inputs.cuda()
        self.test_labels = self.test_labels.cuda()

        wo = model[1]._parameters['weight'].clone()
        hyperparam_groups = self.get_hyperparams(epoch) # hyperparam_groups: [[lr_g1, lr_g2, lr_g3], [wd_g1, wd_g2, gd_g3], ...]
        # print(Fore.RED, wo - model[1]._parameters['weight'], Fore.BLACK)
        assert len(hyperparam_groups) == len(self.update_hyparams_names), 'Number of FPA-upated hyperparameters must equal to number of specified params'
        for hyp, param_group in zip(zip(*hyperparam_groups), self.optimizer.param_groups): # hyp: (lr_g1, wd_g1, mom_g1) param_group: [g1, g2,
            for (p_key, p_value) in zip(self.update_hyparams_names, hyp):
                param_group[p_key] = p_value

    def get_hyperparams(self, epoch):
        def test_fun(hyperparams):
            '''
            Loss = test_fun(lrs)
            :param lrs: 
            :return: 
            '''
            assert isinstance(hyperparams, list) and isinstance(hyperparams[0], list)
            update_hyperparams = {}
            for i, hyperparam_names in enumerate(self.update_hyparams_names):
                update_hyperparams[hyperparam_names] = hyperparams[i]

            for i, param_group in enumerate(self.optimizer.param_groups):
                for key in update_hyperparams.keys():
                    param_group[key] = update_hyperparams[key][i]

            # 设定网络的输入和输出
            if FPA_TESTSET_LOSS:
                self.inputs = self.test_inputs
                self.labels = self.test_labels

            bu = copy.deepcopy(model.state_dict())
            w1 = model[1]._parameters['weight'].clone()
            # 用当前学习率更新一波参数
            self.optimizer.step(FPA_mode=True)
            output = self.model(self.inputs)
            # 试验完毕，回退原先状态
            model.load_state_dict(bu)
            # self.optimizer.step_back()
            # print(Fore.GREEN, w1 - model[1]._parameters['weight'], Fore.BLACK)
            # 损失作为适应度函数
            loss = calculate_loss(output, self.labels)
            loss_float = loss.cpu().data.numpy()
            return loss_float

        fpa = FPA(fitness_function=test_fun, num_iteration=3, num_pollen=5, p_lp=0.8, conditions=[(self.num_groups, 0.00001, 0.01)])
        fitness, pollen = fpa.run(epoch)
        loss = fitness
        # print('Loss:', loss)
        hyperparam_groups = pollen.components

        return hyperparam_groups

calculate_metrics = True
show_detail = False
save = True
lrs = []
# @log(train=True, save=True, show_detail=False, calculate_metrics=True)
def train(model, criterion, optimizer, scheduler, epoch, augmentation_strategy):
    # model_2 = copy.deepcopy(model).cuda()
    # optimizer_2 = MaSGD(model_2.parameters(), lr=0.005, momentum=0, weight_decay=0)
    # scheduler_2 = GILR(optimizer_2, model=model_2, update_hyparams_names=['lr'], update_weight_decays=False,
    #                    update_momentums=False, last_epoch=-1)
    # model_2.train(True)

    model.train(True)
    log_loss = None
    log_y_predictions = []
    log_y_trues = []
    epoch_loss = 0
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
        # output_2 = model_2(inputs)
        # print(output - output_2)
        loss = calculate_loss(output, labels)
        # loss_2 = calculate_loss(output_2, labels)
        prediction = output.data # LongTensor
        loss.backward()
        # loss_2.backward()

        # FODPSO ICS FPA
        if GI:
            # scheduler_2.step(inputs, labels, epoch)

            scheduler.step(inputs, labels, epoch)
            # torch.save(model[1]._parameters['weight'].cpu().data, '1ed')
            optimizer.step(FPA_mode=False)
            torch.save(model[1]._parameters['weight'].cpu().data, '1ed')
        else:
            optimizer.step()
        lr = []
        for group in optimizer.param_groups:
            lr.append(group['lr'])
        lrs.append(lr)

        prediction_cpu = prediction.cpu()
        label_cpu = labels.cpu().data
        loss_cpu = loss.cpu().data.numpy()[0]

        for e in prediction_cpu:
            log_y_predictions.append(e)
        for e in label_cpu:  # label: LongTensor
            log_y_trues.append(e)
        epoch_loss += loss_cpu
        torch.save(lrs, mu.cat_filepath(current_save_folder, 'lrs'))
    log_loss = epoch_loss
    if calculate_metrics:
        mu.log_metrics(True, log_y_trues, log_y_predictions, log_loss, model=model, save=save,
                       show_detail=show_detail, save_path=current_save_folder, note=NOTE)

def test(model, criterion, epoch):
    model.train(False)
    log_loss = None
    log_y_predictions = []
    log_y_trues = []
    epoch_loss = 0
    for (inputs, labels) in test_loader:
        labels = nums2onehots(labels)
        inputs, labels = Tensor2Variable(inputs, labels, loss_type='LSM')
        output = model(inputs)

        _, prediction = torch.max(output.data, 1)
        # loss = criterion(output, labels)
        loss = calculate_loss(output, labels)
        loss_cpu = loss.cpu().data.numpy()[0]
        # yield prediction, label.data, loss_cpu
        for e in prediction:
            log_y_predictions.append(e)
        for e in labels.data:  # label: LongTensor
            m, prediction = torch.max(e, 0)
            assert m[0] == 1, "test label must be one-hot"
            prediction = int(prediction[0])
            log_y_trues.append(prediction)
        epoch_loss += loss_cpu
    log_loss = epoch_loss
    if calculate_metrics:
        mu.log_metrics(False, log_y_trues, log_y_predictions, log_loss, model=model, save=save,
                       show_detail=show_detail, save_path=current_save_folder, note=NOTE)

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

val_list = []
for i, (inputs, labels) in enumerate(validation_loader):
    labels = nums2onehots(labels)
    val_inputs, val_labels = Tensor2Variable(inputs, labels, loss_type='LSM')
    val_list.append((val_inputs, val_labels))

GI = True
NOTE = None
current_save_folder = None

if GI:
    NOTE = 'GI'
else:
    NOTE = 'normal'

resnet18 = nn.Sequential(models.resnet18(pretrained=True), nn.Linear(1000, NUM_CLASSES))
# resnet18 = torch.load('normal0_Sequential.pkl')


# verifynet = VerifyNet((3, 32, 32), num_classes=10)
# multiway_resnet = MultiwayResnet_FcTrained()
# densenet121 = nn.Sequential(models.densenet121(pretrained=True), nn.Linear(1000, NUM_CLASSES))

for model in [resnet18]:
    freer_gpu = mu.get_freer_gpu()
    NOTE = NOTE + str(freer_gpu)
    print('Using gpu %d.' % freer_gpu)
    print(torch.cuda.is_available())
    torch.cuda.set_device(freer_gpu)

    metrics_folder_name = input('metrics folder: ')
    if metrics_folder_name == 'l' or metrics_folder_name == 'L':
        metrics_folder_name = torch.load('last_metrics_folder_name')
        print('metrics saved to %s' % metrics_folder_name)
    else:
        torch.save(metrics_folder_name, 'last_metrics_folder_name')
    current_save_folder = 'metrics/%s/gpu%d' % (metrics_folder_name, freer_gpu)
    if not os.path.exists(current_save_folder):
        os.makedirs(current_save_folder)
    if len(os.listdir(current_save_folder)) == 0:
        print('Metrics folder is empty. Training model.')
    else:
        if input('Clear metrics folder?') == 'y':
            def clear_folder(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        clear_folder(c_path)
                    else:
                        os.remove(c_path)


            clear_folder(current_save_folder)
            print('Clear.')
        else:
            print('Files kept. Continue?')
            assert input() == 'y', 'User terminated process.'

    epochs = 100
    criterion = nn.CrossEntropyLoss()
    model = model.cuda()

    def divide_model_params(model):
        # assert isinstance(model, nn.Sequential)
        params_dict = None
        trainable_models = None
        if isinstance(model, MultibranchResNet) or isinstance(model, MultiwayResnet_AllTrained):
            return model.parameters()
        if isinstance(model, MultiwayResnet_FcTrained):
            return model.fc_ways.parameters()
        if isinstance(model, VerifyNet):
            trainable_models = [model.linear1, model.linear2]
        elif isinstance(model[0], models.ResNet):
            trainable_models = [model[0].conv1, model[0].bn1, model[0].layer1, model[0].layer2, model[0].layer3, model[0].layer4, model[0].fc]  # len = 7
            trainable_models.append(model[1])
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
        optimizer = MaSGD(model.parameters(), lr=0.005, momentum=0, weight_decay=0)
        scheduler = GILR(optimizer, model=model, update_hyparams_names=['lr'], update_weight_decays=False, update_momentums=False, last_epoch=-1)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)#multiway_resnet_lr0.5

    for epoch in range(epochs):
        print('{} Epoch {}/{}'.format(model.__class__.__name__, epoch, epochs))
        print('-' * 10, ' ', datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S'))
        since = time.time()

        train(model, criterion, optimizer, scheduler, epoch=epoch, augmentation_strategy=AUGMENTATION_STRATEGY)
        test(model, criterion, epoch=epoch)

        time_diff = time.time() - since
        print('epoch complete in {:0.2f} seconds on gpu {}'.format(time_diff, torch.cuda.current_device()))
        print()

print('Model execution completed on ' + datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S'))
