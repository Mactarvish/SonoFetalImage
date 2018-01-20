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

use_gpu = torch.cuda.is_available()
image_location = "/home/hdl2/Desktop/SonoFetalImage/ThyImage/"
label_map = {"hashimoto_thyroiditis1": 0, "hyperthyreosis1": 1, "normal1": 2, "postoperative1": 3, "subacute_thyroiditis1": 4, "subhyperthyreosis1": 5}

connection = sqlite3.connect("ThyDataset")
cu = connection.cursor()
# cu.execute("select * from Train where id=2")
# a = cu.fetchall()[0]
# print(a)

torch.manual_seed(123)
torch.cuda.manual_seed(222)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def image2modelinput(file_name):
    image = Image.open(file_name).convert('L')
    #input = Variable(transforms.ToTensor()(image).cuda())
    #torch.squeeze()
    input = Variable(torch.unsqueeze(transforms.ToTensor()(image), dim=0).cuda())
    #print(input)
    return input

class ResizeImage(object):
    """
    Input an numpy array or a PIL image and return a PIL image with given size "new_size", keeping num of channels unchanged.
    """
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, image):
        image = cv2.resize(np.asarray(image), self.new_size)
        return Image.fromarray(image)

class SizeCoorTransform(object):
    def __init__(self, new_size):
        self.n_size = new_size
    def __call__(self, original_image, original_coordinates):
        assert not isinstance(original_image, np.ndarray), "image must be PIL"
        assert isinstance(original_coordinates, list) or isinstance(original_coordinates, tuple), "original_coordinates must be list or tuple"

        # example: original_coordinates: [(10, 29), (45, 675), (89, 43)] or (432, 12)
        o_coordinates = original_coordinates
        o_size = original_image.size
        image = original_image.resize(self.n_size)

        ph = self.n_size[0] / o_size[0]
        pw = self.n_size[1] / o_size[1]

        n_coordinates = []

        if isinstance(o_coordinates, tuple):
            n_coordinates = (int(o_coordinates[0] * ph), int(o_coordinates[1] * pw))
        else:
            for o_coordinate in o_coordinates:
                n_coordinate = (int(o_coordinate[0] * ph), int(o_coordinate[1] * pw))
                n_coordinates.append(n_coordinate)

        return image, n_coordinates

class BrainSliceDataset(Dataset):
    # inl: image and labels
    def __init__(self, train=True, image_transform=None, pre_transform=None):
        super(BrainSliceDataset, self).__init__()
        self.train = train
        self.image_transform = image_transform
        self.pre_transform = pre_transform
        self.point_labels = self.get_point_labels()
        self.categories = self.get_category()
        self.category_transofm = {"circle": 0, "line": 1, "other": 1}
        #print(self.labels)

    def __getitem__(self, item):
        if self.train == True:
            image = Image.open("/home/hdl2/Desktop/SonoDataset/OriginalImages/%d.jpg" % (item)).convert('L')
            label = self.category_transofm[self.categories[item]]
        else:
            image = Image.open("/home/hdl2/Desktop/SonoDataset/OriginalImages/%d.jpg" % (item + 700)).convert('L')
            label = self.category_transofm[self.categories[item + 700]]

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        # except .git
        if self.train == True:
            return 700
        else:
            return 101
        #return len(os.listdir("/home/hdl2/Desktop/SonoDataset/OriginalImages/")) - 1

    def str2coordinate(self, str):
        '''
        :param str: like '(231,55)'
        :return: str to tuple
        :example: '(231,55)' -> tuple (231, 55)
        '''
        nums = str[1:-1].split(',')
        #print(nums)
        return (int(nums[0]), int(nums[1]))

    def get_point_labels(self):
        f = open("RectLabels.txt")
        labels = {}
        lines = f.readlines()
        for line in lines:
            # print(line)
            elements = line.split(' ')
            coordinates = []
            index = None
            for e in elements:
                #print('e:', e)
                if '.jpg' in e:
                    index = int(e[: -4])
                else:
                    #bad condition, need repairing later
                    if e == '\n':
                        #print('happens')
                        continue
                    coordinates.append(self.str2coordinate(e))
            labels[index] = coordinates
        return labels

    def get_category(self):
        f = open("/home/hdl2/Desktop/SonoDataset/Labels/classify.txt")
        labels = {}
        lines = f.readlines()
        for line in lines:
            print(line)
            number = int(line.split('.')[0])
            category = line.split(' ')[-1][:-1]
            labels[number] = category
        print(labels)
        return labels


class ThyDataset(Dataset):
    def __init__(self, train=True, image_transform=None, pre_transform=None):
        super(ThyDataset, self).__init__()
        self.train = train
        self.image_transform = image_transform
        self.pre_transform = pre_transform

    def __getitem__(self, item):
        if self.train == True:
            cu.execute("select * from Train where id=%d" % (item + 1))
        else:
            cu.execute("select * from Validation where id=%d" % (item + 1))

        record = cu.fetchall()[0]
        image_name = record[1]
        category = record[2]
        image = Image.open(image_location + image_name)#.convert('L')
        label = label_map[category]

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        if self.train == True:
            return 4150
        else:
            return 300

    def str2coordinate(self, str):
        '''
        :param str: like '(231,55)'
        :return: str to tuple
        :example: '(231,55)' -> tuple (231, 55)
        '''
        nums = str[1:-1].split(',')
        #print(nums)
        return (int(nums[0]), int(nums[1]))

    def get_point_labels(self):
        f = open("RectLabels.txt")
        labels = {}
        lines = f.readlines()
        for line in lines:
            # print(line)
            elements = line.split(' ')
            coordinates = []
            index = None
            for e in elements:
                #print('e:', e)
                if '.jpg' in e:
                    index = int(e[: -4])
                else:
                    #bad condition, need repairing later
                    if e == '\n':
                        #print('happens')
                        continue
                    coordinates.append(self.str2coordinate(e))
            labels[index] = coordinates
        return labels

    def get_category(self):
        f = open("/home/hdl2/Desktop/SonoDataset/Labels/classify.txt")
        labels = {}
        lines = f.readlines()
        for line in lines:
            print(line)
            number = int(line.split('.')[0])
            category = line.split(' ')[-1][:-1]
            labels[number] = category
        print(labels)
        return labels


transformer = transforms.Compose([
    ResizeImage((255, 255)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

train_loader = DataLoader(ThyDataset(train=True, image_transform=transformer, pre_transform=None), batch_size=6, num_workers=6)
val_loader = DataLoader(ThyDataset(train=False, image_transform=transformer, pre_transform=None), batch_size=6, num_workers=6)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(ThyDataset(train=True)), "val": len(ThyDataset(train=False))}

model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)

if use_gpu:
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(23040, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 10 * 110 * 110
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 20 * 52 * 52
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2)) # 40 * 24 * 24
        x = x.view(-1, 23040)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print("target:", target)
        print(data.shape, type(data))
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print("epoch: %d %d/%d (%0.0f%%)\tLoss: %0.6f" % (epoch, epoch * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))

good_prediction_count = 0

def validation():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        #F.mse_loss()
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    test_loss /= len(val_loader.dataset)
    print("\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)\n" %(test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
    if correct >= 95:
        global good_prediction_count
        good_prediction_count += 1
        if good_prediction_count == 3:
            torch.save(model.state_dict(), 'weights.pkl')
            #print("training is over!")
            assert 0, "training is over!"
    else:
        good_prediction_count = 0

    time.sleep(1)

def test(n_image):
    model.load_state_dict(torch.load('weights.pkl'))
    model.eval()
    file_name = "/home/hdl2/Desktop/SonoDataset/OriginalImages/%d.jpg" % (n_image)
    data = image2modelinput(file_name)
    output = model(data)
    prediction = output.data.max(1, keepdim=True)[1]
    prediction = prediction.cpu().numpy()[0][0]
    couple = {0: "circle", 1: "other"}
    print(couple[int(prediction)])

circle_index = 0
def classify(): # 0~10043
    model.load_state_dict(torch.load('weights.pkl'))
    model.eval()
    for i in range(10044):
        file_name = "/home/hdl2/Desktop/SonoDataset/OriginalImages/%d.jpg" % (i)
        data = image2modelinput(file_name)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        prediction = prediction.cpu().numpy()[0][0]
        if prediction == 0:
            print(i)
            global circle_index
            shutil.copyfile(file_name, "/home/hdl2/Desktop/SonoDataset/Circles/" + "%d.jpg" % (circle_index))
            circle_index += 1

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
