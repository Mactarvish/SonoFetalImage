import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import MaUtilities as mu
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

class Channel3to1(object):
    def __init__(self):
        pass
    def __call__(self, image):
        pass

class BrainSliceDataset(Dataset):
    # inl: image and labels
    def __init__(self, image_transform=None, pre_transform=None):
        super(BrainSliceDataset, self).__init__()
        self.image_transform = image_transform
        self.pre_transform = pre_transform
        self.point_labels = self.get_point_labels()
        self.categories = self.get_category()
        #print(self.labels)

    def __getitem__(self, item):
        image = Image.open("/home/hdl2/Desktop/SonoDataset/Images/%d.jpg" % (item)).convert('L')
        point_label = self.point_labels[item]
        #image, labels = SizeCoorTransform((28, 28))(image, labels)
        #original_size = image.size
        #current_size = (28, 28)
        #image = image.resize(current_size)

        #image = np.asarray(image)[..., 0]
        #image = mu.copy_2D_to_3D(image, 3)
        #image = image.reshape(224, 224, 3)
        #mu.show_detail(image)
        if self.pre_transform is not None:
            image, point_label = self.pre_transform(image, point_label)
            point_label = [point_label[0][0], point_label[0][1], point_label[1][0], point_label[1][1]]
            point_label = torch.LongTensor(point_label)#.view(-1, 1)
            print("after pretranform: ", point_label)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, point_label #self.point_label[item][0]

    def __len__(self):
        # except .git
        return len(os.listdir("/home/hdl2/Desktop/SonoDataset/Images/")) - 1

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
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

train_loader = DataLoader(BrainSliceDataset(image_transform=transformer, pre_transform=SizeCoorTransform((28, 28))), batch_size=6)
test_loader = train_loader
assert 0

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout2d(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        #x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # 20 * 4 * 4 i.e. channels * kernel_w * kernel_h
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x)

        return x

model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)













def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.float()

        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        print(target.data, output.data)
        time.sleep(0.5)

        #loss = F.cross_entropy(output, target)
        loss = F.mse_loss(output, target)
        # got (int, torch.cuda.FloatTensor, !torch.cuda.LongTensor!, torch.cuda.FloatTensor, bool)
        # exp (int, torch.cuda.FloatTensor,  torch.cuda.FloatTensor, torch.cuda.FloatTensor, bool)
        #loss = F.nll_loss(output, target)


        loss.backward()
        optimizer.step()
        #print("epoch: %d %d/%d (%0.0f%%)\tLoss: %0.6f" % (epoch, epoch * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        target = target.float()
        output = model(data)
        #loss = F.mse_loss(output, target, size_average=False)
        test_loss += F.mse_loss(output, target, size_average=False).data[0]
        prediction = output.data.max(1, keepdim=True)[1]
        real = target.data.max(1, keepdim=True)[1]
        print(prediction)
        print(prediction.eq(real.view_as(prediction)).cpu().sum())

        correct += prediction.eq(real.view_as(prediction)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)\n" %(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


for epoch in range(0, 10):
    train(epoch)
    test(epoch)


