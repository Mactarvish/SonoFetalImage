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
from visdom import Visdom

viz = Visdom()
assert viz.check_connection()


torch.manual_seed(27)
torch.cuda.manual_seed(67)

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

class CoordinateTuple2Tensor():
    def __call__(self, point_label):
        output = []
        for coordinate in point_label:
            output.append(coordinate[0])
            output.append(coordinate[1])
        output = torch.LongTensor(output)

        return output

class BrainSliceDataset(Dataset):
    # inl: image and labels
    def __init__(self, train=True, index_sequence_mode=True, pre_transform=None, image_transform=None, label_transform=None):
        super(BrainSliceDataset, self).__init__()
        self.train = train
        self.index_sequence_mode = index_sequence_mode
        self.pre_transform = pre_transform
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.sequence_index_map, self.point_labels = self.get_point_labels(sequence=True)
        self.categories = self.get_category()
        self.category_transofm = {"circle": 0, "line": 1, "other": 1}
        #print(self.sequence_index_map)

    def __getitem__(self, seqence_index):
        # map list index(sequence index) to filename index
        filename_index = self.sequence_index_map[seqence_index]
        image = Image.open("/home/hdl2/Desktop/SonoDataset/OriginalImages/%d.jpg" % (filename_index)).convert('L')
        point_label = self.point_labels[filename_index]
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
            # point_label = [point_label[0][0], point_label[0][1], point_label[1][0], point_label[1][1]]
            # point_label = torch.LongTensor(point_label)#.view(-1, 1)
            #print("after pretranform: ", point_label)
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.label_transform is not None:
            point_label = self.label_transform(point_label)

        #print(point_label[:2])
        return image, point_label#[:2] #self.point_label[item][0]

    def __len__(self):
        # except .git
        if self.train == True:
            return 150
        else:
            return 40
        #return len(os.listdir("/home/hdl2/Desktop/SonoDataset/Images/")) - 1

    def str2coordinate(self, str):
        '''
        :param str: like '(231,55)'
        :return: str to tuple
        :example: '(231,55)' -> tuple (231, 55)
        '''
        nums = str[1:-1].split(',')
        return (int(nums[0]), int(nums[1]))
    # For the case the labeled image names are not continuous, define a dict to map continuous index (0,1,2...) to discret index (0,3,4,6,7,9...)
    def get_point_labels(self, sequence):
        f = open("/home/hdl2/Desktop/SonoDataset/Labels/points.txt")
        filename_labels = {}
        sequence_index_map = {}
        lines = f.readlines()
        #print(lines)

        for line in lines:
            #print(line)
            # print(line[-1] == '\n')
            if line[-1] == '\n':
                line = line[:-1]
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
                        print('happens')
                        continue
                    coordinates.append(self.str2coordinate(e))
            filename_labels[index] = coordinates
            sequence_index_map[len(sequence_index_map)] = index
        return sequence_index_map, filename_labels

    def get_category(self):
        f = open("/home/hdl2/Desktop/SonoDataset/Labels/classify.txt")
        labels = {}
        lines = f.readlines()
        for line in lines:
            #print(line)
            number = int(line.split('.')[0])
            category = line.split(' ')[-1][:-1]
            labels[number] = category
        #print(labels)
        return labels


transformer = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

train_loader = DataLoader(BrainSliceDataset(train=True, pre_transform=None, image_transform=transformer, label_transform=CoordinateTuple2Tensor()), shuffle=True, batch_size=6)
test_loader = DataLoader(BrainSliceDataset(train=False, pre_transform=None, image_transform=transformer, label_transform=CoordinateTuple2Tensor()), batch_size=6)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(23040, 50)
        self.fc2 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 10 * 110 * 110
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 20 * 52 * 52
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2)) # 40 * 24 * 24
        #print("fefefefefefefefefefefefe", x.view(6, -1).data.shape)
        x = x.view(-1, 23040)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x)
model = Net()
model.cuda()

conv_model = torch.nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=7),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(20, 40, kernel_size=5),
    nn.Dropout2d(),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(40, 40, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU()
)

class RemakeNet(nn.Module):
    def __init__(self):
        super(RemakeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4000, 50)
        self.fc2 = nn.Linear(50, 8)
        self.conv_model = conv_model

    def forward(self, x):
        x = self.conv_model(x)
        x = x.view(-1, 4000)
        print("xxxxxxxxxxxxxxxx", x.data.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x)

model = Net()
model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.7)
def adjust_lr(optimizer, lr=0.0001):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

losses = []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda()).float()
        # print("target:", target)
        # print('data:', data)
        optimizer.zero_grad()
        output = model(data)
        # print("output:", output)
        loss = F.mse_loss(output, target)
        loss.backward()
        # if epoch > 0:
        #     adjust_lr(optimizer, 0.00005)
        # if epoch >= 50:
        #     adjust_lr(optimizer, 0.9)

        optimizer.step()
        #print("Loss: %0.6f" % (loss.data[0]))
        torch.save(model.state_dict(), 'RegressionWeights.pkl')
        #losses.append(loss.data[0])
    # viz.line(Y=np.asarray(losses))

good_prediction_count = 0

def validation(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # print(data.shape)
        # for i in range(data.shape[0]):
        #     print(data[i, 0, ...])
        #     Image.fromarray(data[i, 0, ...].numpy())
        data, target = Variable(data.cuda()), Variable(target.cuda()).float()
        output = model(data)
        #print(target.data.long().cpu().numpy())
        print(output.data.long().cpu().numpy())
        print(F.mse_loss(output, target).data[0])
        #print(data.data.long().cpu().numpy())
        #F.mse_loss()
        test_loss += F.mse_loss(output, target).data[0]
    #print("\nTest set: Loss: %.4f\n" %(test_loss))
    #test_loss /= len(test_loader.dataset)
    losses.append(test_loss)

def test(n_image, ion=True):
    model.load_state_dict(torch.load('RegressionWeights.pkl'))
    model.eval()
    file_name = "/home/hdl2/Desktop/SonoDataset/Circles/%d.jpg" % (n_image)
    data = image2modelinput(file_name)
    output = model(data)
    prediction = output.data.long().cpu().numpy()[0]
    coordinates = []
    for i in range(0, len(prediction), 2):
        if prediction[i] <= 0:
            prediction[i] = 0
        if prediction[i+1] <= 0:
            prediction[i+1] = 0
        coordinates.append((prediction[i], prediction[i+1]))
    #print(coordinates)
    image_np = np.asarray(Image.open(file_name))
    for i in range(4):
        cv2.circle(image_np, coordinates[i], 5, (255, 0, 0))

    mu.display(image_np, ion=ion)

d = BrainSliceDataset(train=True, pre_transform=None, image_transform=transformer, label_transform=CoordinateTuple2Tensor())

for epoch in range(5):
    train(epoch)
    validation(epoch)
    print(epoch)
viz.line(Y=np.asarray(losses), X=np.arange(len(losses)))
torch.save(model.state_dict(), 'RegressionWeights.pkl')
assert 0
print("Done")

for i in range(len(d.sequence_index_map)):
    test(d.sequence_index_map[i], ion=True)
#    87    72   155    76    86   140   154   141
# [(61, 51), (109, 55), (59, 97), (109, 97)]