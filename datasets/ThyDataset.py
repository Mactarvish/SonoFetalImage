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
image_location = "/home/hdl2/Desktop/MedicalImage/"
label_map = {"hashimoto_thyroiditis1": 0, "hyperthyreosis1": 1, "normal1": 2, "postoperative1": 3, "subacute_thyroiditis1": 4, "subhyperthyreosis1": 5}
# 去掉术后和亚甲亢
#label_map = {"hashimoto_thyroiditis1": 0, "hyperthyreosis1": 1, "normal1": 2, "subacute_thyroiditis1": 3}

connection = sqlite3.connect("ThyDataset")
# connection = sqlite3.connect("ThyDataset_Small")
cu = connection.cursor()

class ThyDataset(Dataset):
    def __init__(self, train=True, image_transform=None, pre_transform=None):
        super(ThyDataset, self).__init__()
        self.train = train
        self.image_transform = image_transform
        self.pre_transform = pre_transform

        # For unknown reason(maybe memory problem?), loading datas from sqlite will be crashed after thousands of loading,
        # so we directly save datas to memory first avoiding loading from sqlite repeatly.
        # Note that the dict starts from 1, NOT 0.
        self.train_set = {}
        self.val_set = {}

        # load training datas
        cu.execute("select * from Train")
        records = cu.fetchall()
        self.num_train = len(records)
        for record in records:
            self.train_set[record[0]] = record

        # load validation datas
        cu.execute("select * from Validation")
        records = cu.fetchall()
        self.num_test = len(records)

        for record in records:
            self.val_set[record[0]] = record

        loaded_set = None
        if self.train:
            loaded_set = 'training'
        else:
            loaded_set = 'validation'
        print('Database loaded. %d training images, %d validation images, loaded %s set.\n' % (self.num_train, self.num_test, loaded_set))

    def __getitem__(self, item):
        record = None
        if self.train == True:
            # query = "select * from Train where id=%d" % (item + 1)
            # print("select * from Train where id=%d" % (item + 1))
            # cu.execute("select * from Train where id=%d" % (item + 1))

            record = self.train_set[item + 1]

        else:
            # query = "select * from Validation where id=%d" % (item + 1)
            # print("select * from Validation where id=%d" % (item + 1))
            # cu.execute("select * from Validation where id=%d" % (item + 1))

            record = self.val_set[item + 1]

        # record = cu.fetchall()[0]
        image_name = record[1]
        category = record[2]
        #print(image_name, category)
        image = Image.open(image_location + category + '/' + image_name)#.convert('L')
        label = label_map[category]

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        if self.train == True:
            return self.num_train
        else:
            return self.num_test

