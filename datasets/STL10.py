from torchvision.datasets.stl10 import STL10
import MaUtilities as mu
from torchvision import transforms


import MaUtilities as mu
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import sqlite3

use_gpu = torch.cuda.is_available()
image_location = "/home/hdl2/Desktop/STL10/"
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

connection = sqlite3.connect("STL10")
cu = connection.cursor()

class MySTL10(Dataset):
    def __init__(self, train=True, image_transform=None, pre_transform=None):
        super(MySTL10, self).__init__()
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
        print('Database loaded. %d training images, %d validation images, loaded %s set.' % (self.num_train, self.num_test, loaded_set))

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
        image_path = mu.cat_filepath(image_location, category, image_name)
        image = Image.open(image_path)#.convert('L')
        label = label_map[category]

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        if self.train == True:
            return self.num_train
        else:
            return self.num_test





# class MySTL10(object):
#     def __init__(self, mode, dataset_size=1, image_transform=None, target_transform=None):
#         assert mode in ['train', 'test'], 'but got mode {}'.format(mode)
#         assert 0 < dataset_size <= 1
#         if dataset_size != 1:
#             raise NotImplementedError
#         self.mode = mode
#
#         self.train_set = STL10(root='datas', split='train', transform=image_transform, target_transform=target_transform, download=True)
#         self.test_set  = STL10(root='datas', split='test', transform=image_transform, target_transform=target_transform, download=True)
#         self.dataset_size = dataset_size
#
#         print('Loading %s set, %d samples.' % (self.mode, self.__len__()))
#
#     def __len__(self):
#         if self.mode == 'train':
#             return 10400
#         else:
#             return 2600
#
#     def __getitem__(self, index):
#         if self.mode == 'train':
#             if index < 5000:
#                 return self.train_set[index]
#             else:
#                 return self.test_set[index - 5000]
#         else:
#             return self.test_set[5400 + index]

transformer = transforms.Compose([
    mu.ResizeImage((255, 255)),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.44783968, 0.44069089, 0.40721496],
                          std=[0.26026609, 0.25636546, 0.27067264])
])
