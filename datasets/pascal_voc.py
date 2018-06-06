import torchvision.datasets as dset
import MaUtilities as mu
from torchvision import transforms
from colorama import Fore

VALIDATION_SET_SIZE = 5000

import os, numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import xml.etree.ElementTree as ET


class PascalVOC(data.Dataset):
    def __init__(self, mode, image_transform=None, root='/home/hdl2/Desktop/VOCdevkit/VOC2012'):
        print(Fore.RED, 'Split train set to train, validation and test', Fore.BLACK)
        print(Fore.RED, 'UNVERIFIED', Fore.BLACK)
        assert mode in ['train', 'test'], 'but got mode {}'.format(mode) # , 'validation'
        self.mode = mode
        self.trainval = 'trainval'
        self.root = root
        self.image_transform = image_transform
        self.__init_classes()
        self.names, self.labels = self.__dataset_info()

        print('Loading %s set, %d samples.' % (self.mode, self.__len__()))

    def __getitem__(self, index):
        if self.mode == 'test':
            index = index + 9232
        image = Image.open(self.root + '/JPEGImages/' + self.names[index] + '.jpg')
        if self.image_transform is not None:
            image = self.image_transform(image)

        label = self.labels[index]
        return image, label

    def __len__(self):
        if self.mode ==  'train':
            return 9232
        else:
            return 2308
        # return len(self.names)

    def __dataset_info(self):
        # annotation_files = os.listdir(self.root+'/Annotations')
        with open(self.root + '/ImageSets/Main/' + self.trainval + '.txt') as f:
            annotations = f.readlines()

        annotations = [n[:-1] for n in annotations]

        names = []
        labels = []
        for af in annotations:
            # if len(af) != 6:
            #     continue
            filename = os.path.join(self.root, 'Annotations', af)
            tree = ET.parse(filename + '.xml')
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            boxes_cl = np.zeros((num_objs), dtype=np.int32)

            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                boxes_cl[ix] = cls

            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)

        return np.array(names), np.array(labels).astype(np.float32)

    def __init_classes(self):
        self.classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

'''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])
                                     '''