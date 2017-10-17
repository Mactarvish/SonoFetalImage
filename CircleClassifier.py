import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import MaUtilities as mu

root = "/home/hdl2/Desktop/Sono_nofolder/"

class CropCore(object):
    def __init__(self, output_size=None):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]
        image = image[115:712, 279:1094, :]

        return {"image": image}

class TransferToGray(object):
    def __init__(self):
        super(TransferToGray, self).__init__()

    def __call__(self, sample):
        image = sample["image"]
        image = mu.copy_2D_to_3D(image[..., 0], 3)

        return {"image": image}

composed = transforms.Compose([CropCore(), TransferToGray()])

class FetalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, img_name):
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)
        type = "Unset"
        sample = {'image': image, 'type': type}

        if self.transform:
            sample = self.transform(sample)

        return sample

fetal_dataset = FetalImageDataset(root_dir=root, transform=composed)


#print(type(fetal_dataset['158_HAOTINGTING_1_Labeled.jpg']['image']))

# 20160919_112159_9_Labeled_Circle.jpg The last one I marked the circle.
a = fetal_dataset["158_HAOTINGTING_0_No_Circle.jpg"]
print(type(a))
mu.display(fetal_dataset["158_HAOTINGTING_0_No_Circle.jpg"]["image"])