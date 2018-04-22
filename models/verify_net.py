import cv2
import torch
import numpy as np
from PIL import Image
import MaUtilities as mu
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.autograd import Variable

class VerifyNet(torch.nn.Module):
    '''
    输入是图像的形状，例如单通道图片就是(255, 255)，三通道图片就是(3, 255, 255)
    '''
    def __init__(self, input_shape, num_classes):
        super(VerifyNet, self).__init__()
        volume = 1
        for l in input_shape:
            volume *= l
        self.linear = torch.nn.Linear(volume, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.relu(x)
        return x