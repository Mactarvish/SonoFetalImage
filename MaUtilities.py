# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:01:45 2017

@author: ma

A utilities library for simple data process.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import signal
import cv2
from sklearn import metrics
import torch
import torchvision
import datetime
from torch.autograd import Variable
from colorama import Fore
from torchvision import transforms

image_path = './IU22Frame/%d.png'
save_path = "./IU22Result/%d.png"

def get_num_lines(file_name):
    context = None
    with open(file_name, "r") as f:
        context = f.readlines()
        return len(context)

def compose_filepath(path, filename):
    '''
    path最后可以有任意多个'/'，处理成只有1个之后与filename拼接。
    :param path: 
    :param filename: 
    :return: 
    '''
    if path.endswith('/'):
        index = len(path) - 1
        while path[index] == '/':
            index -= 1
            if index == -1:
                break
        index += 1
        # print(index)
        path = path[0: index + 1]
    else:
        path = path + '/'
    full_path = path + filename
    # print(full_path)

    return full_path

def RenameFile(original_dir, original_filename, new_filename):
    '''
    Rename a file in the original folder
    :param original_dir: 
    :param original_filename: 
    :param new_filename: 
    :return: 
    '''
    shutil.copyfile(original_dir + '/' + original_filename, original_dir + '/' + new_filename)
    os.remove(original_dir + '/' + original_filename)
    print("rename file %s to %s in %s" % (str(original_filename), str(new_filename), str(original_dir)))

def get_line(file_name, line_index):
    context = None
    with open(file_name, "r") as f:
        context = f.readlines()
        return context[line_index]


def replace_line(file_name, line_index, new_context):
    context = None
    with open(file_name, "r") as f:
        context = f.readlines()
        n_rows = len(context)
        context[line_index] = new_context + "\n"
    #print(context)
    with open(file_name, "w") as f:
        for line in context:
            f.write(line)


# amplify a image(array) using bilinear interpolation.
def BilinearInterpolation(feature_array, new_shape=(224, 224)):
    #image_array = np.float32(image_array)
    if len(feature_array.shape) == 3:
        return np.asarray([np.asarray(Image.fromarray(feature_array[:, :, i]).resize(new_shape, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)
    if len(feature_array.shape) == 4:
        return np.asarray([np.asarray(Image.fromarray(feature_array[0, :, :, i]).resize(new_shape, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)

'''
# add a Hann window to feature.
def hann2D(feature):
    assert feature.shape[:2] == (224, 224), "Hann2D : feature must be (224, 224)."
    window = signal.hann(224 * 224).reshape((224, 224))
    if len(feature.shape) == 3:
        window = copy_2D_to_3D(window, feature.shape[-1])
        return feature * window * window.transpose(1, 0, 2)
    if len(feature.shape) == 2:
        return feature * window * window.transpose()
'''

# add a Hann window to feature.
def hann2D(feature):
    width = feature.shape[0]
    height = feature.shape[1]
    window = np.dot(signal.hann(width).reshape((width, 1)), signal.hann(height).reshape((1, height)))
    if len(feature.shape) == 3:
        window = copy_2D_to_3D(window, feature.shape[-1])

    return feature * window

# Translate an image file to an numpy array.
def image_to_array(file_name):
    # RGB
    if type(file_name) == int:
        file_name = image_path % file_name
    image = Image.open(file_name)
    return np.asarray(image)[..., 0: 3]

# def save_rected_image(frame, positions):
#     plt.subplot(111)
#     path = image_path % frame
#     plt.imshow(image_to_array(path))
#     width = 20
#     colors = ['red', 'green', 'blue', 'yellow']
#     color_index = 0
#     for position in positions:
#         rect = plt.Rectangle((position[1] - width // 2, position[0] - width // 2), width, width,
#                              linewidth=1, alpha=1, facecolor='none', edgecolor=colors[color_index % len(colors)])
#         plt.subplot(111).add_patch(rect)
#         color_index += 1
#
#     plt.savefig(save_path % frame)
#     plt.close()

def save_rected_image(frame, positions):
    path = save_path % frame
    image = Image.open(path)
    painter = ImageDraw.Draw(image)
    width = 5
    colors = ['red', 'green', 'blue', 'yellow']
    color_index = 0
    for position in positions:
        painter.rectangle([(position[1] - width//2, position[0] - width//2), (position[1] + width//2, position[0] + width//2)], fill=colors[color_index % len(colors)], outline=colors[color_index % len(colors)])
        color_index += 1
    image.save(path)
    del painter

def create_gif(images, gif_name, duration=0.2):
    '''
    把images里的图像排成一个gif动态图，每帧的间隔是duration
    :param images: np组成的列表，例如[img1_np, img2_np, img3_np]
    :param gif_name: 
    :return: 
    '''
    import imageio
    frames = []
    for image_name in images:
        frames.append(image_name)
        # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def display(*inputs, delay_offs=None):
    '''
    显示图片
    :param inputs: 图片们
    :param delay_offs: None：不阻塞；number：延迟delay_offs（s）后显示下一张图或退出
    :return: 
    '''
    assert len(inputs) < 10, "number of inputs must be smaller than 10"
    # need some delay_offs, turn plt to interactive mode
    if delay_offs != None:
        plt.ion()
    if delay_offs == None:
        plt.ioff()
    # transfer string(file path), nparray, PILImage all to nparray
    image_arrays = []
    for e in inputs:
        if isinstance(e, str):
            image = Image.open(e)
            image = np.asarray(image)
            image_arrays.append(image)
        if isinstance(e, np.ndarray):
            image_arrays.append(e)
        if "PIL" in str(type(e)):
            image = np.asarray(e)
            image_arrays.append(image)
    plt.figure("MaUtilities Display")
    num_images = len(image_arrays)

    for i in range(1, num_images + 1):
        plt.subplot(100 + num_images * 10 + i)
        plt.imshow(image_arrays[i - 1])

    # Note: If not block in ioff, when the first image shown in ion and the second one shown in ioff, the second image
    # can NOT be shown (maybe the switch of interactive status causes that first image blocks the process.
    if delay_offs == None:
        plt.pause(1)
    else:
        plt.pause(delay_offs)
    # if not ion:
    plt.show()

def make_grid(*images_np, nrow=8, padding=2):
    '''
    把若干图片按栅格形状组合成一张图片，每张图的间隙为padding, 一行最多nrow张图
    example:
    r = make_grid(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], nrow=4, padding=50)
    :param images_np: cwh or whc（transform to cwh later)
    :param padding: 
    :return: 
    '''
    if type(images_np[0]) == list:
        images_np = images_np[0]
    image_size = images_np[0].shape
    for image in images_np:
        assert image_size == image.shape

    image_block = np.stack(images_np)
    # nwhc -> ncwh
    # POTENTIAL BUG: w and h CANNOT be 3
    if image_block.shape[-1] == 1 or image_block.shape[-1] == 3:
        image_block = image_block.transpose(0, 3, 1, 2)
    image_block = torch.from_numpy(image_block)
    return np.uint8(torchvision.utils.make_grid(image_block, nrow=nrow, padding=padding).numpy()).transpose(1, 2, 0)

# Resize an array to 'new_shape' using bilinear interpolation.
def resize_image(feature_array, new_shape=(224, 224)):
    '''
    :param feature_array: 
    :param new_shape: 
    :return: wh(2d) whc(3d) nwhc(4d)
    '''
    shape_PIL = (new_shape[1], new_shape[0])
    if len(feature_array.shape) == 2:
        return np.asarray(Image.fromarray(feature_array).resize(shape_PIL, Image.BILINEAR))
    if len(feature_array.shape) == 3:
        return np.asarray([np.asarray(Image.fromarray(feature_array[:, :, i]).resize(shape_PIL, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)
    if len(feature_array.shape) == 4:
        return np.asarray([np.asarray(Image.fromarray(feature_array[0, :, :, i]).resize(shape_PIL, Image.BILINEAR))
                        for i in range(feature_array.shape[-1])]).transpose(1, 2, 0)

    #return np.expand_dims(image, axis = 0)

# Return the indice of the greatest element in a tensor.
def get_argmax_pos(a):
    return np.int64(np.transpose(np.where(a == np.max(a)))[0])

# Return an Gaussian matrix of 'shape' whose center is 'center'(x, y).
def Gaussian_2D(center, shape, sigma = 0.1):
    #signal.gaussian()
    confidence_score = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            confidence_score[i, j] = (i - center[0])**2 + (j - center[1])**2
    confidence_score = confidence_score * -1. / (2 * sigma**2)
    confidence_score = np.exp(confidence_score)
    return confidence_score

# Judge equality of two same-shape array ignoring tiny difference.
def is_equal(a, b, error=0.0001):
    '''
    比较a和b是否几乎完全相同，可以是Variable或者nparray
    :param a: 
    :param b: 
    :param error: 
    :return: 
    '''
    if isinstance(a, Variable):
        a = a.cpu().data.numpy()
    if isinstance(b, Variable):
        b = b.cpu().data.numpy()
    assert a.shape == b.shape, "Two arrays must be the same shape."
    return np.sum(abs(a - b)) < error


# Return a cropped array whose shape is 'shape' & given center. Support string of file name & array.
def crop_image(image, center, shape=(224, 224), mode='gray'):
    if type(image) == int:
        image = image_path % image
    if type(image) == str:
        image = image_to_array(image)
    cropped_image = None
    if type(shape) == int:
        shape = (shape, shape)
    assert image.shape[0] >= shape[0] and image.shape[1] >= shape[1], "Uncorrect crop shape"
    rectified_x = max(center[0], shape[0] // 2)
    rectified_x = min(rectified_x, image.shape[0] - shape[0] // 2)
    rectified_y = max(center[1], shape[1] // 2)
    rectified_y = min(rectified_y, image.shape[1] - shape[1] // 2)
    rectified_center = (rectified_x, rectified_y)
    if len(image.shape) == 3:
        cropped_image = image[rectified_center[0]-shape[0]//2 : rectified_center[0]+shape[0]//2, rectified_center[1]-shape[1]//2 : rectified_center[1]+shape[1]//2, :]
        if mode == 'gray':
            cropped_image = copy_2D_to_3D(cropped_image[..., 0], 3)
    if len(image.shape) == 2:
        cropped_image = image[rectified_center[0]-shape[0]//2 : rectified_center[0]+shape[0]//2, rectified_center[1]-shape[1]//2 : rectified_center[1]+shape[1]//2]

    return cropped_image

# Given a shape, return its center coordinate(tuple).
def get_shape_center(shape):
    return (shape[0] // 2, shape[1] // 2)

# For a cube-shape tensor, compression to a 2D array by summerizing each layer.
def summerize_each_layer(cube):
    assert len(cube.shape) == 3, "Cube must be 3dim."
    return np.sum(cube, axis = 2)

# Return a 3D array copyed from a 2D array i.e. Pile up num_layers 2D arrays.
def copy_2D_to_3D(array_2D, num_layers):
    return np.tile(array_2D, (num_layers, 1, 1)).transpose((1, 2, 0))

def rgb2gray(rgb_np):
    '''
    输入一张3通道的rgb图像（numpy）格式(whc)，返回对应的numpy格式的单通道灰度图
    :param rgb: 
    :return: 
    '''
    return np.dot(rgb_np[...,:3], [0.2989, 0.5870, 0.1140])

def check_if_cfirst(image_np):
    '''
    Check if image_np is cfirst (cwh or ncwh)
    :param image_np: 
    :return: 
    '''
    assert len(image_np.shape) == 4 or len(image_np.shape) == 3
    if (image_np.shape[-1] == 1 or image_np.shape[-1] == 3):
        return False
    else:
        return True

def np_channels2c_last(image_np):
    '''
    transform ncwh to nwhc or cwh to whc
    POTENTIAL BUG: w and h CANNOT be 3
    :param image_np: 
    :return: 
    '''
    assert len(image_np.shape) == 4 or len(image_np.shape) == 3
    # if already c last, return image_np
    if not check_if_cfirst(image_np):
        return image_np
    if len(image_np.shape) == 4:
        # ncwh -> nwhc
        image_np = image_np.transpose(0, 2, 3, 1)
    else:
        # cwh -> whc
        image_np = image_np.transpose(1, 2, 0)
    return image_np

def np_channels2c_first(image_np):
    '''
    transform nwhc to ncwh or whc to cwh
    POTENTIAL BUG: w and h CANNOT be 3
    :param image_np: 
    :return: 
    '''
    assert len(image_np.shape) == 4 or len(image_np.shape) == 3
    # if already c first, return image_np
    if check_if_cfirst(image_np):
        return image_np
    if len(image_np.shape) == 4:
        # nwhc -> ncwh
        image_np = image_np.transpose(0, 3, 1, 2)
    else:
        # whc -> cwh
        image_np = image_np.transpose(2, 0, 1)
    return image_np

def stack3ctorgb(channels):
    '''
    把三张单通道图像堆叠在一起，形成一张rgb图像
    :param channels: 含有三张单通道图像的list， 例如 [img[..., 0], img[..., 1], img[..., 2]]
    :return: 
    '''
    return np.stack(channels).transpose((1, 2, 0))

# Print detail info of the given np_array.
def show_detail(np_array, comment=None):
    print(comment, "shape:", np_array.shape, "dtype:", np_array.dtype, "max:", np.max(np_array), "min:", np.min(np_array))

def get_test_image(image_type='np'):
    '''
    
    :param image_type: 
    :return: whc
    '''
    image = Image.open('lena_std.tif')
    if image_type == 'PIL':
        return image
    if image_type == 'np' or image_type == 'numpy':
        return np.asarray(image)

def ConfusionMatrixPng(cm, classlist, title):
    '''
    confusion_matrix_vision.ConfusionMatrixPng([temp['confusion_matrix'], temp['confusion_matrix']], title=['a', 'b'],
                                               classlist=['Soap', 'Price', 'Ghost', 'Nikolai', 'Roach', 'Ozone'])
    :param cm: [nparray1, nparray2, ...]
    :param classlist: ['Soap', 'Price', 'Ghost', 'Nikolai', 'Roach', 'Ozone']
    :param title: ['a', 'b']
    :return: 
    '''
    def matrix_transfer(matrix):
        '''
        归一化，把混淆矩阵的每一个值都整到0~1之间
        :param matrix: 
        :return: 
        '''
        norm_conf = []
        for i in matrix:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)
        return norm_conf

    for i in range(len(cm)):
        cm[i] = matrix_transfer(cm[i])

    fig = plt.figure(figsize=(10, 5))
    # fig.set_size_inches(30, 30)
    plt.clf()
    num_metrics = len(cm)
    assert num_metrics < 10
    assert len(cm) == len(title)
    BASE_NUM = 100 + num_metrics * 10
    for i in range(len(cm)):
        ax = fig.add_subplot(BASE_NUM + i + 1)
        ax.set_aspect("equal")
        ax.set_title(title[i])
        plt.yticks(fontsize=5)
        plt.xticks(fontsize=5)
        # classlist = ['', '', '', '', '']
        plt.yticks(range(len(classlist)), classlist)
        plt.xticks(range(len(classlist)), classlist, rotation=-90)
        res = ax.imshow(np.array(cm[i]), cmap=plt.cm.plasma,
                        interpolation='nearest')
        cb = fig.colorbar(res, shrink=0.3)
        if i != len(cm)-1:
            cb.remove()
    plt.savefig('confusion_matrix.png', dpi=500)
    plt.show()

def VisdomDrawLines(*lines, legends=None, title=None):
    #legends必须显式给定，即这样调用：mu.VisdomDrawLines(train_acc, test_acc, legends=['train', 'test'])
    '''
    
    :param lines: np.array or list e.g: [1,2,3,4,5] or np.asarray([1,2,3,4,5])
    :param legends: 
    :param title: 
    :return: 
    '''
    point_count = 0
    for i, line in enumerate(lines):
        if i == 0:
            point_count = len(line)
        assert (len(line) == point_count), "all lines must have the same number of points"

    Y = np.column_stack(lines)
    if title == None:
        title = datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S')
    if legends == None:
        opts = dict(title=title)
    else:
        opts = dict(legend=legends, title=title)
    from visdom import Visdom
    viz = Visdom()
    viz.line(
        Y=Y,
        X=np.linspace(0, Y.shape[0]-1, Y.shape[0]),
        opts=opts,
    )

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

def image2modelinput(file_name, model_input_size=None):
    image = Image.open(file_name)#.convert('L')
    image = ResizeImage(model_input_size)(image)
    #input = Variable(transforms.ToTensor()(image).cuda())
    #torch.squeeze()
    input = Variable(torch.unsqueeze(transforms.ToTensor()(image), dim=0).cuda())
    #print(input)
    return input

def get_freer_gpu():
    '''
    # TODO
    os.system('nvidia-smi -q  >tmp')
        Utilization
        Gpu                         : 0 %
    :return: 
    '''
    import os
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

def log_metrics(train, y_true, y_pred, loss, model, save_path, note=None, save=True, show_detail=True):
    net_name = model.__class__.__name__
    if train:
        print('training loss:', loss)
        dic = {'loss': loss}
    else:
        classify_report    = metrics.classification_report(y_true, y_pred, digits=6)
        confusion_matrix   = metrics.confusion_matrix(y_true, y_pred)
        overall_accuracy   = metrics.accuracy_score(y_true, y_pred)
        top1_error_rate    = 1 - overall_accuracy
        precision_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
        average_precision   = np.mean(precision_for_each_class)
        f1_score = metrics.f1_score(y_true, y_pred, average='micro')

        if show_detail:
            print('classify_report : \n', classify_report)
            print('confusion_matrix : \n', confusion_matrix)
            print('precision_for_each_class : \n', precision_for_each_class)
        print('test loss:', loss)
        print('average_precision: {0:f}'.format(average_precision))
        print(Fore.RED, 'overall_accuracy: {0:f}'.format(overall_accuracy), Fore.BLACK)
        print('f1-score: {0:f}'.format(f1_score))
        print('top-1 error rate: {0:f}'.format(top1_error_rate))
        print()
        dic = {'net_name': net_name, 'classify_report': classify_report, 'confusion_matrix': confusion_matrix, 'precision_for_each_class': precision_for_each_class,
         'average_precision': average_precision, 'overall_accuracy': overall_accuracy, 'f1_score': f1_score, 'loss': loss, 'top1_error_rate': top1_error_rate}
    # 保存指标
    if save:
        train_str = 'train' if train else 'test'
        try:
            if note == None:
                log = torch.load('%s/%s' % (save_path, train_str))
            else:
                log = torch.load('%s/%s_%s' % (save_path, note, train_str))
        except:
            log = []
        finally:
            # 保存指标
            log.append(dic)
            if note == None:
                torch.save(log, '%s/%s' % (save_path, train_str))
            else:
                torch.save(log, '%s/%s_%s' % (save_path, note, train_str))
            # 保存模型
            if not train:
                max_f1_score = 0
                for d in log:
                    f1_score = d['f1_score']
                    if f1_score > max_f1_score:
                        max_f1_score = f1_score
                if f1_score >= max_f1_score:
                    if note == None:
                        torch.save(model, '%s/%s.pkl' % (save_path, net_name))
                    else:
                        torch.save(model, '%s/%s_%s.pkl' % (save_path, note, net_name))


################################################################ pytorch transformer ################################################################
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
        assert 0, "transformer must has only one parameter in __call__()"
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
