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

image_path = './IU22Frame/%d.png'
save_path = "./IU22Result/%d.png"

def get_num_lines(file_name):
    context = None
    with open(file_name, "r") as f:
        context = f.readlines()
        return len(context)

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

def display(*inputs, ion=True):
    assert len(inputs) < 10, "number of inputs must be smaller than 10"

    if ion:
        plt.ion()
    else:
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
        plt.pause(1)
    if not ion:
        plt.show()

# Resize an array to 'new_shape' using bilinear interpolation.
def resize_image(feature_array, new_shape=(224, 224)):
    shape_PIL = (new_shape[1], new_shape[0])
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

# Print detail info of the given np_array.
def show_detail(np_array, comment=None):
    print(comment, "shape:", np_array.shape, "dtype:", np_array.dtype, "max:", np.max(np_array), "min:", np.min(np_array))

def VisdomDrawLines(*lines, legends=None):
    #legends must be assigned by legends=... explictly.
    point_count = 0
    for i, line in enumerate(lines):
        if i == 0:
            point_count = len(line)
        assert (len(line) == point_count), "all lines must have the same number of points"

    Y = np.column_stack(lines)
    if legends == None:
        opts = None
    else:
        opts = dict(legend=legends)
    print(legends)
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
    image = mu.ResizeImage(model_input_size)(image)
    #input = Variable(transforms.ToTensor()(image).cuda())
    #torch.squeeze()
    input = Variable(torch.unsqueeze(transforms.ToTensor()(image), dim=0).cuda())
    #print(input)
    return input

def save_matrics(y_true, y_pred, losses, net_name):
    classify_report    = metrics.classification_report(y_true, y_pred)
    confusion_matrix   = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy   = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy   = np.mean(acc_for_each_class)
    score = metrics.accuracy_score(y_true, y_pred)

    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))
    dic = {'net_name': net_name, 'classify_report': classify_report, 'confusion_matrix': confusion_matrix, 'acc_for_each_class': acc_for_each_class,
     'average_accuracy': average_accuracy, 'overall_accuracy': overall_accuracy, 'score': score, 'losses': losses}
    torch.save(dic, 'matrics/%s' % (net_name))

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
