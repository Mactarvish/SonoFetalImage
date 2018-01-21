import numpy as np
import cv2
from PIL import Image
import os
import MaUtilities as mu

def CalculateDatasetMeanAndStd(dir, new_size):
    image_names = os.listdir(dir)
    channels1 = []
    channels2 = []
    channels3 = []
    count = 0
    for image_name in image_names:
        # load image and preprocess it
        image = Image.open(dir + '/' + image_name)
        #mu.display(image, ion=False)
        image = cv2.resize(np.asarray(image), new_size)
        if image.shape != (255, 255, 3):
            assert 0, (image_name, image.shape)
        #mu.display(image, ion=False)
        # check if image is in (0, 1)
        if np.max(image) > 1:
            image = image / 255
        #mu.display(image, ion=False)

        print(image[..., 0].shape)
        channels1.append(image[..., 0])
        channels2.append(image[..., 1])
        channels3.append(image[..., 2])
        #print(image[..., 0])

        count += 1
        print(count)
    for channel in channels1:
        print(channel.shape)
    np.stack(channels1, axis=2)
    np.stack(channels2, axis=2)
    np.stack(channels3, axis=2)
    cubes = [np.stack(channels1, axis=2), np.stack(channels2, axis=2), np.stack(channels3, axis=2)]

    means = []
    stds = []
    for cube in cubes:
        means.append(np.mean(cube))
        stds.append(np.std(cube))
    print('=' * 10)
    print(cubes)
    return means, stds

print(CalculateDatasetMeanAndStd("ThyImage", (255, 255)))

