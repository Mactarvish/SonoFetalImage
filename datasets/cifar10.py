import torchvision.datasets as dset
import MaUtilities as mu
from torchvision import transforms

VALIDATION_SET_SIZE = 5000


class Cifar10(dset.CIFAR10):
    '''
    把train set劈成两部分，45000 和 5000，5000是验证集
    '''
    def __init__(self, mode, dataset_size=1, binclassify=None, image_transform=None, target_transform=None,):
        assert mode in ['train', 'test', 'validation'], 'but got mode {}'.format(mode)
        assert 0 < dataset_size <= 1
        assert binclassify is None or (isinstance(binclassify, int) and 0 <= binclassify < 10)
        self.mode = mode
        super(Cifar10, self).__init__(root='/home/hdl2/Desktop/SonoFetalImage/datas', train=(self.mode is not 'test'), transform=image_transform, target_transform=target_transform, download=True)
        self.dataset_size = dataset_size
        self.binclassify = binclassify

        print('Loading %s set, %d samples.' % (self.mode, self.__len__()))

    def __len__(self):
        total_size = None
        if self.mode == 'test':
            total_size = super(Cifar10, self).__len__()
        elif self.mode == 'train':
            total_size = super(Cifar10, self).__len__() - VALIDATION_SET_SIZE
        else:
            total_size = VALIDATION_SET_SIZE

        return int(total_size * self.dataset_size)


    def __getitem__(self, index):
        if self.mode == 'validation':
            index = index + 50000 - VALIDATION_SET_SIZE
        img, target = super(Cifar10, self).__getitem__(index)
        if self.binclassify is not None:
            target = 1 if (target == self.binclassify) else 0
        return img, target

transformer = transforms.Compose([
    mu.ResizeImage((255, 255)),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),

])