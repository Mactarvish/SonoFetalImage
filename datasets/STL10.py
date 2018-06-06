from torchvision.datasets.stl10 import STL10

class MySTL10(object):
    def __init__(self, mode, dataset_size=1, image_transform=None, target_transform=None):
        assert mode in ['train', 'test'], 'but got mode {}'.format(mode)
        assert 0 < dataset_size <= 1
        if dataset_size != 1:
            raise NotImplementedError
        self.mode = mode

        self.train_set = STL10(root='datas', split='train', transform=image_transform, target_transform=target_transform, download=True)
        self.test_set  = STL10(root='datas', split='test', transform=image_transform, target_transform=target_transform, download=True)
        self.dataset_size = dataset_size

        print('Loading %s set, %d samples.' % (self.mode, self.__len__()))

    def __len__(self):
        if self.mode == 'train':
            return 10400
        else:
            return 2600

    def __getitem__(self, index):
        if self.mode == 'train':
            if index < 5000:
                return self.train_set[index]
            else:
                return self.test_set[index - 5000]
        else:
            return self.test_set[5400 + index]

