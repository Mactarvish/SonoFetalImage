import torchvision.datasets as dset
import MaUtilities as mu
from torchvision import transforms

class Cifar10(dset.CIFAR10):
    def __init__(self, train, dataset_size=1, binclassify=None, image_transform=None, target_transform=None,):
        assert 0 < dataset_size <= 1
        super(Cifar10, self).__init__(root='/home/hdl2/Desktop/SonoFetalImage/datas', train=train, transform=image_transform, target_transform=target_transform, download=True)
        self.dataset_size = dataset_size
        assert binclassify is None or (isinstance(binclassify, int) and 0 <= binclassify < 10)
        self.binclassify = binclassify

    def __len__(self):
        return int(super(Cifar10, self).__len__() * self.dataset_size)

    def __getitem__(self, index):
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
                         std=[0.2675, 0.2565, 0.2761])]
)