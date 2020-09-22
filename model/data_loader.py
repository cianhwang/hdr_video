from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import numbers
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import random
import h5py

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class GroupToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, *images):
        return [TF.to_pil_image(img, self.mode) for img in images]

class GroupToTensor(object):
    def __call__(self, *images):
        return [TF.to_tensor(img) for img in images]

class GroupRandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, *images):
        randnum = random.random()
        return [TF.hflip(img) if randnum < self.p else img for img in images]

class GroupRandomVerticalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, *images):
        randnum = random.random()
        return [TF.vflip(img) if randnum < self.p else img for img in images]

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *images):
        return [TF.resize(img, self.size, self.interpolation) for img in images]

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, *images):

        i, j, h, w = self.get_params(images[0], self.size)
        return [TF.crop(img, i, j, h, w) for img in images]
        
class GroupComposed(object):
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *images):
        for t in self.transforms:
            images = t(*images)
        return images

#train_transformer = transforms.Compose([
#    transforms.ToPILImage() # can convert numpy array (https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToPILImage)
#    transforms.RandomCrop(1024)
#    transforms.ToTensor() #[0, 255] -> [0, 1]
#])
train_transformer = GroupComposed([
    GroupToPILImage(mode = 'L'),
    GroupRandomCrop(512), ## brake Bayer pattern!!
    GroupRandomHorizontalFlip(),
    GroupRandomVerticalFlip(),
    GroupToTensor()
])

eval_transformer =  GroupComposed([
    GroupToPILImage(mode = 'L'),
    GroupRandomCrop(512), ## brake Bayer pattern!!
    GroupToTensor()
])

def raw_normalize(*raw_imgs): 
    ## may include quantization error?
    return [((img - img.min())*255.0/(img.max()-img.min())).astype(np.uint8) for img in raw_imgs]

class HDRDataset(Dataset):

    def __init__(self, transform = None, length = 1000):
        self.transform = transform
        self.inputs = (np.load('data/inputs_2.npy')[:8]).transpose(1, 2, 0) # 8x2048x3584
        self.gt = np.load('data/gt.npy')[np.newaxis].transpose(1, 2, 0)  # 1x2048x3584
        #self.inputs = raw_normalize(*(np.split(self.inputs, self.inputs.shape[-1], axis=-1)), self.gt)
        #self.inputs, self.gt = self.frames[:-1], self.frames[-1]  ## frames: list of N numpy array of size HxWx1
                                                                    ## gt: numpy array of size HxWx1
        self.length = length
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #if self.transform is not None:
        frames = raw_normalize(*(np.split(self.inputs, self.inputs.shape[-1], axis=-1)), self.gt)
        frames = self.transform(*frames) ## 8x512x512
        img = torch.cat(frames[:-1], dim=0) ## 1x512x512
        gt = frames[-1]
        return img, gt

def fetch_dataloader(params = None, types = 'train'):

    dataloaders = {}

    for split in ['train', 'val','test']:
        if split == 'train':
            dl = DataLoader(HDRDataset(train_transformer),
                            batch_size = params.batch_size, shuffle = True,
                            num_workers = int(params.num_workers),
                            pin_memory=params.cuda)
        else:
            dl = DataLoader(HDRDataset(eval_transformer, 100),
                            batch_size = params.val_batch_size, shuffle = False,
                            num_workers = int(params.num_workers),
                            pin_memory=params.cuda)
        dataloaders[split] = dl

    return dataloaders

def imshow(img):
    npimg = img.numpy()[0]
    plt.imshow(npimg)
    plt.show()

if __name__ == '__main__':

    dl =  DataLoader(HDRDataset(train_transformer),
                            batch_size = 1, shuffle = True,
                            num_workers = int(8),
                            pin_memory=True)
    dataiter = iter(dl)
    images, labels = dataiter.next() ## 1x8x512x512, 1x1x512x512
    fig, ax = plt.subplots(2, 1, figsize=(20, 16))
    ax[0].imshow(torchvision.utils.make_grid(images[0, :4].unsqueeze(1)).numpy()[0])
    #ax[1].imshow(torchvision.utils.make_grid(images[:, 1:2]).numpy()[0])
    ax[1].imshow(torchvision.utils.make_grid(labels).numpy()[0])
    plt.show()
