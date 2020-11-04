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
    
class GroupRawRandomCrop(object):
    def __init__(self, size, offset = 0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.offset = offset
    @staticmethod
    def get_params(img, output_size):

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) //2 * 2 + self.offset
        j = random.randint(0, w - tw) //2 * 2 + self.offset
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
    GroupToPILImage(mode = 'F'),
    GroupRawRandomCrop(512),
#     GroupRandomHorizontalFlip(),
#     GroupRandomVerticalFlip(),
    GroupToTensor()
])

eval_transformer =  GroupComposed([
    GroupToPILImage(mode = 'F'),
    GroupRawRandomCrop(512),
    GroupToTensor()
])

def raw_normalize(*raw_imgs): 
    ## may include quantization error?
    #return [((img - img.min())/(img.max() - img.min())).astype(np.float32) for img in raw_imgs]
    #[(img/2.0**np.ceil(np.log2(img.max()))).astype(np.float32) for img in raw_imgs] #[img.astype(np.float32) for img in raw_imgs]#
    return [img.astype(np.float32) for img in raw_imgs]

def paired_normalize(inputs, gt, percentage = 99.9):
    thres = np.percentile(gt, percentage)
    return torch.clamp(inputs, 0., thres)/thres, torch.clamp(gt, 0., thres)/thres

class HDRDataset(Dataset):

    def __init__(self, transform = None, length = 500, input_path = 'data/inputs_1015a.npy', gt_path = 'data/gt_1015a.npy'):
        self.transform = transform
        self.inputs = np.load(input_path)[:, ::-1, ::-1].transpose(1, 2, 0).astype(np.float32)/1023. # 8x2174x3864
        self.gt = np.load(gt_path)[np.newaxis][:, ::-1, ::-1].transpose(1, 2, 0).astype(np.float32)/(1024.*8.-1.)  # 1x2174x3864

        self.length = length
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frames = raw_normalize(*(np.split(self.inputs, self.inputs.shape[-1], axis=-1)), self.gt)
        if self.transform is not None:
            frames = self.transform(*frames) ## 8x512x512
        img = torch.cat(frames[:-1], dim=0) ## 1x512x512
        gt = frames[-1]
        return img, gt

## ----------------------begin Minghao llrawset---------------------
class LlrawSet(Dataset):
    """
    A barebone loading method to load low_light raw video dataset
    Currently do NOT support slice indexing
    """
    def __init__(self, n_samples = None,
                 h5file = 'data/mh_lowlight_10bit_Oct22/ll_dataset_hdr_10bit_Oct22.hdf5',
                 frame_per_sample = 8,
                 transform = None,
                 pos='mid'):
        self.fpath = h5file
        self.N = frame_per_sample
        self.pos = pos
        if self.pos == 'first':
            self.ref_offset = 0
        elif self.pos == 'mid':
            self.ref_offset = (self.N-1)//2
        elif self.pos == 'last':
            self.ref_offset = self.N-1
        else:
            raise RuntimeError("'{}' is not a valid position parameter. Choose from first/mid/last")
        with h5py.File(self.fpath, 'r') as db:
            self.frame_counts = np.array(db['frame_counts'])
            self.noise_params = np.array(db['noise_params'])
            
        self.sample_cumsum = np.cumsum(self.frame_counts-self.N+1) #calculate vid&frame index from index
        self.transform = transform
        if n_samples is None:
            self.n_samples = self.sample_cumsum[-1]
        else:
            self.n_samples = n_samples
            assert n_samples<=self.sample_cumsum[-1], \
                "Requesting {} samples, but the dataset contains only {} samples".format(n_samples, self.sample_cumsum[-1])
        
    def __len__(self):
        return self.n_samples #self.sample_cumsum[-1]
    
    def __getitem__(self, ind):
        assert ind>=0 and ind<len(self), "Index out of bound"
        vid_ind = np.searchsorted(self.sample_cumsum, ind, side = 'right')
        first_frame_ind = int(ind-self.N+1 - (int(self.sample_cumsum[vid_ind])-int(self.frame_counts[vid_ind])))

        with h5py.File(self.fpath, 'r') as db:
            llvid = np.array(db["llvid_{:03d}".format(vid_ind)][first_frame_ind:first_frame_ind+self.N])   ## n_seq x H x W
            llvid = np.concatenate([llvid[self.ref_offset:self.ref_offset+1], np.delete(llvid, self.ref_offset, 0)], 0) # ref frame to first
            llvid = llvid.transpose(1, 2, 0) # H x W x n_seq
            assert llvid.shape[-1] == self.N 
            gt = np.array(db["gtvid_{:03d}".format(vid_ind)][first_frame_ind+self.ref_offset])[..., np.newaxis] ## H x W x 1
        noise_param = self.noise_params[vid_ind]
        
        #### modified by Qian 10/27/20 ####
        #### normalization ####
        llvid = (llvid - 50.0)/(1023.0-50.0)
        gt = gt/65535.0
        frames = raw_normalize(*(np.split(llvid, llvid.shape[-1],axis=-1)), gt)
        frames = self.transform(*frames)
        img = torch.cat(frames[:-1], dim=0)
        gt = frames[-1]
#         img, gt = paired_normalize(img, gt)
        return img, gt#, noise_param

## ----------------------end Minghao llrawset---------------------

def fetch_dataloader(params = None, types = 'train'):

    dataloaders = {}

    for split in ['train', 'val','test']:
        if split == 'train':
#             dl = DataLoader(HDRDataset(train_transformer, 500),
#                             batch_size = params.batch_size, shuffle = True,
#                             num_workers = int(params.num_workers),
#                             pin_memory=params.cuda)
            train_set = LlrawSet(transform = train_transformer)
            sublist = list(range(0, len(train_set), 50))
            train_subset = torch.utils.data.Subset(train_set, sublist)
        
            dl = DataLoader(train_subset,
                            batch_size = params.batch_size, shuffle = True,
                            num_workers = int(params.num_workers),
                            pin_memory=params.cuda)
        else:
#             dl = DataLoader(HDRDataset(eval_transformer, 50),
#                             batch_size = params.val_batch_size, shuffle = False,
#                             num_workers = int(params.num_workers),
#                             pin_memory=params.cuda)
#             dl = DataLoader(LlrawSet(transform = eval_transformer, n_samples = 50),
#                             batch_size = params.val_batch_size, shuffle = False,
#                             num_workers = int(params.num_workers),
#                             pin_memory=params.cuda)
            test_set = LlrawSet(transform = eval_transformer)
            sublist = list(range(22, len(test_set), 500))
            test_subset = torch.utils.data.Subset(test_set, sublist)  
            dl = DataLoader(test_subset,
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

#    dl =  DataLoader(HDRDataset(train_transformer),
#                            batch_size = 1, shuffle = True,
#                            num_workers = int(8),
#                            pin_memory=True)
    dl =  DataLoader(LlrawSet(transform = train_transformer),
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
