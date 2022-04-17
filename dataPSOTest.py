import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import cv2
from dataset import mytransform as transfor

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, imageArray, blueChannel, loader=default_loader):

        self.imgArr = imageArray
        self.blue = blueChannel
        self.loader = loader
       
       
    def __getitem__(self, index):
        orgimg = self.imgArr[index]
        blueChannel = self.blue[index]
        orgimg = Image.fromarray(np.uint8(orgimg[:,:,::-1])).convert('RGB')

        orgimg = orgimg.resize((256,256))
        blueChannel = Image.fromarray(np.uint8(blueChannel)).convert('RGB')
        blueChannel = blueChannel.resize((256,256))

        sample = {'image': orgimg,
                'GT':blueChannel,
                'mylabel':blueChannel,
                'mylabel_binary':blueChannel,
                'part':'part',
                'train':False,
                'mask':blueChannel,
                'regTargeted':0,
                'imagename':'pso'}

        regTargeted  = 0  
        sample = self.transform_val(sample)
        orgimg = sample['image']
        mask = sample['mask']
        return orgimg, mask, regTargeted

    def __len__(self):
        return len(self.imgArr)

    def transform_train(self,sample):
        composed_transforms = transforms.Compose([
            #distort.PhotometricDistort(),
            transfor.FixedResize(size=256),
            transfor.RandomHorizontalFlip(),
            #transfor.RandomScaleCrop(base_size=256, crop_size=256, min_ratio = 0.7, max_ratio = 1.3, fill=255),
            transfor.FixedResize(size=256),
            transfor.RandomGaussianBlur(),
            transfor.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transfor.REFUGE_norm(),
            transfor.ToTensor()])
        
        return composed_transforms(sample)
        #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    def transform_val(self,sample):
        composed_transforms = transforms.Compose([
            transfor.FixedResize(size=256),
            transfor.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transfor.REFUGE_norm(),
            transfor.ToTensor()])

        return composed_transforms(sample)





class MyTestDataset(Dataset):
    def __init__(self, imageArray, target_list, transform, loader=default_loader):

        self.imgArr = imageArray
        self.target_list = target_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        orgimg = self.imgArr[index]
      
        orgimg = Image.fromarray(np.uint8(orgimg)).convert('RGB')
        orgimg = self.transform(orgimg)
        regTargeted  = self.target_list[index]  ## 无意义的返回
        return orgimg, regTargeted

    def __len__(self):
        return len(self.imgArr)