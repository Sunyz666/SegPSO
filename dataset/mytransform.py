import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        mask = sample['mask'] 
        
        part = sample['part']
        train = sample['train'] 
        imagename = sample['imagename']
        regTargeted = sample['regTargeted']

        img = np.array(img).astype(np.float32)
        GT = np.array(GT).astype(np.float32)
        mylabel = np.array(mylabel).astype(np.float32)
        mylabel_binary = np.array(mylabel_binary).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
       
        img /= 255.0
        img -= self.mean
        img /= self.std

        mask /= 255.0
        mask -= self.mean
        mask /= self.std



        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
''' 
class binary_norm(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['GT']
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        mask[mask < 127.] = 0.
        mask[mask > 127.] = 1.
        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
 
class DGS_norm(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['GT']
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        new_mask = np.zeros(mask.shape, np.float32)
        new_mask[mask>64] = 1.
        new_mask[mask>200] = 2.
        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
'''
class REFUGE_norm(object):
    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        mask = sample['mask'] 
        
        part = sample['part']
        train = sample['train'] 
        imagename = sample['imagename']
        regTargeted = sample['regTargeted']

        mylabel = np.ceil(np.abs(mylabel - 255)/128.0).astype(np.float32)
        GT = np.ceil(np.abs(GT - 255)/128.0).astype(np.float32)
        #mylabel_binary = np.ceil(np.abs(mylabel_binary - 255)/128.0).astype(np.float32)
        #GT = np.ceil(np.abs(GT - 255)/128.0).astype(np.float32)
        
        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        mask = sample['mask'] 
        
        part = sample['part']
        train = sample['train'] 
        imagename = sample['imagename']
        regTargeted = sample['regTargeted']
        
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        GT = np.array(GT).astype(np.float32)
        mylabel = np.array(mylabel).astype(np.float32)
        mylabel_binary = np.array(mylabel_binary).astype(np.float32)
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        GT = torch.from_numpy(GT).float()
        mylabel = torch.from_numpy(mylabel).float()
        mylabel_binary = torch.from_numpy(mylabel_binary).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        mask = sample['mask'] 
        
        part = sample['part']
        train = sample['train'] 
        imagename = sample['imagename']
        regTargeted = sample['regTargeted']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            GT = GT.transpose(Image.FLIP_LEFT_RIGHT)
            mylabel = mylabel.transpose(Image.FLIP_LEFT_RIGHT)
            mylabel_binary = mylabel_binary.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)


        return {'image': img,
            'GT': GT,
            'mylabel':mylabel,
            'mylabel_binary':mylabel_binary,
            'part':part,
            'train':train,
            'mask':mask,
            'regTargeted':regTargeted,
            'imagename':imagename}
    
class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            GT = GT.transpose(Image.FLIP_TOP_BOTTOM)
            mylabel = mylabel.transpose(Image.FLIP_TOP_BOTTOM)
            mylabel_binary = mylabel_binary.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        GT = GT.rotate(rotate_degree, Image.NEAREST)
        mylabel = mylabel.rotate(rotate_degree, Image.NEAREST)
        mylabel_binary = mylabel_binary.rotate(rotate_degree, Image.NEAREST)
        mask = mask.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, min_ratio, max_ratio, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.fill = fill

    def __call__(self, sample):

        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.min_ratio), int(self.base_size * self.max_ratio))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR) 
        GT = GT.resize((ow, oh), Image.NEAREST)
        mylabel = mylabel.resize((ow, oh), Image.NEAREST)
        mylabel_binary = mylabel_binary.resize((ow, oh), Image.NEAREST)
        mask = mask.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) #left, top, right, bottom
            GT = ImageOps.expand(GT, border=(0, 0, padw, padh), fill=self.fill)
            mylabel = ImageOps.expand(mylabel, border=(0, 0, padw, padh), fill=self.fill)
            mylabel_binary = ImageOps.expand(mylabel_binary, border=(0, 0, padw, padh), fill=self.fill)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))  #left, top, right, bottom
        GT = GT.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mylabel = mylabel.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mylabel_binary = mylabel_binary.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
       
        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['GT']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'GT': GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)
    def __call__(self, sample):
       
        img = sample['image']
        GT = sample['GT']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        GT = GT.resize(self.size, Image.NEAREST)
        mylabel = mylabel.resize(self.size, Image.NEAREST)
        mylabel_binary = mylabel_binary.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size,Image.BILINEAR)

        return {'image': img,
                'GT': GT,
                'mylabel': mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
