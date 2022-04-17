from PIL import Image
import cv2
import numpy as np
import types
from numpy import random

import cv2
import numpy as np
import types
from numpy import random

def Gamma_trans(img,gamma):
    img = np.array(img, dtype=np.uint8)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def CLAHE(img):
    img = np.array(img, dtype=np.uint8)
    shape = img.shape
    new_img = np.empty(shape)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if len(shape) == 2:
        new_img = clahe.apply(img)
    elif len(shape) == 3:
        for i in range(shape[-1]):
            new_img[:,:,i] = clahe.apply(img[:,:,i])
    else:
        raise Exception("format error")
    return new_img.astype('uint8')


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)
    
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):

        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

    
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            
        return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness(32.)
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, sample):
        im = sample['image']
        GT = sample['GT']
        mylabel = sample['image'] 
        mylabel_binary = sample['mylabel_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']

        im = np.array(im).copy()
        im = im.astype(np.float32)
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        #im = self.rand_light_noise(im) # shuffle channels
        im = np.maximum(im, 0)
        im = np.minimum(im, 255)
        im = im.astype('uint8')
        im = Image.fromarray(im)
        
        return {'image': im,
                'GT': GT,
                'mylabel': mylabel,
                'mylabel_binary':mylabel_binary,
                'part':part,
                'train':train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}