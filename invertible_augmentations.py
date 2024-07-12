import itertools
import math
import numpy as np
import skimage.transform as ski_tra

class AugmentationFlipAffine:
    def __init__(self, hor_flip, scale, angle, shear, translation, img_shape, fill=0):
        self.hor_flip = hor_flip
        cor = (np.array(img_shape)-1) / 2
        self.angle = angle
        self.affine = (ski_tra.AffineTransform(translation = translation) 
            + (ski_tra.AffineTransform(translation=-cor)
            + (ski_tra.AffineTransform(scale = scale, rotation = angle, shear = shear)
            + ski_tra.AffineTransform(translation=cor))))
        self.fill = fill
        
    def apply(self, img):
        if self.hor_flip:
            img = np.fliplr(img)
        img = ski_tra.warp(
            img,
            inverse_map=self.affine.inverse,
            cval=self.fill,
            order=1)
        return img
    
    def apply_inverse(self, img):
        img = ski_tra.warp(
            img,
            inverse_map=self.affine,
            cval=self.fill,
            order=1)
        if self.hor_flip:
            img = np.fliplr(img)
        return img
    
    def __str__(self):
        return f"angle = {self.angle}, hor_flip = {self.hor_flip}"
        
def generate_flip_rot90_augmentations(img_shape, fill=0):
    augmentations = []
    for angle, do_flip in itertools.product([0, 90, 180, 270], [True, False]):
        augmentations.append(AugmentationFlipAffine(
            do_flip,
            1,
            math.radians(angle),
            0,
            (0,0),
            img_shape,
            fill=fill
        ))
    return augmentations
    
def generate_flip_augmentations(img_shape, fill=0):
    augmentations = []
    for do_flip in [True, False]:
        augmentations.append(AugmentationFlipAffine(
            do_flip,
            1,
            0,
            0,
            (0,0),
            img_shape,
            fill=fill
        ))
    return augmentations
    
def generate_random_CT_augmentations(num, img_shape, fill=0):
    augmentations = []
    for i in range(num):
        augmentations.append(AugmentationFlipAffine(
            np.random.choice([True, False]),
            np.random.uniform(0.95, 1.05, (2)),
            np.random.uniform(0, math.pi*2),
            np.random.uniform(-math.radians(10), math.radians(10)),
            np.random.uniform(-10, 10, (2)),
            img_shape,
            fill=fill
        ))
    return augmentations
    
def generate_random_proj_augmentations(num, img_shape, fill=0):
    augmentations = []
    for i in range(num):
        augmentations.append(AugmentationFlipAffine(
            np.random.choice([True, False]),
            np.random.uniform(0.913, 1.087, (2)),
            np.random.uniform(-math.radians(12), math.radians(12)),
            np.random.uniform(-math.radians(13.8), math.radians(13.8)),
            (0, 0),
            img_shape,
            fill=fill
        ))
    return augmentations
    
def generate_no_augmentations(fill=0):
    return [AugmentationFlipAffine(
        False,
        1,
        0,
        0,
        (0,0),
        (0,0),
        fill=fill)]
