import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .utils import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def get_images(path,idxs = []):
    imgs_loc = []
    assert os.path.isdir(path), '%s is not a valid directory' % dir
    print("we will be doing an os.walk through ",path)
    for root, _, fnames in sorted(os.walk(path)):
        if not len(idxs) == 0:
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                imgs_loc.append(path)
        else:
            # rnames = np.random.choice(fnames)
            for fname in sorted(fnames[:5]):
                path = os.path.join(root, fname)
                imgs_loc.append(path)
    images =[]
    for loc in imgs_loc:
        img = Image.open(loc)#.convert('RGB')
        # Now normalize them
        transform = transforms.Compose([
            transforms.Resize(256,256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            ])
        img  = transform(img)
        images.append(img)

    return images

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_mode = 'free_form', image_size=[256, 256]):

        # Just chose 5 out of all of them 
        # Assume it is a dictionary 
        self.imgs = get_images(data_root)

        print("Using mask mode :{}".format(mask_mode))
        self.mask_mode = mask_mode
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        mask = self.get_mask()
        img = self.imgs[index]
        
        # Introduce some noise
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        # ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

