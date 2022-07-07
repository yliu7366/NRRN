import random
import numpy as np
from PIL import Image, ImageOps
from os import listdir
from os.path import join

import torch.utils.data as data

#20220705 YL
from os.path import splitext
import glob
import os

#Define a constant
VOL_SIZE = 10  # number of images in a slice

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()

    return y


def _sync_transform(img, clean, img1,img2):
    """
    Random Image augmentation synchronized among the triplet input and the ground truth images
    """
    base_size = 256
    crop_size = 200
    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        clean = clean.transpose(Image.FLIP_LEFT_RIGHT)

    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    img1 = img1.resize((ow, oh), Image.BILINEAR)
    img2 = img2.resize((ow, oh), Image.BILINEAR)
    clean = clean.resize((ow, oh), Image.BILINEAR)

    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img1 = ImageOps.expand(img1, border=(0, 0, padw, padh), fill=0)
        img2 = ImageOps.expand(img2, border=(0, 0, padw, padh), fill=0)
        clean = ImageOps.expand(clean, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    img1 = img1.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    img2 = img2.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    clean = clean.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    return img, img1, clean, img2

# Simplified version to use syn dataset
class DatasetFromFolderSyn(data.Dataset):
    def __init__(self, image_dir, mode='train', synthesize=True, target_transform=None, input_transform=None):
        super(DatasetFromFolderSyn, self).__init__()
        self.image_dir=image_dir
        self.image_filenames = []
        self.target_filenames = []
        self.image2_filenames = []
        self.image3_filenames = []
        self.mode = mode
        self.synthesize=synthesize

        files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

        self.image_filenames = files[0::3]
        self.target_filenames = files[0::3]
        self.image2_filenames = files[1::3]
        self.image3_filenames = files[2::3]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # Input triplets
        image1 = load_img(self.image_filenames[index])
        image2 = load_img(self.image2_filenames[index])
        image3 = load_img(self.image3_filenames[index])

        # Corresponding ground truth
        clean = load_img(self.target_filenames[index])
        # in the case of synthesize noise we add a bit of data augmentation but only to the training dataset
        if self.mode == 'train' and self.synthesize:
            image1, image2, clean, image3 = _sync_transform(image1, clean, image2, image3)

        # applying the input and target transformations
        if self.input_transform:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)
            image3 = self.input_transform(image3)

        if self.target_transform:
            clean = self.target_transform(clean)

        return image1, image2, image3, clean

    def __len__(self):
        return len(self.image_filenames)

# Acquire image triplets and their ground truth
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_dir, mode='train', synthesize=False, target_transform=None, input_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir=image_dir
        self.image_filenames = []
        self.target_filenames = []
        self.image2_filenames = []
        self.image3_filenames = []
        self.mode = mode
        self.synthesize=synthesize

        # Get input filenames and its corresponding target name
        files = sorted(listdir(image_dir))

        for i in range(len(files)):
            x = files[i]
            if is_image_file(x):
                self.image_filenames.append(join(image_dir, x))

                if synthesize:
                    target_name = x
                else:
                    target_name = "Outputs_" + x.partition("_")[2]
                self.target_filenames.append(join(target_dir,target_name))
                # Geting the image before and the image after the one we want to denoise
                separator = '_'
                img_name_spl = splitext(x)[0].split(separator)

                if synthesize:      # example EPFL data set
                    k = int(img_name_spl[-1])
                    l = len(listdir(image_dir)) #running through all images in the folder
                else:               # OHSU data set
                    k = int(img_name_spl[-3])
                    l = VOL_SIZE    # running through the images in a volume

                n = k + 1
                n2 = k - 1

                if i == 0:
                    n = k + 1
                    n2 = k + 2
                elif i == l-1:
                    n = k - 1
                    n2 = k - 2

                if synthesize:
                    inp_name = img_name_spl[0] + separator + str(n).zfill(4) + '.' + x.split(".")[-1]
                    inp_name2 = img_name_spl[0] + separator + str(n2).zfill(4) + '.' + x.split(".")[-1]
                else: # OHSU if we are denoising image Inputs_5_8_14_22.png
                    common_name = 'Inputs'+separator+img_name_spl[-4] #Inputs_5

                    inp_name = common_name + separator+str(n) + separator#Inputs_5_7_
                    inp_name += img_name_spl[-2] + separator + img_name_spl[-1] #Inputs_5_7_14_22.png
                    inp_name2 = common_name+separator + str(n2) + separator #Inputs_5_9_
                    inp_name2 += img_name_spl[-2] + separator + img_name_spl[-1] #Inputs_5_9_14_22.png

                self.image2_filenames.append(join(self.image_dir,inp_name))
                self.image3_filenames.append(join(self.image_dir,inp_name2))

        self.input_transform = target_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # Input triplets
        image1 = load_img(self.image_filenames[index])
        image2 = load_img(self.image2_filenames[index])
        image3 = load_img(self.image3_filenames[index])

        # Corresponding ground truth
        clean = load_img(self.target_filenames[index])
        # in the case of synthesize noise we add a bit of data augmentation but only to the training dataset
        if self.mode == 'train' and self.synthesize:
            image1, image2, clean, image3 = _sync_transform(image1, clean, image2, image3)

        # applying the input and target transformations
        if self.input_transform:
            image1 = self.target_transform(image1)
            image2 = self.target_transform(image2)
            image3 = self.target_transform(image3)

        if self.target_transform:
            clean = self.target_transform(clean)

        return image1, image2, image3, clean

    def __len__(self):
        return len(self.image_filenames)
