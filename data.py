from os.path import exists, join, basename
from os import makedirs, remove
import numpy as np

from torchvision.transforms import Compose, Normalize, ToTensor
import torch

from dataset import DatasetFromFolder, DatasetFromFolderSyn


class Poiss_noise(object):
    """
    Add poisson noise to PIL image
    """
    def __init__(self,PEAK):
        self.PEAK=PEAK

    def __call__(self,img):
        sampled_lambda = np.random.uniform(1, 50)
        return torch.poisson(img *sampled_lambda )/sampled_lambda


class Gaussian_noise(object):
    """
    Add Gaussian noise to PIL image for random sigma between 1 and 75
    """
    def __init__(self,std=10):
        self.std=std
    def __call__(self,img):
        std = np.random.uniform(1, 75)
        return img + torch.randn(img.shape)*float(std/255)

def input_transform(PEAK=30):
    """
    Performe transformation on the input image
    """

    return Compose([
        ToTensor(),Normalize((0.0,),(1.0,)),Poiss_noise(PEAK),Gaussian_noise()

    ])


def target_transform():
    """
    Performe transformation on the ground truth image
    """
    return Compose([
        ToTensor(),Normalize((0.0,),(1.0,))
    ])


def get_training_set(f_dir,synthesize=False):
    """
    Get the training dataset.
    """

    root_dir = f_dir
    train_dir = join(root_dir, "train")
    train_dir_inputs = join(train_dir, "inputs")
    train_dir_labels = join(train_dir, "labels")
    if synthesize:
        return DatasetFromFolderSyn(train_dir_inputs,
                             mode='train',synthesize=synthesize,
                             target_transform=target_transform(),
                             input_transform=input_transform()
                             )
    else:
        return DatasetFromFolder(train_dir_inputs,train_dir_labels,
                             mode='train',synthesize=synthesize,
                             target_transform=target_transform(),
                             input_transform=target_transform()
                             )


def get_test_set(f_dir,synthesize=False):
    """
    Get the testing dataset.
    """
    root_dir = f_dir
    test_dir = join(root_dir, "val")
    test_dir_inputs = join(test_dir, "inputs")
    test_dir_labels = join(test_dir, "labels")

    if synthesize:
        return DatasetFromFolderSyn(test_dir_inputs,
                             mode='val',synthesize=synthesize,
                             target_transform=target_transform(),
                             input_transform=input_transform()
                             )
    else:
        return DatasetFromFolder(test_dir_inputs,test_dir_labels,
                             mode='val',synthesize=synthesize,
                             target_transform=target_transform(),
                             input_transform=target_transform()
                             )
