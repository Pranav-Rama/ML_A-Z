# Taken from Deep Learning for PyTorch Part Four: Image Classifacation using Convolutional Neural Network and ResNets [t = 4:45:09]
import os
import torch
import torchvision
import tarfile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import cv2

# Training and validation dataset path:
train_dataset_path = '/home/pranavan/Documents/PyTorch_dev/cifar10/train'
test_dataset_path = '/home/pranavan/Documents/PyTorch_dev/cifar10/test'

classes = os.listdir(train_dataset_path)
print(classes)

# Convert training dataset images into Tensors:
train_dataset = ImageFolder(train_dataset_path, transform=ToTensor())

# Check converted Images:
img, label = train_dataset[0]
print(img.shape, label)
print(img)

print(train_dataset.classes)



# # # View the image:
def show_example(img, label):
    print('Label: ', train_dataset.classes[label], '('+str(label)+')')
    # img = img.permute(1, 2, 0)
    # cv2.imshow('Label', img)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

# # cv2.waitkey(0)
# # cv2.destroyAllWindows

show_example(*train_dataset[0])
