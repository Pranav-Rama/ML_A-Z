# Taken from Deep Learning for PyTorch Part Four: Image Classifacation using Convolutional Neural Network and ResNets [t = 4:45:09]
import os
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F 
import tarfile

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

# Splitting test dataset to train and validation:
def split_indices(n, val_pct=0.1, seed=99):
    # Determine size of validation set:
    n_val = int(val_pct*n)
    # Set the random seed (for reproducibility)
    np.random.seed(seed)
    # Create random permutation of 0 to n-1:
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set:
    return idxs[n_val:], idxs[:n_val]

val_pct = 0.2
rand_seed = 42

train_indices, val_indices = split_indices(len(train_dataset), val_pct, rand_seed)
print(len(train_indices), len(val_indices))
print('Sample validation indices: ', val_indices[:10])

# Dataloader:
batch_size = 100

# Training sampler and data loader:
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(train_dataset, 
                      batch_size,
                      sampler=train_sampler)

# Valiodation sampler and data loader:
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(train_dataset,
                    batch_size,
                    sampler=val_sampler)

### Define a Convolutional Neural Network: ###
# simple_model = nn.Sequential(
#     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
#     nn.MaxPool2d(2,2)
# )

# # Check shape of input images and output of the convolution layer:
# for images, labels in train_dl:
#     print('Image Shape: ', images.shape)
#     out = simple_model(images)
#     print('Output Shape: ', out.shape)
#     break

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2), # output: bs x 16 x 16 x 16

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2), # output: bs x 16 x 8 x 8

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2), # output: bs x 16 x 4 x 4

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2), # output: bs x 16 x 2 x 2

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2), # output: bs x 16 x 1 x 1

    nn.Flatten(), # output: bs x 16
    nn.Linear(16,10) # output: bs x 10 
)

# for images, labels in train_dl:
#     print('Image Shape: ', images.shape)
#     out = model(images)
#     print('Output Shape: ', out.shape)
#     print('Output[0]: ', out[0])
#     break

#  To train model on GPU:
# device = torch.device('cuda')
def get_default_device():
    if torch.cuda.is_available():
        print('running on gpu')
        return torch.device('cuda')
    else:
        print('running on cpu')
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

device = get_default_device()
# print(device)
# Wrap the training and validation data loaders using DeviceDataLoader:
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(train_dl, device)
to_device(model, device)

# Train the model:
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

# Evaluation:
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                    for xb,yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metric,nums)) / total
    return avg_loss, total, avg_metric  

# Optimizer: 

def fit(epochs, model, loss_fn, train_dl, valid_dl,
          opt_fn=None, lr=None, metric=None):
    train_losses, val_losses, val_metrics = [], [], []

    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr)   

    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            train_loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)

        # Evaluation
        model.eval()
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # Record loss & metric:
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        # Print progress:
        if metric is None:
            print('Epoch [{}/{}], train_loss: {:4f}, val_loss: {:.4f}'
                  .format(epoch+1, epochs, train_loss, val_loss))
        else:
            print('Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}'
                  .format(epoch+1, epochs, train_loss, val_loss,
                          metric.__name__,val_metric))
    return train_losses, val_losses, val_metrics                      

# Accuracy:
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

num_epochs = 100
opt_fn = torch.optim.Adam
lr = 0.005
history = fit(num_epochs, model, F.cross_entropy,
              train_dl, valid_dl, opt_fn, lr, accuracy)
              
train_losses, val_losses, val_metrics = history
