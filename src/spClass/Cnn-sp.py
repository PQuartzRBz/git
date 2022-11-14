import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from Modelss import MyModel

#Tensor board dependencies

import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment = str(time.time()))

from utils import AverageMeterSet
meters = AverageMeterSet()
meters.reset()

import matplotlib.pyplot as plt
import numpy as np

ds_dir = 'C:\\Users\\Ribuzari\\Documents\\KMITL\\git\\src\\materials\\dataset'
target = 'C:\\Users\\Ribuzari\\Documents\\KMITL\\git\\src\\materials\\testDataset'
img_height,img_width=224,224

ds = torchvision.datasets.ImageFolder(ds_dir)

BATCH_SIZE = 32

## transformations
transform = transforms.Compose([
    transforms.Resize((img_height,img_width)),
    transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.ImageFolder(root=ds_dir,
                                         transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

## download and load testing dataset
testset = torchvision.datasets.ImageFolder(root=target,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

for images, labels in trainloader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break



learning_rate = 0.001
num_epochs = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

count = 0
last = 0
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step / BATCH
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        ## update model params
        optimizer.step()
        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, BATCH_SIZE)

        writer.add_scalar('another loss',loss.detach().item(),count+i)
        meters.update('Batch loss', loss.detach().item())
        print('batch: %d | Loss: %.4f | count: %d' \
          %(i, loss.detach().item(),count+i+1))
        last = i+1
    
    count += last
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i))

test_acc = 0.0
for i, (images, labels) in enumerate(testloader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
    count = i+1
        
print('Test Accuracy: %.2f'%( test_acc/count))

# print('result = {}'.format(meters['Batch loss']))
writer.close()
print(meters.counts())
print(meters.metrics)