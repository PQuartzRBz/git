
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#Tensor board dependencies
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('path_to_logs')

from utils import AverageMeterSet
meters = AverageMeterSet()
meters.reset()

print('Start the code Here')
print(f'Torch version is {torch.__version__}')

ds_dir = 'C:\\Users\\Ribuzari\\Documents\\KMITL\\git\\src\\materials\\dataset'
target = 'C:\\Users\\Ribuzari\\Documents\\KMITL\\git\\src\\materials\\testDataset'
img_height,img_width=224,224

ds = torchvision.datasets.ImageFolder(ds_dir)

BATCH_SIZE = 128

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

import matplotlib.pyplot as plt
import numpy as np

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

## get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

## show images
imshow(torchvision.utils.make_grid(images))

for images, labels in trainloader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break

class MyModel(nn.Module):
    def __init__(self, image_size):
        super(MyModel, self).__init__()
        self.image_size = image_size
        self.img_size = image_size - 2
        # declare layer
        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) # can set acti arg
        self.d1 = nn.Linear(self.img_size * self.img_size * 32, 128)
        self.d2 = nn.Linear(128, 15)# ?? -> output(classes = 15)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out

model = MyModel(img_height)
for images, labels in trainloader:
    print("batch size:", images.shape)
    out = model(images)
    print(out.shape)
    break


learning_rate = 0.001
num_epochs = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel(img_height)
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

        writer.add_scalar('New loss',loss.detach().item(),count+i)
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
        
print('Test Accuracy: %.2f'%( test_acc/i))

print('result = {}'.format(meters['Batch loss']))

print(meters.counts())
print(meters.metrics)