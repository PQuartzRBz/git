import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # declare layer
        # 28x28x3 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) # can set acti arg
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.m1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # can set acti arg
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.m2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) # can set acti arg
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.m3 = nn.Dropout(p=0.2)

        self.d1 = nn.Linear(86528, 128)
        self.m4 = nn.Dropout(p=0.2)

        self.d2 = nn.Linear(128, 15)# ?? -> output(classes = 15)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        # x = self.m1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        # x = self.m2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        # x = self.m3(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)
        # x = self.m4(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return logits