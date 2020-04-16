#Alexnet module architecture

import numpy as np
import torch 
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, number_of_classes=6):
        super(Alexnet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4)
        self.conv1_bn = nn.BatchNorm2d(48)
        
        self.maxpool1=nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.cnn2 = nn.Conv2d(in_channels=48, out_channels=256, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(256)

        self.maxpool2=nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
        
        self.cnn4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
        
        self.cnn5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        
        self.fc2 = nn.Linear(4096,4096)
        
        self.fc3 = nn.Linear(4096,number_of_classes)
    
    def forward(self, x):
        x = self.cnn1(x)
        x=self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x=self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = self.cnn3(x)
        x = torch.relu(x)
        
        x = self.cnn4(x)
        x = torch.relu(x)
        
        x = self.cnn5(x)
        x = torch.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        
        return x
