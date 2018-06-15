## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Assume input size = 224 x 224, then output will be (224 - 5) / 1 + 1 = 32x220x220 (CxHxW)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.conv1.weight)
        
        # Maybe include dropout after conv layers
        self.conv1_drop = nn.Dropout(p = 0.2) 
        
        # If we do 2x2 max pooling then
        # Input = 32 x 110 x 110, Ouput = 64 x 108 x 108
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout(p = 0.2) 
        
        # After pooling
        # Input = 64 x 54 x 54, Output = (54 - 2) / 1 + 1 = 128 x 53 x 53
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout(p = 0.3)
        
        # After pooling 
        # Input = 128 x 26 x 26
        self.conv4 = nn.Conv2d(128, 256, 1)
        # Output = 256 x 26 x 26
        
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout(p = 0.3)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layer 1, after pooling input = 256 x 13 x 13
        self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        # FC1 dropout
        self.fc1_drop = nn.Dropout(p = 0.3)
        
        # Fully Connected Layer 2        
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        #FC2 dropout
        self.fc2_drop = nn.Dropout(p = 0.4)
        
        # Fully Connected Layer 3
        self.fc3 = nn.Linear(1000, 136)
        self.fc3_bn = nn.BatchNorm1d(136)
        #FC3 dropout
        self.fc3_drop = nn.Dropout(p = 0.4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # 1st conv layer -> batch norm -> activation -> pooling -> dropout
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        #x = self.conv1_drop(x)
        
        
   
        # 2nd conv layer -> batch norm -> activation -> pooling -> dropout
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        #x = self.conv2_drop(x)
        
        
        # 3rd conv layer -> batch norm -> activation -> pooling -> dropout
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))        
        #x = self.conv3_drop(x)
        
        
        # 4th conv layer -> batch norm -> activation -> pooling -> dropout
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))        
        #x = self.conv4_drop(x)
                      
                      
        # Before fully connected layers we flatten
        x = x.view(x.size(0), -1)
        
        # 1st fully connected layer -> batch norm -> activation -> dropout
        x = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(x))))
        
        # 2nd fully connected layer -> batch norm -> activaation -> dropout
        x = self.fc2_drop(F.relu(self.fc2_bn(self.fc2(x))))
        
        # 3rd fully connected layer followed by activation and dropout
        x = (self.fc3(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
