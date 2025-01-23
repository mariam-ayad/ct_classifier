'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.

    2022 Benjamin Kellenberger
'''

import torch.nn as nn
from torchvision.models import resnet


class CustomResNet(nn.Module):

    def __init__(self, num_classes, layers=18):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(CustomResNet, self).__init__()

        # this is hack-y. Come up with better switch that doesn't involve hardcoding maybe?s
        if layers==18:
            self.feature_extractor = resnet.resnet18(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet
        elif layers==50:
            self.feature_extractor = resnet.resnet50(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet
            
        self.feature_extractor.conv1 = nn.Conv2d(in_channels=16, out_channels=self.feature_extractor.conv1.out_channels,
                                                 kernel_size=self.feature_extractor.conv1.kernel_size, 
                                                 stride=self.feature_extractor.conv1.stride, 
                                                 padding=self.feature_extractor.conv1.padding,bias=False)
        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
    

    def forward(self, x):
        '''
            Forward pass. Here, we define how to apply our model. It's basically
            applying our modified ResNet-18 on the input tensor ("x") and then
            apply the final classifier layer on the ResNet-18 output to get our
            num_classes prediction.
        '''
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction