import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-1])
        self.linear = nn.Linear( models.resnet152(pretrained=True).fc.in_features, 64)
        self.fc1 = nn.Linear(64, args.num_cls)
        self.bn = nn.BatchNorm1d(args.num_cls, momentum=0.01)

    def forward(self, image):
        with torch.no_grad():
            img = self.resnet(image)
        features = img.reshape(img.size(0), -1)
        output = self.bn(self.fc1(self.linear(features)))
        return output
