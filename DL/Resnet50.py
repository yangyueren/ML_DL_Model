import torch.nn as nn
import torch.nn.functional as F
import torch

class OneBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(OneBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, num_class=100):
        super(Resnet50, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, dilation=1)
        self.conv2_x = self.make_layer(64, 3, stride=1)
        self.conv3_x = self.make_layer(128, 4)
        self.conv4_x = self.make_layer(256, 6)
        self.conv5_x = self.make_layer(512, 3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * 4, num_class)


    def make_layer(self, planes, block_num, stride=2):
        downsample = None
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(self.in_planes, planes*4, kernel_size=1,  stride=stride, bias=False),
            nn.BatchNorm2d(planes*4),
        )
        layers.append(OneBlock(self.in_planes, planes, stride=stride, downsample=downsample))
        self.in_planes = planes*4
        for i in range(1, block_num):
            layers.append(OneBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1
        out = self.conv1(x)
        print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        print(out.shape)
        # conv2_x
        out = self.maxpool(out)
        out = self.conv2_x(out)

        # conv3_x
        out = self.conv3_x(out)

        # conv4_x
        out = self.conv4_x(out)

        # conv5_x
        out = self.conv5_x(out)

        # average pool and fc
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
