import math
import torch.nn.functional as F
from torch import nn
import torch
from thop import profile

class Res2(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X

class Res(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups=1):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Z = self.res2(X)
        return F.relu(X + Y + Z)


class sprn(nn.Module):
    def __init__(self, bands, classes, groups, groups_width, spa=False):
        super(sprn, self).__init__()
        self.bands = bands
        self.classes = classes
        self.spa = spa
        fc_planes = 128

        # pad the bands with final values
        new_bands = math.ceil(bands / groups) * groups
        pad_size = new_bands - bands
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))

            # SPRN
        self.conv1 = nn.Conv2d(new_bands, groups * groups_width, (1, 1), groups=groups)
        self.bn1 = nn.BatchNorm2d(groups * groups_width)

        self.res0 = Res(groups * groups_width, (1, 1), (0, 0), groups=groups)
        self.res1 = Res(groups * groups_width, (1, 1), (0, 0), groups=groups)

        self.conv2 = nn.Conv2d(groups_width * groups, fc_planes, (1, 1))
        self.bn2 = nn.BatchNorm2d(fc_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fc = nn.Linear(fc_planes, classes)

    def forward(self, x):
        # input: (b, 1, d, w, h)
        x = self.pad(x).squeeze(1)
        if self.spa:
            x = self.spa_att(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res0(x)
        x = self.res1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        x = self.out_fc(x)
        return x


def SPRN(dataset):
    model = None
    if dataset == 'PaviaU':
        model = sprn(111, 9, 5, 64)
    elif dataset == 'ip':
        model = sprn(200, 16, 11, 37)
    elif dataset == 'sa':
        model = sprn(204, 16, 11, 37)
    return model

if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 200, 8, 8))
    print("input shape:", t.shape)
    net = SPRN(dataset='ip')
    flops, params = profile(net, inputs=(t,))
    print('params', params)
    print('flops', flops)  ## 打印计算量

    print("output shape:", net(t).shape)