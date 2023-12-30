import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from thop import profile

class HybridSN(nn.Module):
    def __init__(self, num_classes, self_attention=False):
        super(HybridSN, self).__init__()
        # out = (width - kernel_size + 2*padding)/stride + 1
        # => padding = ( stride * (out-1) + kernel_size - width)
        # 这里因为 stride == 1 所有卷积计算得到的padding都为 0

        # 默认不使用注意力机制
        self.self_attention = self_attention

        # 3D卷积块
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(7, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )

        # if self_attention:
        #     self.spatial_attention_1 = SpatialAttention(kernel_size=7)

        # 2D卷积块
        self.block_2_2D = nn.Sequential(
            nn.Conv2d(
                in_channels=576,
                out_channels=64,
                kernel_size=(3, 3)
            ),
            nn.ReLU(inplace=True)
        )

        # if self_attention:
        #     self.spatial_attention_2 = SpatialAttention(kernel_size=7)

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=18496,
                out_features=256
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
            # pytorch交叉熵损失函数是混合了softmax的。不需要再使用softmax
        )

    def forward(self, x):
        y = self.block_1_3D(x)
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
        if self.self_attention:
            #            y = self.channel_attention_1(y) * y
            y = self.spatial_attention_1(y) * y
        y = self.block_2_2D(y)
        if self.self_attention:
            #            y = self.channel_attention_2(y) * y
            y = self.spatial_attention_2(y) * y

        y = y.view(y.size(0), -1)

        y = self.classifier(y)
        return y


def hybridsn(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = HybridSN(num_classes=16)
    elif dataset == 'ip':
        model = HybridSN(num_classes=16)
    elif dataset == 'whulk':
        model = HybridSN(num_classes=9)
    elif dataset == 'botswana':
        model = HybridSN(num_classes=14)
    return model

if __name__ == '__main__':
    t = torch.randn(size=(1, 1, 30, 25, 25))
    print("input shape:", t.shape)
    net = hybridsn(dataset='ip', patch_size=25)
    print("output shape:", net(t).shape)
    flops, params = profile(net, inputs=(t,))
    print('params', params)
    print('flops', flops)  ## 打印计算量