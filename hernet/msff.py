import torch
from torch import nn
import numpy as np
class FEBlock(nn.Module):

    def __init__(self, ch_num, scale=(1, 3, 5)):
        super().__init__()
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0])]
        self.scale1 = nn.Sequential(*block)
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0]), nn.Conv2d(ch_num, ch_num, 3, 1, scale[1], scale[1])]
        self.scale2 = nn.Sequential(*block)
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0]), nn.Conv2d(ch_num, ch_num, 3, 1, scale[1], scale[1]), nn.Conv2d(ch_num, ch_num, 3, 1, scale[2], scale[2])]
        self.scale3 = nn.Sequential(*block)

    def forward(self, x):
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        return [x, f1, f2, f3]

class FFBlock(nn.Module):

    def __init__(self, ch_num, ch_out):
        super().__init__()
        self.sff1 = SFFBlock(ch_num)
        self.sff2 = SFFBlock(ch_num)
        block = [nn.Conv2d(ch_num, (ch_num + ch_out) // 2, 1, 1), nn.BatchNorm2d((ch_num + ch_out) // 2), nn.ReLU(), nn.Conv2d((ch_num + ch_out) // 2, ch_out, 1, 1)]
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        y = self.sff1(x[1], x[2])
        y = self.sff2(y, x[3])
        y = x[0] + y
        return self.conv(y)

class SFFBlock(nn.Module):

    def __init__(self, ch_num):
        super().__init__()
        self.ch_num = ch_num
        self.conv1 = nn.Conv2d(ch_num * 2, ch_num, 1, 1, 0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(ch_num, ch_num // 8, 1, 1, 0)
        self.relu = nn.ReLU()
        self.conv3_1 = nn.Conv2d(ch_num // 8, ch_num, 1, 1, 0)
        self.conv3_2 = nn.Conv2d(ch_num // 8, ch_num, 1, 1, 0)
        self.conv3_3 = nn.Conv2d(ch_num // 8, ch_num, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, f1, f2):
        f3 = self.conv1(torch.cat((f1, f2), 1))
        f4 = self.relu(self.conv2(self.gap(f3)))
        f4_1 = self.conv3_1(f4)
        f4_2 = self.conv3_2(f4)
        f4_3 = self.conv3_3(f4)
        f4_12 = torch.cat((f4_1, f4_2), dim=1)
        f4_12 = f4_12.view(-1, 2, self.ch_num, 1, 1)
        f4_12 = self.softmax(f4_12)
        s1 = f4_12[:, 0, :, :, :]
        s2 = f4_12[:, 1, :, :, :]
        s3 = self.sigmoid(f4_3)
        return s1 * f1 + s2 * f2 + s3 * f3

class MSFFBlock(nn.Module):

    def __init__(self, ch_num, ch_out, scale=(1, 3, 5)):
        super().__init__()
        self.fe = FEBlock(ch_num, scale=scale)
        self.ff = FFBlock(ch_num, ch_out)

    def forward(self, x):
        return self.ff(self.fe(x))
