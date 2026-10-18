import torch
from torch import nn
import numpy as np
def get_block_list_generator(funcs_list):

    def get_block_list(args_list, kwargs_list):
        block = []
        for i in range(len(funcs_list)):
            block.append(funcs_list[i](*args_list[i], **kwargs_list[i]))
        return block
    return get_block_list

class PlainResidualBlock(nn.Module):

    def __init__(self, ch_num, act_type='relu'):
        super().__init__()
        block = [nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1), nn.BatchNorm2d(ch_num)]
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
            block.append(nn.ReLU(inplace=True))
        elif act_type == 'prelu':
            self.act = nn.PReLU()
            block.append(nn.PReLU())
        else:
            assert act_type == 'prelu_n'
            self.act = nn.PReLU(ch_num)
            block.append(nn.PReLU(ch_num))
        block += [nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1), nn.BatchNorm2d(ch_num)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = x + self.block(x)
        y = self.act(y)
        return y
