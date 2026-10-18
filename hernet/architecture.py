import torch
from torch import nn
import numpy as np
from . import msff
from .blocks import get_block_list_generator, PlainResidualBlock
from .transformer import TransformerBlockV2
class SpatialSoftRouter(nn.Module):

    def __init__(self, ch_num, res_n=0, need_norm=False):
        super().__init__()
        self.ch_num = ch_num
        self.need_norm = need_norm
        self.norm1 = nn.LayerNorm(ch_num)
        self.norm2 = nn.LayerNorm(ch_num)
        self.conv1 = nn.Conv2d(ch_num, ch_num // 8, 1, 1, 0)
        self.conv2 = nn.Conv2d(ch_num, ch_num // 8, 1, 1, 0)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(ch_num // 8, 2, 1, 1, 0)
        self.softmax = nn.Softmax(1)
        if res_n > 0:
            block = [PlainResidualBlock(ch_num, act_type='prelu_n') for _ in range(res_n)]
        else:
            block = [nn.Identity()]
        self.res_block = nn.Sequential(*block)

    def forward(self, x1, x2):
        if self.need_norm:
            x1_norm = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2_norm = self.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            f = self.relu(self.conv1(x1_norm) + self.conv2(x2_norm))
        else:
            f = self.relu(self.conv1(x1) + self.conv2(x2))
        f = self.softmax(self.conv3(f))
        s1 = f[:, [0], :, :]
        s2 = f[:, [1], :, :]
        return (self.res_block(s1 * x1 + s2 * x2), s1, s2)

class OrientationHead(nn.Module):

    def __init__(self, ch_num, need_fe=False):
        super().__init__()
        if need_fe:
            block = [msff.MSFFBlock(ch_num, 2), nn.Tanh()]
        else:
            block = [msff.FFBlock(ch_num, 2), nn.Tanh()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class MinutiaeHead(nn.Module):

    def __init__(self, ch_num, need_fe=False):
        super().__init__()
        if need_fe:
            block = [msff.MSFFBlock(ch_num, 1), nn.Sigmoid()]
        else:
            block = [msff.FFBlock(ch_num, 1), nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class FCE(nn.Module):

    def __init__(self, ch_num, res_n_h):
        super().__init__()
        block = [PlainResidualBlock(ch_num, act_type='prelu_n') for _ in range(res_n_h)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class LTE(nn.Module):

    def __init__(self, ch_num, tf_img_sz, res_n_l, pos_embed_flag=True):
        super().__init__()
        if isinstance(tf_img_sz, int):
            tf_img_sz = (tf_img_sz, tf_img_sz)
        conv_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.PReLU])
        args = [[ch_num, ch_num, 3, 1, 1], [ch_num], []]
        kwargs = [{}, {}, {}]
        block = conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 2, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 1, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 2, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        tf_block = TransformerBlockV2(in_chans=ch_num, embed_dim=ch_num, img_sz=tf_img_sz, patch_size=(1, 1), pos_embed_flag=pos_embed_flag, out_chans=ch_num, encoder_n=res_n_l, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0, act_layer=None, norm_layer=nn.LayerNorm, final_norm=True)
        block.append(tf_block)
        conv_t_generator = get_block_list_generator([nn.ConvTranspose2d, nn.BatchNorm2d, nn.PReLU, nn.Conv2d, nn.BatchNorm2d])
        args = [[ch_num, ch_num, 2, 2], [ch_num], [], [ch_num, ch_num, 3, 1, 1], [ch_num]]
        kwargs = [{}, {}, {}, {}, {}]
        block += conv_t_generator(args, kwargs)
        args = [[ch_num, ch_num, 2, 2], [ch_num], [], [ch_num, ch_num, 3, 1, 1], [ch_num]]
        block += conv_t_generator(args, kwargs)
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class CCE(nn.Module):

    def __init__(self, ch_num, res_n_b):
        super().__init__()
        conv_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.PReLU])
        args = [[ch_num, ch_num, 3, 1, 1], [ch_num], []]
        kwargs = [{}, {}, {}]
        block = conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 2, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 1, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        args = [[ch_num, ch_num, 3, 2, 1], [ch_num], []]
        block += conv_generator(args, kwargs)
        if res_n_b > 0:
            block += [PlainResidualBlock(ch_num, act_type='prelu_n') for _ in range(res_n_b)]
        conv_t_generator = get_block_list_generator([nn.ConvTranspose2d, nn.BatchNorm2d, nn.PReLU, nn.Conv2d, nn.BatchNorm2d])
        args = [[ch_num, ch_num, 2, 2], [ch_num], [], [ch_num, ch_num, 3, 1, 1], [ch_num]]
        kwargs = [{}, {}, {}, {}, {}]
        block += conv_t_generator(args, kwargs)
        args = [[ch_num, ch_num, 2, 2], [ch_num], [], [ch_num, ch_num, 3, 1, 1], [ch_num]]
        block += conv_t_generator(args, kwargs)
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class HierarchicalFusion(nn.Module):

    def __init__(self, ch_num, res_n_hl, res_n_fb):
        super().__init__()
        self.sf_block_hl = SpatialSoftRouter(ch_num, res_n_hl)
        self.sf_block_fb = SpatialSoftRouter(ch_num, res_n_fb)
        self.msff_fe = msff.FEBlock(ch_num)
        self.msff_ff = msff.FFBlock(ch_num, ch_num)

    def forward(self, y_h, y_l, y_b):
        y_hl, at_h, at_l = self.sf_block_hl(y_h, y_l)
        y_fb, at_f, at_b = self.sf_block_fb(y_hl, y_b)
        y_msf = self.msff_fe(y_fb)
        y_out = self.msff_ff(y_msf)
        return (y_out, y_hl, at_h, at_l, at_f, at_b)

class HERModule(nn.Module):

    def __init__(self, ch_num, res_num_n, tf_img_sz, auxiliary=False):
        super().__init__()
        res_n_h, res_n_l, res_n_b, res_n_hl, res_n_fb = res_num_n
        self.high_branch = FCE(ch_num, res_n_h)
        self.low_branch = LTE(ch_num, tf_img_sz, res_n_l)
        self.background_branch = CCE(ch_num, res_n_b)
        self.branch_fusion = HierarchicalFusion(ch_num, res_n_hl, res_n_fb)
        self.o_guidance = OrientationHead(ch_num, need_fe=True) if auxiliary else nn.Identity()
        self.m_guidance = MinutiaeHead(ch_num, need_fe=True) if auxiliary else nn.Identity()

    def forward(self, x, act_b='blh'):
        y_h = self.high_branch(x)
        y_l = self.low_branch(x)
        o_l = self.o_guidance(y_l)
        y_b = self.background_branch(x)
        if 'h' not in act_b:
            y_h = torch.zeros_like(y_h)
        if 'l' not in act_b:
            y_l = torch.zeros_like(y_l)
        if 'b' not in act_b:
            y_b = torch.zeros_like(y_b)
        y_out, y_hl, at_h, at_l, at_f, at_b = self.branch_fusion(y_h, y_l, y_b)
        m_hl = self.m_guidance(y_hl)
        return (y_out, o_l, m_hl, y_h, y_l, y_b, at_h, at_l, at_f, at_b)

class Encoder(nn.Module):

    def __init__(self, ch_in, ch_n):
        super().__init__()
        conv_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.PReLU])
        args = [[ch_in, ch_n, 3, 1, 1], [ch_n], []]
        kwargs = [{}, {}, {}]
        block = conv_generator(args, kwargs)
        args = [[ch_n, ch_n, 3, 2, 1], [ch_n], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n, ch_n * 2, 3, 1, 1], [ch_n * 2], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n * 2, ch_n * 2, 3, 2, 1], [ch_n * 2], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n * 2, ch_n * 4, 3, 1, 1], [ch_n * 4], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n * 4, ch_n * 4, 3, 2, 1], [ch_n * 4], []]
        block += conv_generator(args, kwargs)
        block += [PlainResidualBlock(ch_n * 4, act_type='prelu_n') for _ in range(1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):

    def __init__(self, ch_n):
        super().__init__()
        block_generator = get_block_list_generator([nn.ConvTranspose2d, nn.BatchNorm2d, nn.PReLU, nn.Conv2d, nn.BatchNorm2d])
        args = [[ch_n * 4, ch_n * 2, 2, 2], [ch_n * 2], [], [ch_n * 2, ch_n * 2, 3, 1, 1], [ch_n * 2]]
        kwargs = [{}, {}, {}, {}, {}]
        block = block_generator(args, kwargs)
        args = [[ch_n * 2, ch_n, 2, 2], [ch_n], [], [ch_n, ch_n, 3, 1, 1], [ch_n]]
        block += block_generator(args, kwargs)
        args = [[ch_n, ch_n // 2, 2, 2], [ch_n // 2], [], [ch_n // 2, ch_n // 2, 3, 1, 1], [ch_n // 2]]
        block += block_generator(args, kwargs)
        block += [nn.Conv2d(ch_n // 2, 1, 7, 1, 3), nn.Tanh()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = self.block(x)
        y = (y + torch.ones_like(y)) * 0.5
        return y
