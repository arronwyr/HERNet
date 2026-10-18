import torch
import torch.nn as nn
from functools import partial
class PatchEmbedV2(nn.Module):

    def __init__(self, in_chans, embed_dim, img_sz, patch_size=(1, 1), norm_layer=None, pos_embed_flag=True, drop_ratio=0.0):
        super().__init__()
        if isinstance(img_sz, int):
            img_sz = (img_sz, img_sz)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert img_sz[0] % patch_size[0] == 0
        assert img_sz[1] % patch_size[1] == 0
        self.img_sz = img_sz
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.pos_embed_flag = pos_embed_flag
        if self.pos_embed_flag:
            self.pos_embed = nn.Parameter(torch.randn(1, img_sz[0] // patch_size[0] * (img_sz[1] // patch_size[1]), embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_drop = nn.Dropout(p=drop_ratio)

    def interpolate_pos_embed(self, h, w):
        train_h = self.img_sz[0] // self.patch_size[0]
        train_w = self.img_sz[1] // self.patch_size[1]
        if h == train_h and w == train_w:
            return self.pos_embed
        pos_embed = self.pos_embed.reshape(1, train_h, train_w, -1).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pos_embed

    def forward(self, x):
        input_sz = x.shape[-2:]
        if self.training:
            assert input_sz[0] == self.img_sz[0]
            assert input_sz[1] == self.img_sz[1]
            x = self.proj(x).flatten(2).transpose(1, 2)
            x = self.norm(x)
            if self.pos_embed_flag:
                x = x + self.pos_embed
                x = self.pos_drop(x)
            return x
        else:
            assert input_sz[0] % self.patch_size[0] == 0
            assert input_sz[1] % self.patch_size[1] == 0
            x = self.proj(x).flatten(2).transpose(1, 2)
            x = self.norm(x)
            if self.pos_embed_flag:
                grid_h = input_sz[0] // self.patch_size[0]
                grid_w = input_sz[1] // self.patch_size[1]
                x = x + self.interpolate_pos_embed(grid_h, grid_w)
                x = self.pos_drop(x)
            return x

class PatchUnEmbedV2(nn.Module):

    def __init__(self, out_chans, embed_dim, img_sz, patch_size=(1, 1)):
        super().__init__()
        if isinstance(img_sz, int):
            img_sz = (img_sz, img_sz)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert img_sz[0] % patch_size[0] == 0
        assert img_sz[1] % patch_size[1] == 0
        self.img_sz = img_sz
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, input_sz=None):
        b, n, d = x.shape
        if self.training:
            assert input_sz is None
            assert n == self.img_sz[0] // self.patch_size[0] * (self.img_sz[1] // self.patch_size[1])
            x = x.transpose(1, 2).view(b, d, self.img_sz[0] // self.patch_size[0], self.img_sz[1] // self.patch_size[1])
            x = self.proj(x)
            return x
        else:
            assert input_sz is not None
            h, w = input_sz
            assert h % self.patch_size[0] == 0
            assert w % self.patch_size[1] == 0
            assert n == h // self.patch_size[0] * (w // self.patch_size[1])
            x = x.transpose(1, 2).view(b, d, h // self.patch_size[0], w // self.patch_size[1])
            x = self.proj(x)
            return x

class MHA(nn.Module):

    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.0, proj_drop_ratio=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale if qk_scale else head_dim ** (-0.5)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features if out_features else in_features
        hidden_features = hidden_features if hidden_features else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask

class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0, act_layer=None, norm_layer=None):
        super().__init__()
        act_layer = act_layer if act_layer else nn.GELU
        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, eps=1e-06)
        self.mha_norm = norm_layer(embed_dim)
        self.mha = MHA(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.mlp_norm = norm_layer(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_features, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.mha(self.mha_norm(x)))
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x

class TransformerBlockV2(nn.Module):

    def __init__(self, in_chans=256, embed_dim=256, img_sz=(16, 16), patch_size=(1, 1), pos_embed_flag=True, out_chans=256, encoder_n=6, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0, act_layer=None, norm_layer=None, final_norm=True):
        super().__init__()
        act_layer = act_layer if act_layer else nn.GELU
        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbedV2(in_chans=in_chans, embed_dim=embed_dim, img_sz=img_sz, patch_size=patch_size, pos_embed_flag=pos_embed_flag, drop_ratio=drop_ratio)
        encoder_blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, encoder_n)]
        for i in range(encoder_n):
            encoder_block = EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            encoder_blocks.append(encoder_block)
        self.encoder = nn.Sequential(*encoder_blocks)
        if final_norm:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()
        self.patch_un_embed = PatchUnEmbedV2(out_chans=out_chans, embed_dim=embed_dim, img_sz=img_sz, patch_size=patch_size)

    def forward(self, x):
        input_sz = x.shape[-2:] if not self.training else None
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.patch_un_embed(x, input_sz)
        return x
