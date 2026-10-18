import cv2
import random
import torch
import typing
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as tnf
from pathlib import Path
from PIL import features, Image
from scipy.ndimage import zoom
from typing import Union
import subprocess
from scipy.fft import fft2
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from scipy.ndimage import distance_transform_edt, generic_filter, geometric_transform, map_coordinates
from time import localtime, strftime
import copy
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import partial
from time import localtime, strftime, time
from tqdm import tqdm
import os
import time
def get_support_extensions():
    path = Path('pil_info.txt')
    with open(path, 'w') as f:
        features.pilinfo(f, supported_formats=True)
    with open(path, 'r') as f:
        info = f.read()
    path.unlink()
    info = info.replace(', ', ' ')
    ext = []
    for s1 in info.split('-'):
        if 'open' in s1:
            for s2 in s1.split('\n'):
                if 'Extensions:' in s2:
                    for s3 in s2.split(' '):
                        if '.' in s3:
                            ext.append(s3)
    return ext

def ima_float2uint(ima):
    if np.issubdtype(ima.dtype, np.integer):
        out = ima.astype(np.uint8).copy()
    else:
        out = np.round(ima * 255).astype(np.uint8).copy()
    return out

def ima_gaussian(ima, ksz, sigma=(0, 0), pad_md='reflect101'):
    if isinstance(ksz, int):
        spk = (ksz, ksz)
    else:
        spk = ksz
    if isinstance(sigma, (float, int)):
        sigma_x = sigma_y = sigma
    else:
        sigma_x, sigma_y = sigma
    bu = pad_md.upper()
    bu = 'cv2.BORDER_' + bu
    bu = eval(bu)
    out = cv2.GaussianBlur(ima, spk, sigmaX=sigma_x, sigmaY=sigma_y, borderType=bu)
    return out

def ima_minmax(ima, min_v=0, max_v=1):
    ima_min = ima.min(axis=(0, 1), keepdims=True)
    ima_max = ima.max(axis=(0, 1), keepdims=True)
    out = (ima - ima_min) / (ima_max - ima_min + 1e-09) * max_v + min_v
    return out

def ima_pad_crop(ima, sp, pad_md: typing.Any='reflect', **kwargs):
    pad_n = np.zeros((len(ima.shape), 2), dtype=np.int32)
    crop_n = []
    for i in range(len(sp)):
        d = ima.shape[i] - sp[i]
        if d < 0:
            v = -1 * d
            pad_n[i, 0] = v - v // 2
            pad_n[i, 1] = v // 2
            crop_n.append([0, 0])
        else:
            v = d
            pad_n[i, 0] = 0
            pad_n[i, 1] = 0
            crop_n.append([v - v // 2, v // 2])
    for i in range(len(ima.shape) - len(sp)):
        crop_n.append([0, 0])
    out = np.pad(ima, pad_n, pad_md, **kwargs)
    crop_slice = []
    for i in range(len(out.shape)):
        crop_slice.append(slice(crop_n[i][0], out.shape[i] - crop_n[i][1]))
    crop_slice = tuple(crop_slice)
    out = out[crop_slice]
    return out

def ima_read(im_p):
    if isinstance(im_p, str):
        path = Path(im_p)
    else:
        path = im_p
    im = Image.open(path)
    if im.mode not in ['L', 'RGB']:
        im = im.convert('RGB')
    ima = np.asarray(im, dtype=np.float32) / 255.0
    return ima

def ima_rgb2l(ima, md='a'):
    if ima.ndim == 3:
        if md == 'a':
            out = np.mean(ima, axis=2)
        elif md == 'm':
            out = np.min(ima, axis=2)
        elif md == 'c':
            out = np.median(ima, axis=2)
        elif md == 'M':
            out = np.max(ima, axis=2)
        elif md == 'r':
            out = ima[:, :, 0].squeeze().copy()
        elif md == 'g':
            out = ima[:, :, 1].squeeze().copy()
        elif md == 'b':
            out = ima[:, :, 2].squeeze().copy()
        else:
            raise Exception('Unknown Mode!')
    else:
        out = ima.copy()
    return out

def ima_save(ima_in, im_p, sz=None, rs_md='bicubic'):
    if isinstance(im_p, str):
        path = Path(im_p)
    else:
        path = im_p
    ima = np.clip(ima_in, 0.0, 1.0)
    ima = ima.squeeze()
    md = 'L'
    if ima.ndim == 3:
        md = 'RGB'
    ima = ima * 255
    ima = np.round(ima)
    ima = ima.astype(np.uint8)
    im = Image.fromarray(ima, mode=md)
    if sz is not None:
        assert len(sz) == 2
        im = im.resize(sz, resample=eval('Image.' + rs_md.upper()))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)

def ima_show(ima_in):
    ima = np.clip(ima_in, 0.0, 1.0)
    ima = ima.squeeze()
    md = 'L'
    if ima.ndim == 3:
        md = 'RGB'
    ima = ima * 255
    ima = np.round(ima)
    ima = ima.astype(np.uint8)
    im = Image.fromarray(ima, mode=md)
    im.show()

def ima_uint2float(ima):
    if np.issubdtype(ima.dtype, np.integer):
        out = ima.astype(np.float32).copy()
        out = out / 255.0
    else:
        out = ima.astype(np.float32).copy()
    return out

def ima2imt(ima, device='cpu'):
    imt = torch.from_numpy(ima)
    imt = imt.to(device)
    if imt.ndim == 3:
        imt = imt.permute(2, 0, 1)
    elif imt.ndim == 2:
        imt = imt.unsqueeze(0)
    return imt

def imt_cal_grad(imt, md='sobel', pad_md='reflect', pad_v=0, **kwargs):
    if md == 'sobel':
        conv_hk = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        conv_wk = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        pd_sp = (2, 2)
    elif md == 'scharr':
        conv_hk = torch.tensor([[-3.0, -10.0, -3], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
        conv_wk = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
        pd_sp = (2, 2)
    else:
        raise Exception('Unknown Mode!')
    im = imt_pad_crop(imt, (imt.shape[-2] + pd_sp[-2], imt.shape[-1] + pd_sp[-1]), pad_md, pad_v)
    out_h = imt_conv2d(im, conv_hk, **kwargs)
    out_w = imt_conv2d(im, conv_wk, **kwargs)
    return (out_h, out_w)

def imt_conv2d(imt, kt, s=1, **kwargs):
    c = imt.shape[-3]
    im = imt.unsqueeze(0).clone()
    if kt.ndim == 2:
        g = c
        k = kt.unsqueeze(0).clone()
        k = k.unsqueeze(0)
        if c != 1:
            k = torch.cat([k for _ in range(c)])
    elif kt.ndim == 3 and kt.shape[-3] == c:
        g = 1
        k = kt.unsqueeze(0).clone()
    else:
        raise Exception('Error of the kernel size!')
    k = k.type(im.type())
    k = k.to(im.device)
    out = torch.nn.functional.conv2d(im, k, groups=g, stride=s, **kwargs)
    out = out.squeeze()
    if out.ndim == 2:
        out = out.unsqueeze(0)
    return out

def imt_pad_blk(imt, blk_sz, pad_md='reflect', pad_v=0):
    if isinstance(blk_sz, int):
        bh = bw = blk_sz
    else:
        bh, bw = blk_sz
    h, w = imt.shape[-2:]
    ph = round(np.ceil(h / bh) * bh)
    pw = round(np.ceil(w / bw) * bw)
    if ph == h and pw == w:
        out = imt.clone()
    else:
        out = imt_pad_crop(imt, (ph, pw), pad_md=pad_md, pad_v=pad_v)
    return (out, imt.shape[-2:])

def imt_pad_crop(imt, sp, pad_md='reflect', pad_v=0):
    pad_n = []
    crop_n = []
    for i in range(len(sp)):
        n = int(-1 * (i + 1))
        d = imt.shape[n] - sp[n]
        if d < 0:
            v = -1 * d
            pad_n += [v - v // 2, v // 2]
            crop_n.append([0, 0])
        else:
            v = d
            pad_n += [0, 0]
            crop_n.append([v - v // 2, v // 2])
    pad_n = tuple(pad_n)
    for i in range(len(imt.shape) - len(sp)):
        crop_n.append([0, 0])
    crop_n = crop_n[-1::-1]
    out = torch.nn.functional.pad(imt, pad_n, mode=pad_md, value=pad_v)
    crop_slice = []
    for i in range(len(out.shape)):
        crop_slice.append(slice(crop_n[i][0], out.shape[i] - crop_n[i][1]))
    out = out[tuple(crop_slice)]
    return out

def imt_sum_average(imt, ksz, kmd='m', s=1, pad_md='reflect', pad_v=0, keep_shape=True, **kwargs):
    if isinstance(ksz, int):
        spk = (ksz, ksz)
    else:
        spk = ksz
    conv_k = torch.ones(spk)
    if kmd == 'm':
        conv_k = conv_k / conv_k.numel()
    if isinstance(s, int):
        sps = (s, s)
    else:
        sps = s
    im, imt_sp = imt_pad_blk(imt, sps, pad_md, pad_v)
    pd_sp = (spk[-2] - sps[-2], spk[-1] - sps[-1])
    im = imt_pad_crop(im, (im.shape[-2] + pd_sp[-2], imt.shape[-1] + pd_sp[-1]), pad_md, pad_v)
    out = imt_conv2d(im, conv_k, s=s, **kwargs)
    if keep_shape:
        out = out.repeat_interleave(sps[-1], -1).repeat_interleave(sps[-2], -2)
        out = imt_pad_crop(out, imt_sp)
    return out

def imt2ima(imt_in):
    imt = imt_in.squeeze().clone()
    if imt.ndim == 3:
        imt = imt.permute(1, 2, 0)
    imt = imt.to('cpu')
    ima = imt.numpy()
    return ima

def type_as(im, md='t', device=None) -> Union[np.ndarray, torch.Tensor]:
    if not isinstance(im, (np.ndarray, torch.Tensor)):
        raise Exception('Type Error! The type of im should be numpy.ndarray or torch.Tensor!')
    if isinstance(md, str):
        if md == 't':
            if isinstance(im, np.ndarray):
                if not device:
                    out = ima2imt(im, 'cpu')
                else:
                    out = ima2imt(im, device)
            elif not device:
                out = im.clone()
            else:
                out = im.to(device).clone()
        elif md == 'a':
            if isinstance(im, torch.Tensor):
                out = imt2ima(im)
            else:
                out = im.copy()
        else:
            raise Exception('Unknown Command!')
    else:
        if not isinstance(md, (np.ndarray, torch.Tensor)):
            raise Exception('Type Error! The type of md should be numpy.ndarray or torch.Tensor!')
        if isinstance(im, np.ndarray) and isinstance(md, torch.Tensor):
            out = ima2imt(im, md.device)
        elif isinstance(im, torch.Tensor) and isinstance(md, np.ndarray):
            out = imt2ima(im)
        elif isinstance(im, np.ndarray):
            out = im.copy()
        elif isinstance(im, torch.Tensor):
            out = im.clone()
        else:
            raise Exception('Error!')
    return out

def diffusion_filter(im, ot, c1, c2, s, itn, mk=None):
    ima = type_as(im, md='a')
    ima = ima_rgb2l(ima)
    ima = ima_minmax(ima)
    ota = type_as(ot, md='a')
    if mk is None:
        mka = np.ones_like(ota)
    else:
        mka = mk
    ft, filter_func = get_diffusion_function(ota, c1, c2, s, mk=mka)
    for i in range(itn):
        ima = generic_filter(ima, filter_func, footprint=ft)
    ima = type_as(ima, im)
    return ima

def display_transform(sp, index, blk_sz=(16, 16), g_color=(1.0, 0, 0)):
    if isinstance(blk_sz, (float, int)):
        bsp = (blk_sz, blk_sz)
    else:
        bsp = blk_sz
    if isinstance(g_color, (float, int)):
        grid_color = (g_color, g_color, g_color)
    else:
        grid_color = g_color
    grid_im = np.ones((sp[0], sp[1], 3))
    for i in range(sp[0]):
        for j in range(sp[1]):
            if i % bsp[0] == 0 or j % bsp[1] == 0:
                grid_im[i, j, :] = grid_color
    out = np.zeros_like(grid_im)
    for i in range(grid_im.shape[2]):
        out[:, :, i] = map_coordinates(grid_im[:, :, i], [index[0], index[1]], cval=1.0)
    ima_show(out)

def distortion_cappelli2001(im, skin_k=3, affine_p=None, region_p=None, pad_md='constant', pad_v=0.0, display_transform_flag=False, display_out_flag=False):
    ima = type_as(im, md='a')
    h, w = ima.shape[:2]
    if affine_p is None:
        affine_dict = {'td_h': 15, 'td_w': 15, 'rt_ch': h - h // 2, 'rt_cw': w - w // 2, 'rt_ag': 15}
    else:
        affine_dict = affine_p
    affine_dict['td_h'] = -affine_dict['td_h']
    affine_dict['td_w'] = -affine_dict['td_w']
    affine_dict['rt_ag'] = affine_dict['rt_ag'] / 180 * np.pi
    if region_p is None:
        region_dict = {'ch': h - h // 2, 'cw': w - w // 2, 'ra': h // 12, 'rb': w // 16}
    else:
        region_dict = region_p
    ar_td = np.array([[affine_dict['td_h']], [affine_dict['td_w']]])
    ar_rc = np.array([[affine_dict['rt_ch']], [affine_dict['rt_cw']]])
    cos, sin = (np.cos(affine_dict['rt_ag']), np.sin(affine_dict['rt_ag']))
    ar_r = np.array([[cos, sin], [-sin, cos]])
    ar_ac = np.array([[region_dict['ch']], [region_dict['cw']]])
    ar_ai = np.array([[region_dict['ra'] ** (-2), 0], [0, region_dict['rb'] ** (-2)]])
    grid = np.mgrid[0:h, 0:w].reshape(2, -1)
    dist = np.sqrt(np.sum((grid - ar_ac).T @ ar_ai * (grid - ar_ac).T, 1)) - 1
    dist[dist <= 0] = 0
    dist = np.where(dist <= skin_k, 0.5 * (1 - np.cos(dist * np.pi / skin_k)), np.ones_like(dist))
    grid = grid + (ar_r @ (grid - ar_rc) + ar_rc + ar_td - grid) * dist
    grid = grid.reshape(2, h, -1)
    if display_transform_flag:
        display_transform((h, w), index=grid)
    if ima.ndim == 2:
        out = map_coordinates(ima, grid, mode=pad_md, cval=pad_v)
        if display_out_flag:
            ima_show(out)
    else:
        out = np.zeros_like(ima)
        for i in range(ima.shape[2]):
            pv = pad_v
            if pad_md == 'constant':
                if isinstance(pad_v, (list, tuple, np.ndarray)):
                    pv = pad_v[i]
            out[:, :, i] = map_coordinates(ima[:, :, i], grid, mode=pad_md, cval=pv)
            if display_out_flag:
                ima_show(out[:, :, i])
    out = type_as(out, im)
    return out

def get_diffusion_function(oa, c1, c2, s, mk=None):
    if mk is None:
        mka = np.ones_like(oa)
    else:
        mka = mk
    ft = np.ones((3, 3))
    k = []
    for i in range(oa.size):
        coordinates = np.unravel_index(i, oa.shape)
        if mka[coordinates] == 0:
            k.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        else:
            o = oa[coordinates]
            cos, sin = (np.cos(o), np.sin(o))
            a = (c1 - c2) * cos * cos + c2
            b = (c1 - c2) * cos * sin
            c = (c1 - c2) * sin * sin + c2
            k.append(np.array([b, 2 * a, -b, 2 * c, -4 * (c1 + c2), 2 * c, -b, 2 * a, b]) * s / 2)

    class FilterFunction:

        def __init__(self):
            self.n = 0

        def __call__(self, in_elements):
            out = in_elements[4] + (k[self.n] * in_elements).sum()
            self.n = (self.n + 1) % oa.size
            return out
    return (ft, FilterFunction())

def lbl2msk(lb, convex_flag=False, polydp_flag=False, smooth_flag=False):
    lba = type_as(lb, 'a')
    lba = np.where(lba > 0.5, np.ones_like(lba), np.zeros_like(lba))
    inv_flag = lba[:10, :10].mean() > 0.5
    if inv_flag:
        lba = 1 - lba
    lba = ima_float2uint(lba)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mka = cv2.morphologyEx(lba, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mka, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        pass
    else:
        assert len(contours) >= 2
        print(f'轮廓数为{len(contours)}！')
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        areas = [cv2.contourArea(c) for c in contours[1:]]
        areas = areas
        for area in areas:
            print(area)
            rd_nm = f"{strftime('%m%d%H%M%S', localtime())}"
            ima_save(lba, f'temp_result/{rd_nm}_{area}_lba.png')
            ima_save(mka, f'temp_result/{rd_nm}_{area}_mka.png')
    contour = contours[0]
    if convex_flag:
        approx = cv2.convexHull(contour)
    elif polydp_flag:
        approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
    else:
        approx = contour
    mka = np.zeros_like(mka)
    cv2.drawContours(mka, [approx], -1, 255, -1)
    if smooth_flag:
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mka = cv2.morphologyEx(mka, cv2.MORPH_CLOSE, smooth_kernel)
    mka = mka.astype(np.float32)
    mka = np.where(mka > 0.0, np.ones_like(mka), np.zeros_like(mka))
    return type_as(mka, lb)

def noise_cappelli2004(mk, tb=20, pl=0.1, k=3, o=3, p=0.5, la=2, freq=16, edge_noise=True):
    mka = type_as(mk, md='a')
    dta = distance_transform_edt(mka)
    dta[dta > tb] = 0
    dta = np.where(dta > 0, (tb - dta) / tb, dta)
    if edge_noise:
        npa = pl * (1 + dta ** 3)
    else:
        npa = pl * (1 - dta ** 3)
    grid_h = np.arange(mka.shape[0]) / (o * freq)
    grid_w = np.arange(mka.shape[1]) / (o * freq)
    noise = perlin2d_octave(grid_h, grid_w, o, p, la)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    npa = npa * (1 + noise ** k)
    return type_as(npa, mk)

def ort_rao1990(im, blk_sz, wd_sz=(5, 5), device=None, need_quality=True):
    if isinstance(blk_sz, int):
        ksp = (blk_sz, blk_sz)
    else:
        ksp = blk_sz
    if isinstance(wd_sz, int):
        wsp = (wd_sz, wd_sz)
    else:
        wsp = wd_sz
    imt = type_as(im, md='t', device=device)
    hg, wg = imt_cal_grad(imt)
    vy = imt_sum_average(2 * hg * wg, ksp, kmd='s')
    vx = imt_sum_average(hg * hg - wg * wg, ksp, kmd='s')
    out = 0.5 * torch.atan2(vy, vx)
    if not need_quality:
        out = type_as(out, im)
        return out
    else:
        cost, sint = (torch.cos(out), torch.sin(out))
        g = hg ** 2 + wg ** 2
        g_cos = imt_sum_average(g * cost * cost, wsp, kmd='s')
        g_sin = imt_sum_average(g * sint * sint, wsp, kmd='s')
        g_2cs = imt_sum_average(2 * g * sint * cost, wsp, kmd='s')
        sum_g = imt_sum_average(g, wsp, kmd='s')
        cd = (cost * cost * g_cos + sint * sint * g_sin + cost * sint * g_2cs) / (sum_g + 1e-08)
        out = type_as(out, im)
        cd = type_as(cd, im)
        return (out, cd)

def perlin2d(h, w):
    assert len(h.shape) == len(w.shape) == 2
    assert h.shape[0] == w.shape[0]
    assert h.shape[1] == w.shape[1]
    n = max(h.shape[0], h.shape[1])
    ptable = np.arange(n, dtype=int)
    np.random.shuffle(ptable)
    ptable = np.stack([ptable, ptable]).flatten()
    hi, wi = (h.astype(int), w.astype(int))
    id00 = ptable[ptable[hi % n] + wi % n] % 4
    id01 = ptable[ptable[hi % n] + wi % n + 1] % 4
    id10 = ptable[ptable[hi % n + 1] + wi % n] % 4
    id11 = ptable[ptable[hi % n + 1] + wi % n + 1] % 4
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g00 = vectors[id00]
    g01 = vectors[id01]
    g10 = vectors[id10]
    g11 = vectors[id11]
    hf0, wf0 = (h - hi, w - wi)
    hf1 = 6 * hf0 ** 5 - 15 * hf0 ** 4 + 10 * hf0 ** 3
    wf1 = 6 * wf0 ** 5 - 15 * wf0 ** 4 + 10 * wf0 ** 3
    c00 = g00[:, :, 0] * hf0 + g00[:, :, 1] * wf0
    c01 = g01[:, :, 0] * hf0 + g01[:, :, 1] * (wf0 - 1)
    c10 = g10[:, :, 0] * (hf0 - 1) + g10[:, :, 1] * wf0
    c11 = g11[:, :, 0] * (hf0 - 1) + g11[:, :, 1] * (wf0 - 1)
    y1 = c00 + hf1 * (c10 - c00)
    y2 = c01 + hf1 * (c11 - c01)
    out = y1 + wf1 * (y2 - y1)
    return out

def perlin2d_octave(grid_h, grid_w, o=1, p=0.5, la=2, need_norm=True):
    ind_h, ind_w = np.meshgrid(grid_h, grid_w, indexing='ij')
    noise = np.zeros(ind_h.shape)
    max_v = 0
    for i in range(o):
        noise += perlin2d(ind_h * la ** i, ind_w * la ** i) * p ** i
        max_v += p ** i
    if need_norm:
        return noise / max_v
    else:
        return noise

def mask_mnt(mt, mk, distance=0):
    out = copy.deepcopy(mt)
    mka = type_as(mk, 'a')
    if distance > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * distance + 1, 2 * distance + 1))
        mka = ima_uint2float(cv2.erode(ima_float2uint(mka), kernel))
    for i in range(mt['ind_hw'].shape[1]):
        ind_h, ind_w = (mt['ind_hw'][0, i], mt['ind_hw'][1, i])
        if abs(mka[ind_h, ind_w]) <= 0.0001:
            out['quality'][i] = -1
    return select_mnt(out, 0)

def mnt2map(mt, th_q=0.2, fix_v=1.0, use_quality=False, filter_fn=None):
    if filter_fn is None:
        filter_fn = partial(ima_gaussian, ksz=35, sigma=8)
    mnt = select_mnt(mt, th_q)
    mta = np.zeros(mnt['shape'])
    if use_quality:
        mta[mnt['ind_hw'][0], mnt['ind_hw'][1]] = mnt['quality']
    else:
        mta[mnt['ind_hw'][0], mnt['ind_hw'][1]] = fix_v
    out = filter_fn(mta)
    return out

def select_mnt(mt, th_q):
    q = mt['quality']
    flag = q >= th_q
    ind_h = mt['ind_hw'][0][flag]
    ind_w = mt['ind_hw'][1][flag]
    angle = mt['angle'][flag]
    quality = mt['quality'][flag]
    shape = mt['shape']
    out = {'ind_hw': np.array([ind_h, ind_w], dtype=np.int32), 'angle': angle.copy(), 'quality': quality.copy(), 'shape': shape}
    return out

def xyt2array(xyt_d, sp, inverse_h=True, inverse_w=False, rotate_t=0):
    if isinstance(sp, int):
        h, w = (sp, sp)
    else:
        h, w = sp
    with open(xyt_d, 'r') as xyt_f:
        mts = xyt_f.readlines()
    ind_h, ind_w, angle, quality = ([], [], [], [])
    for i in range(len(mts)):
        mt = mts[i].strip().split()
        if inverse_h:
            ind_h.append(h - int(float(mt[1])))
        else:
            ind_h.append(int(float(mt[1])))
        if inverse_w:
            ind_w.append(w - int(float(mt[2])))
        else:
            ind_w.append(int(float(mt[0])))
        angle.append(int(float(mt[2]) + rotate_t) / 180.0 * np.pi)
        if len(mt) == 3:
            quality.append(1.0)
        else:
            quality.append(int(float(mt[3])) / 100.0)
    out = {'ind_hw': np.array([ind_h, ind_w], dtype=np.int32), 'angle': np.array(angle, dtype=np.float32), 'quality': np.array(quality, dtype=np.float32), 'shape': (h, w)}
    return out

def get_data_dirs(data_folder_path, data_dirs, ext=None, top_flag=True, specified_content=None):
    if not ext:
        ext = get_support_extensions()
    if isinstance(data_folder_path, str):
        data_folder_p = Path(data_folder_path)
    else:
        data_folder_p = data_folder_path
    for path in data_folder_p.iterdir():
        if path.is_dir():
            data_dirs = get_data_dirs(path, data_dirs, ext=ext, top_flag=False)
        elif path.suffix in ext:
            data_dirs.append(str(path.absolute()))
    if top_flag:
        if specified_content:
            temp = []
            for d in data_dirs:
                if specified_content in d:
                    temp.append(d)
            data_dirs = temp
        data_dirs = sorted(data_dirs)
    return data_dirs

def set_random_seed(seed, deterministic_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
