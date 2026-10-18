from _utils import *
from nbis import extract_minutiae
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from scipy.ndimage import grey_dilation, grey_erosion, rotate, zoom
from time import localtime, strftime, time
from tqdm import tqdm

def main():
    start_time_str = strftime('%Y%m%d%H%M%S', localtime())
    option_parser = ArgumentParser()
    option_parser.add_argument('--ims_dir', type=str, required=True, help='Source root containing matching image/ and label/ trees')
    option_parser.add_argument('--txs_dir', type=str, required=True, help='Background texture directory')
    option_parser.add_argument('--out_dir', type=str, required=True, help='Parent output directory')
    option_parser.add_argument('--out_num', type=int, default=40000, help='Number of generated pairs')
    option_parser.add_argument('--start_n', type=int, default=0, help='Output repeat-index offset')
    option_parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    option_parser.add_argument('--q_th', type=float, default=0.25, help='MINDTCT supervision quality threshold')
    option_parser.add_argument('--mindtct', required=True, help='Path to the NBIS MINDTCT executable')
    opt = option_parser.parse_args()
    quality_threshold = opt.q_th
    set_random_seed(opt.random_seed)
    ims_path = Path(opt.ims_dir).resolve()
    ims_dir = str(ims_path.absolute())
    dt_dirs = []
    dt_dirs = get_data_dirs(ims_path, dt_dirs)
    im_dirs = []
    for d in dt_dirs:
        rf_d = d.replace(ims_dir, '')
        if 'image' in rf_d:
            im_dirs.append(d)
    im_num = len(im_dirs)
    if not im_num:
        raise ValueError('No source images found; use matching image/ and label/ trees.')
    txs_path = Path(opt.txs_dir).resolve()
    tx_dirs = []
    tx_dirs = get_data_dirs(txs_path, tx_dirs)
    np.random.shuffle(tx_dirs)
    tx_num = len(tx_dirs)
    if not tx_num:
        raise ValueError('No background textures found.')
    res_path = Path(opt.out_dir).resolve()
    res_path = res_path / f'generated_{start_time_str}'
    res_path.mkdir(parents=True, exist_ok=False)
    res_dir = str(res_path.absolute())
    if opt.out_num <= 0:
        out_num = im_num
    else:
        out_num = opt.out_num
    with tqdm(total=out_num) as pbar:
        pbar.set_description('生成指纹中')
        for ii in range(out_num):
            start_time = time()
            im_d = im_dirs[ii % im_num]
            relative = Path(im_d).relative_to(ims_path / 'image')
            lb_d = str(ims_path / 'label' / relative)
            tx_d = tx_dirs[ii % tx_num]
            relative = relative.with_name(f'{relative.stem}_{ii // im_num + opt.start_n}{relative.suffix}')
            save_img_d = str(res_path / 'image' / relative)
            save_lbl_d = str(res_path / 'label' / relative)
            save_msk_d = str(res_path / 'mask' / relative)
            save_ort_d = str(res_path / 'orientation' / relative.with_suffix('.npy'))
            Path(save_ort_d).parent.mkdir(parents=True, exist_ok=True)
            save_mnt_d = str(res_path / 'minutia' / relative.with_suffix('.npy'))
            Path(save_mnt_d).parent.mkdir(parents=True, exist_ok=True)
            save_inf_d = str(res_path / 'information' / relative)
            save_val_d = str(res_path / 'value' / relative.with_suffix('.txt'))
            Path(save_val_d).parent.mkdir(parents=True, exist_ok=True)
            im = ima_rgb2l(ima_read(im_d))
            lb = ima_rgb2l(ima_read(lb_d))
            tx = ima_rgb2l(ima_read(tx_d))
            if tx.shape != im.shape:
                tx = cv2.resize(tx, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            rg = np.stack((np.zeros_like(im), np.zeros_like(im), np.zeros_like(im)), axis=2)
            rdv = {}
            fxv = {}
            h, w = im.shape
            rdv['rt'] = np.random.randint(0, 360)
            rdv['scar_n'] = np.random.randint(0, 5)
            fxv['ed_p'] = 0.6
            fxv['ed_h'] = np.random.randint(1, 4)
            fxv['ed_w'] = np.random.randint(1, 4)
            fxv['wd_p'] = 0.5
            fxv['pn_p'] = 0.5
            rdv['wd_f'] = np.random.randint(9, 27)
            rdv['wd_t'] = np.random.uniform(1.2, 1.5)
            fxv['perlin_p'] = np.random.uniform(0.04, 0.1)
            rdv['perlin_f'] = np.random.randint(16, 64)
            fxv['gd_n'] = 5
            rdv['ld_f'] = np.random.randint(32, 64)
            rdv['ld_t'] = np.random.uniform(0.98, 1.02)
            rdv['ld_n'] = np.random.randint(3, 9)
            rdv['ld_ag'] = np.random.uniform(-np.pi / 2, np.pi / 2)
            fxv['gb_n'] = 3
            rdv['lb_f'] = np.random.randint(16, 48)
            rdv['lb_t'] = np.random.uniform(0.98, 1.02)
            rdv['lb_n'] = np.random.randint(5, 12)
            rdv['dt_ra'] = np.random.randint(32, 64)
            if 45 < rdv['rt'] <= 135 or 225 < rdv['rt'] <= 315:
                rdv['dt_rb'] = np.random.randint(rdv['dt_ra'] - 1, 64)
            else:
                rdv['dt_rb'] = np.random.randint(32, rdv['dt_ra'] + 1)
            fxv['dt_k'] = 3
            temp = int(fxv['dt_k'] * rdv['dt_ra'] / 2 ** 2.5)
            rdv['dt_th'] = np.random.randint(-temp, temp)
            temp = int(fxv['dt_k'] * rdv['dt_rb'] / 2 ** 2.5)
            rdv['dt_tw'] = np.random.randint(-temp, temp)
            rdv['dt_ag'] = np.random.randint(-25, 25)
            rdv['crop_ra'] = np.random.uniform(0.25, 0.38) * h
            rdv['crop_rb'] = np.random.uniform(0.25, 0.38) * w
            rdv['crop_ag'] = np.random.randint(0, 360)
            rdv['mix_c'] = np.random.uniform(0.35, 0.65)
            rdv['transpose'] = np.random.uniform()
            rdv['fliplr'] = np.random.uniform()
            rdv['flipud'] = np.random.uniform()
            ima = im
            lba = lb
            if rdv['transpose'] < 0.5:
                ima = np.transpose(ima)
                lba = np.transpose(lba)
            if rdv['fliplr'] < 0.5:
                ima = np.fliplr(ima)
                lba = np.fliplr(lba)
            if rdv['flipud'] < 0.5:
                ima = np.flipud(ima)
                lba = np.flipud(lba)
            ima = rotate(ima, rdv['rt'], reshape=True, mode='constant', cval=1.0)
            lba = rotate(lba, rdv['rt'], reshape=True, mode='constant', cval=1.0)
            s_h, s_w = (im.shape[0] / ima.shape[0], im.shape[1] / ima.shape[1])
            ima = zoom(ima, (s_h, s_w), mode='constant', cval=1.0)
            lba = zoom(lba, (s_h, s_w), mode='constant', cval=1.0)
            ima = ima_pad_crop(ima, im.shape, pad_md='constant', constant_values=1.0)
            lba = ima_pad_crop(lba, im.shape, pad_md='constant', constant_values=1.0)
            rdv['scar_p'] = []
            for i in range(rdv['scar_n']):
                temp1 = [np.random.randint(w // 3, w - w // 3), np.random.randint(h // 3, h - h // 3)]
                temp2 = [np.random.randint(16, 64), np.random.randint(1, 3)]
                temp3 = np.random.randint(0, 360)
                cv2.ellipse(ima, temp1, temp2, temp3, 0.0, 360.0, [1.0], -1)
                rdv['scar_p'].append([temp1, temp2, temp3])
            ota = ort_rao1990(lba, (16, 16))[0]
            mka = lbl2msk(lba)
            rdv['ed_p_v'] = np.random.random()
            rdv['wd_p_v'] = np.random.random()
            rdv['pn_p_v'] = np.random.random()
            rdv['ed_wd_pn_p'] = np.random.random()
            if rdv['ed_p_v'] < fxv['ed_p']:
                if rdv['ed_wd_pn_p'] < 0.5:
                    ima = grey_erosion(ima, size=(fxv['ed_h'], fxv['ed_w']))
                else:
                    ima = grey_dilation(ima, size=(fxv['ed_h'], fxv['ed_w']))
            if rdv['wd_p_v'] < fxv['wd_p']:
                rga = noise_cappelli2004(mka, tb=50, freq=rdv['wd_f'], edge_noise=False)
                rga = np.where(rga >= rga.mean() * rdv['wd_t'], np.ones_like(rga), np.zeros_like(rga))
                rga = rga * mka
                if rdv['ed_wd_pn_p'] < 0.5:
                    ima[rga == 1] = 0.0
                else:
                    ima[rga == 1] = 1.0
                rg[:, :, 0] = rga
            if rdv['pn_p_v'] < fxv['pn_p']:
                pna = noise_cappelli2004(mka, pl=fxv['perlin_p'], freq=rdv['perlin_f'])
                if rdv['ed_wd_pn_p'] < 0.5:
                    ima = np.clip(ima - pna, 0, 1)
                else:
                    ima = np.clip(ima + pna, 0, 1)
                rg[:, :, 1] = pna
            rdv['df_p_v'] = np.random.random()
            rdv['df_p_v2'] = np.random.random()
            if rdv['df_p_v'] <= 0.5:
                if rdv['df_p_v2'] >= 0.6:
                    ima = diffusion_filter(ima, ota, 0.9, 0.1, 0.5, fxv['gd_n'])
                elif 0.4 <= rdv['df_p_v2'] < 0.6:
                    ima = diffusion_filter(ima, ota + rdv['ld_ag'], 0.9, 0.1, 0.5, fxv['gd_n'])
                else:
                    ima = diffusion_filter(ima, rdv['ld_ag'] * np.ones_like(ota), 0.9, 0.1, 0.5, fxv['gd_n'])
                rga = noise_cappelli2004(mka, freq=rdv['ld_f'])
                rga = np.where(rga >= rga.mean() * rdv['ld_t'], np.ones_like(rga), np.zeros_like(rga))
                if rdv['df_p_v2'] >= 0.6:
                    ima = diffusion_filter(ima, ota, 0.9, 0.1, 0.5, rdv['ld_n'], rga)
                    rg[:, :, 2] = rga
                elif 0.4 <= rdv['df_p_v2'] < 0.6:
                    ima = diffusion_filter(ima, ota + rdv['ld_ag'], 0.9, 0.1, 0.5, rdv['ld_n'], rga)
                    rg[:, :, 2] = rga
                else:
                    ima = diffusion_filter(ima, rdv['ld_ag'] * np.ones_like(ota), 0.9, 0.1, 0.5, rdv['ld_n'], rga)
                    rg[:, :, 2] = rga
            else:
                for i in range(fxv['gb_n']):
                    ima = ima_gaussian(ima, (5, 5))
                rga = noise_cappelli2004(mka, freq=rdv['lb_f'])
                rga = np.where(rga >= rga.mean() * rdv['lb_t'], np.ones_like(rga), np.zeros_like(rga))
                rg[:, :, 2] = rga
                temp = ima.copy()
                for i in range(rdv['lb_n']):
                    temp = ima_gaussian(temp, (5, 5))
                ima = np.where(rga > 0, temp, ima)
            mean_v = ima[mka < 0.5].mean()
            skin_k = fxv['dt_k']
            affine_p = {'td_h': rdv['dt_th'], 'td_w': rdv['dt_tw'], 'rt_ch': h - h // 2, 'rt_cw': w - w // 2, 'rt_ag': rdv['dt_ag']}
            region_p = {'ch': h - h // 2, 'cw': w - w // 2, 'ra': rdv['dt_ra'], 'rb': rdv['dt_rb']}
            temp = distortion_cappelli2001(np.dstack((ima, lba, rg)), skin_k, affine_p, region_p, pad_v=(mean_v, 1.0, 0.0, 0.0, 0.0))
            ima = temp[:, :, 0]
            lba = temp[:, :, 1]
            rga = temp[:, :, 2:]
            mka = np.zeros(ima.shape, dtype=np.float32)
            cv2.ellipse(mka, [w // 2, h // 2], [int(rdv['crop_rb']), int(rdv['crop_ra'])], rdv['crop_ag'], 0.0, 360.0, [1.0], -1)
            ota = ort_rao1990(lba, (16, 16))[0]
            temp_mka = lbl2msk(lba)
            mka = mka * temp_mka
            mka = np.where(mka > 0, np.ones_like(mka), np.zeros_like(mka))
            ima = np.where(mka, ima * mka, mean_v * np.ones_like(ima))
            ima = ima_gaussian(ima, (5, 5))
            rga = np.clip(rga * np.expand_dims(mka, axis=2), 0.0, 1.0)
            ota = ota * mka
            inv_flag = lba[:10, :10].mean() > 0.5
            if inv_flag:
                ima_save(lba, Path(save_lbl_d))
                temp = 1 - lba
                temp = temp * mka
                lba = 1 - temp
            else:
                ima_save(1 - lba, Path(save_lbl_d))
                lba = lba * mka
            extract_minutiae(save_lbl_d, save_inf_d, opt.mindtct)
            mt_d = str(Path(save_inf_d).with_suffix('.xyt'))
            mnt = xyt2array(mt_d, lba.shape[:2])
            mnt = select_mnt(mnt, quality_threshold)
            mnt = mask_mnt(mnt, mka, distance=16)
            mta = mnt2map(mnt, th_q=quality_threshold)
            rdv['tx_transpose'] = np.random.uniform()
            rdv['tx_fliplr'] = np.random.uniform()
            rdv['tx_flipud'] = np.random.uniform()
            txa = tx
            if rdv['tx_transpose'] < 0.5:
                txa = np.transpose(txa)
            if rdv['tx_fliplr'] < 0.5:
                txa = np.fliplr(txa)
            if rdv['tx_flipud'] < 0.5:
                txa = np.flipud(txa)
            ima = ima * rdv['mix_c'] + txa * (1 - rdv['mix_c'])
            ima_save(ima, Path(save_img_d))
            ima_save(lba, Path(save_lbl_d))
            ima_save(mka, Path(save_msk_d))
            np.save(save_ort_d, ota.astype(np.float16))
            np.save(save_mnt_d, mta.astype(np.float16))
            with open(save_val_d, 'w', encoding='utf-8') as f:
                assert fxv.keys().isdisjoint(rdv.keys())
                for key, value in fxv.items():
                    f.write(f'{key}\t{value}\n')
                for key, value in rdv.items():
                    f.write(f'{key}\t{value}\n')
                end_time = time()
                f.write(f'time_cost\t{end_time - start_time}\n')
            pbar.update(1)
        total_time = pbar.format_dict['elapsed']
    msg = ''
    for k, v in sorted(vars(opt).items()):
        msg += f'{str(k)}: {str(v)}\n'
    msg += f'图像总数：{im_num}\n'
    msg += f'纹理总数：{tx_num}\n'
    msg += f'花费时间：{total_time}\n'
    msg_path = res_path / f'msg.txt'
    with open(msg_path, 'w') as msg_file:
        msg_file.write(msg)
if __name__ == '__main__':
    main()
