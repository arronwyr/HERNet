"""Batch enhancement of a directory of fingerprint images."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from hernet import load_model

EXTENSIONS = {'.png','.bmp','.tif','.tiff','.jpg','.jpeg'}

def read_image(path):
    with Image.open(path) as image:
        if image.mode not in ('L','RGB'):
            if image.mode.startswith('I') or image.mode == 'F':
                raise ValueError(f'{path}: convert high-bit-depth images to calibrated 8-bit images first.')
            image = image.convert('RGB')
        array = np.asarray(image,dtype=np.float32)/255.
    # Original preprocessing uses arithmetic RGB-channel mean, not luminance.
    if array.ndim == 3:
        array = array.mean(axis=2)
    return torch.from_numpy(array.copy())[None]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--weights',type=Path,default=Path(__file__).parent/'weights/hernet_inference.pth')
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--recursive',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    src=args.input.resolve();dst=args.output.resolve()
    if not src.is_dir():parser.error('Input must be an existing directory.')
    if src == dst or src in dst.parents:parser.error('Output must be outside the input directory.')
    files=sorted(p for p in (src.rglob('*') if args.recursive else src.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:parser.error('No supported images found.')
    targets=[dst/p.relative_to(src).with_suffix('.png') for p in files]
    if len(set(targets)) != len(targets):parser.error('Input names collide after conversion to PNG.')
    if not args.overwrite and any(p.exists() for p in targets):parser.error('Output exists; use a new directory or --overwrite.')
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=load_model(args.weights,args.device)
    with torch.inference_mode():
        for index,(path,target) in enumerate(zip(files,targets),1):
            result=model.enhance(read_image(path).to(args.device)).squeeze(0).cpu().numpy()
            # Match the original ima_float2uint rounding, including bankers' ties.
            image=np.uint8(np.clip(np.round(result*255.),0,255))
            target.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(image).save(target)
            print(f'[{index}/{len(files)}] {path.relative_to(src)} -> {target.relative_to(dst)}',flush=True)

if __name__=='__main__':main()
