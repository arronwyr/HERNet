"""Pack generated training fields into the HERNet ILSOM HDF5 layout."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

def gray(path):
    with Image.open(path) as image:
        array=np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit grayscale image: {path}')
    return array

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='Generated directory containing image/, label/, mask/, orientation/, minutia/.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args();root=args.input
    files=sorted(p for p in (root/'image').rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.bmp','.tif','.tiff'})
    if not files:parser.error('No generated images found.')
    if args.output.exists():parser.error('Output exists; refusing to overwrite.')
    # Validate all paths before creating a potentially large archive.
    for path in files:
        relative=path.relative_to(root/'image')
        for field in ['label','mask','orientation','minutia']:
            target=root/field/(relative.with_suffix('.npy') if field in ['orientation','minutia'] else relative)
            if not target.is_file():raise FileNotFoundError(target)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    n=len(files);s=args.size
    with h5py.File(args.output,'x') as archive:
        fields={name:archive.create_dataset(name,(n,channels,s,s),dtype=dtype,chunks=(1,channels,s,s))
            for name,channels,dtype in [('img',1,'u1'),('lbl',1,'u1'),('msk',1,'u1'),('ort',2,'f2'),('mnt',1,'f2')]}
        archive.attrs['image_number']=n
        archive.attrs['orientation_encoding']='cos(2*theta),sin(2*theta); zero outside mask'
        archive.create_dataset('filenames',data=[p.relative_to(root/'image').as_posix() for p in files],dtype=h5py.string_dtype())
        for i,path in enumerate(files):
            r=path.relative_to(root/'image');mask=gray(root/'mask'/r)
            arrays={'img':gray(path)[None],'lbl':gray(root/'label'/r)[None],'msk':mask[None]}
            theta=np.load(root/'orientation'/r.with_suffix('.npy'),allow_pickle=False)
            arrays['ort']=np.where(mask[None]>0,np.stack([np.cos(2*theta),np.sin(2*theta)]),0).astype(np.float16)
            arrays['mnt']=np.load(root/'minutia'/r.with_suffix('.npy'),allow_pickle=False)[None].astype(np.float16)
            for name,array in arrays.items():
                if array.shape != fields[name].shape[1:]:raise ValueError(f'{path}: wrong shape for {name}: {array.shape}')
                if not np.isfinite(array).all():raise ValueError(f'{path}: nonfinite {name}')
                fields[name][i]=array
            print(f'[{i+1}/{n}] {r}',flush=True)

if __name__=='__main__':main()
