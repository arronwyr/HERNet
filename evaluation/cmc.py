"""Compute image-candidate CMC from a score matrix and ordered filename lists."""
import argparse, csv
from pathlib import Path
import numpy as np

def names(path):
    # Interpret both Windows and POSIX filename lists on either platform.
    return [Path(line.strip().replace('\\','/')).stem.lower() for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def identity(name,dataset):
    parts=name.lower().split('_')
    if dataset=='molf':return parts[0]
    if dataset=='must':return (parts[0],parts[2])
    if dataset=='sd302':return (parts[0],int(parts[-1]))
    raise ValueError(dataset)

def first_relevant_ranks(scores,probes,gallery,dataset,chunk=128):
    if scores.shape != (len(probes),len(gallery)):raise ValueError('Matrix dimensions do not match the ordered lists.')
    gallery_ids=[identity(name,dataset) for name in gallery]
    groups={key:np.array([i for i,v in enumerate(gallery_ids) if v==key]) for key in set(gallery_ids)}
    ranks=[]
    for start in range(0,len(probes),chunk):
        block=np.asarray(scores[start:start+chunk])
        if not np.isfinite(block).all():raise ValueError('Nonfinite matching scores.')
        # Preserve matrix dtype and the historical default NumPy argsort tie rule.
        ordered=np.argsort(block,axis=1)[:,::-1]
        for row,probe in zip(ordered,probes[start:start+len(block)]):
            key=identity(probe,dataset)
            if key not in groups:raise ValueError(f'No gallery mate for {probe}')
            hits=np.flatnonzero(np.isin(row,groups[key]))
            ranks.append(int(hits[0])+1)
    return np.asarray(ranks,dtype=np.int32)

def curve(ranks,gallery_size):
    histogram=np.bincount(ranks,minlength=gallery_size+1)
    return np.cumsum(histogram)[1:]/len(ranks)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',required=True,type=Path)
    parser.add_argument('--probes',required=True,type=Path)
    parser.add_argument('--gallery',required=True,type=Path)
    parser.add_argument('--dataset',required=True,choices=['molf','must','sd302'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    scores=np.load(args.scores,mmap_mode='r',allow_pickle=False)
    rates=curve(first_relevant_ranks(scores,names(args.probes),names(args.gallery),args.dataset),scores.shape[1])
    if args.output.exists():parser.error('Output exists; refusing to overwrite.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['rank','identification_rate_percent'])
        writer.writerows((i+1,float(rate*100)) for i,rate in enumerate(rates))

if __name__=='__main__':main()
