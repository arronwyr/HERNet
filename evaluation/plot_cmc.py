"""Plot the distributed CMC CSV files without loading score matrices."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['Raw','FingerNet','Autoenc','FingerGAN','SFP','SNFEN','UNetEnh','HERNet']
COLORS=['#808080','#4c78a8','#9c755f','#b279a2','#bcbd22','#59a14f','#f28e2b','#d62728']
YEARS={'Raw':'Raw','FingerNet':'FingerNet (2017)','Autoenc':'Autoenc (2020)','FingerGAN':'FingerGAN (2023)','SFP':'SFP (2024)','SNFEN':'SNFEN (2026)','UNetEnh':'UNetEnh (2026)','HERNet':'HERNet'}
PIPELINES=['NBIS','VeriFinger','LEADER_BOZORTH','FLARE']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-rank',type=int,default=50)
    args=parser.parse_args()
    if args.max_rank<20:parser.error('Use at least 20 ranks for this comparison.')
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    overview,axs=plt.subplots(4,3,figsize=(16,18),layout='constrained')
    for i,pipeline in enumerate(PIPELINES):
        for j,dataset in enumerate(['MOLF','MUST','SD302']):
            name=f'{pipeline}_{dataset}'
            with (args.data/(name+'.csv')).open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))[:args.max_rank]
            single,ax=plt.subplots(figsize=(6.8,5.2),layout='constrained')
            for target in [ax,axs[i,j]]:
                for method,color in zip(METHODS,COLORS):
                    if method not in rows[0]:continue
                    target.plot([int(r['rank']) for r in rows],[float(r[method]) for r in rows],color=color,lw=2.5 if method=='HERNet' else 1.5,label=YEARS[method])
                target.set(xlim=(1,args.max_rank),ylim=(0,100),xlabel='Rank',ylabel='Identification rate (%)',title=f'{dataset} | {pipeline.replace("LEADER_BOZORTH","LEADER+Bozorth3")}')
                target.set_xticks([1,10,20,30,40,50] if args.max_rank==50 else [1,20,args.max_rank])
                target.grid(alpha=.2)
                # Keep legends outside the data area so no curve is obscured.
                target.legend(fontsize=7.5,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,columnspacing=.9,handlelength=1.8)
            single.savefig(args.output/(name+'.png'),dpi=220)
            single.savefig(args.output/(name+'.svg'))
            plt.close(single)
    overview.savefig(args.output/'CMC_all_datasets_matchers.png',dpi=200)
    overview.savefig(args.output/'CMC_all_datasets_matchers.svg')
    plt.close(overview)

if __name__=='__main__':main()
