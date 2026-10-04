"""Export the completed selection distribution and paired held-out comparisons."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from thicket_runtime.visual_runtime import read

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--stem',default='selection-and-transfer')
a = p.parse_args()
if Path(a.stem).name != a.stem: raise ValueError('figure stem must be a filename')
targets = [a.analysis/(a.stem+'.'+suffix) for suffix in ('png','svg')]
if any(path.exists() for path in targets): raise FileExistsError('figure outputs already exist')
distribution = read(a.analysis/'distribution.json')
experts = read(a.analysis/'individual_experts.json')
histogram = distribution['histogram_correct_counts']
fig,axes = plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
axes[0].bar([100*int(k)/150 for k in histogram],list(histogram.values()),width=.58,color='#4279a8')
axes[0].axvline(distribution['base_selection_accuracy']*100,color='#333333',linestyle='--',label='Base selection')
axes[0].set(xlabel='Selection accuracy (%)',ylabel='Candidates',title='300 new candidates; frozen sigma 0.0005')
axes[0].legend(frameon=False,fontsize=9)
for i,r in enumerate(experts):
    gain = r['comparison']['gain']*100
    lo,hi = np.asarray(r['comparison']['paired_bootstrap95'])*100
    color = '#c66b26' if i == 0 else '#4279a8'
    axes[1].plot([lo,hi],[i+1,i+1],color=color,linewidth=2)
    axes[1].scatter([gain],[i+1],color=color,s=32,zorder=3)
axes[1].axvline(0,color='#333333',linestyle='--',label='Base held-out')
axes[1].axvline(5,color='#48864c',linestyle=':',label='+5 pp effect-size threshold')
axes[1].set(xlabel='Held-out gain over base (percentage points)',ylabel='Frozen selection rank',
            title='Top-10 held-out transfer; paired 95% intervals',yticks=range(1,11),ylim=(10.7,.3))
axes[1].text(0,.65,'Base',ha='center',va='bottom',fontsize=8,color='#333333')
axes[1].text(5,.65,'+5 pp threshold',ha='center',va='bottom',fontsize=8,color='#48864c')
fig.suptitle('Final v1 line tracing: selection150 → held-out500',fontsize=13)
for target in targets: fig.savefig(target,dpi=180)
plt.close(fig)
print([str(path) for path in targets])
