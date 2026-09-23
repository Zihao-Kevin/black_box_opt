import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
D=json.loads((P/'curves.json').read_text())
styles={'qcv':('Online QCV','#0072B2'),'row_delta':('Online Q + row delta','#D55E00'),'rloo':('RLOO','#009E73'),'relax':('RELAX','#CC79A7'),'reinforce':('REINFORCE','#E69F00'),'otb':('OTB','#56B4E9'),'grpo':('GRPO','#777777')}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
fig,axes=plt.subplots(1,2,figsize=(14,6.4))
x=np.arange(33);means={}
for method,(label,color) in styles.items():
 a=np.array([[v['mean_reward'] for v in s['validation']] for s in D[method]])
 mean=a.mean(axis=0);means[method]=mean
 half=2.262157*a.std(axis=0,ddof=1)/np.sqrt(10)
 assert np.all(mean+half<1)
 for ax,transform in zip(axes,[lambda r:r,lambda r:-np.log10(1-r)]):
  ax.plot(x,transform(mean),label=label,color=color,lw=1.9)
  ax.fill_between(x,transform(mean-half),transform(mean+half),color=color,alpha=.10,linewidth=0)
for ax in axes:
 ax.set_xlabel('Training epoch (1 epoch = 1,024 training evaluations)')
 ax.set_xlim(0,32);ax.set_xticks(np.arange(0,33,4));ax.grid(alpha=.2)
axes[0].set(title='Original validation reward',ylabel='Mean reward (LB / makespan)',ylim=(.58,.795))
axes[1].set(title='Logarithmic distance to the upper bound',ylabel=r'$-\log_{10}(1-\overline{R})$',ylim=(.37,.70))
fig.suptitle('All 7 algorithms · 10 new seeds · entropy β = 0.1',fontsize=16)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,frameon=False)
fig.subplots_adjust(left=.07,right=.985,bottom=.28,top=.84,wspace=.22)
fig.text(.5,.014,'Per-epoch validation mean, not best-so-far. Shading: pointwise 95% t intervals across 10 seeds, transformed identically.\nLog transform is applied after averaging rewards; upper bound 1 is not assumed attainable. No smoothing or fitted scale.',ha='center',fontsize=9,color='#444444')
for ext in ('png','svg','pdf'):fig.savefig(P/f'all-methods-raw-and-log.{ext}')
plt.close(fig)
# Paired contrast exposes whether a transformed advantage is consistent.
a=np.array([[v['mean_reward'] for v in s['validation']] for s in D['row_delta']]);b=np.array([[v['mean_reward'] for v in s['validation']] for s in D['qcv']]);diff=a-b
fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained');mu=diff.mean(0);half=2.262157*diff.std(0,ddof=1)/np.sqrt(10)
ax.plot(x,mu,color='#D55E00',lw=2);ax.fill_between(x,mu-half,mu+half,color='#D55E00',alpha=.18);ax.axhline(0,color='#444444',lw=1)
ax.set(xlim=(0,32),xlabel='Training epoch',ylabel='Paired reward difference',title='Online Q + row delta minus online QCV (10 paired seeds)');ax.grid(alpha=.2)
fig.savefig(P/'paired-difference.png');plt.close(fig)
q=means['qcv'];r=means['row_delta']
stats={'mean_reward_advantage_epochs_1_32':float((r-q)[1:].mean()),'positive_epochs':int(np.sum(r[1:]>q[1:])),'final_epoch_reward_difference':float(r[-1]-q[-1]),'final_epoch_gap_reduction_fraction':float((r[-1]-q[-1])/(1-q[-1])),'final_epoch_log_difference':float(np.log10((1-q[-1])/(1-r[-1])))}
(P/'analysis.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
(P/'README.md').write_text('''# Validation learning curves: raw reward and logarithmic gap

Campaign: kings-beta010-val128-n10; training seeds 10–19; 128 validation instances, four samples per instance; all seven methods, beta=0.1, 32 epochs.

The displayed transform is **S = -log10(1 - mean reward)**, applied to the per-epoch reward averaged over validation samples and training seeds. Reward is LB/makespan, with theoretical upper bound 1. The bound need not be attainable on every instance. We do not transform individual samples (some could equal 1), choose epsilon, fit a reference optimum, smooth curves, or use best-so-far envelopes.

An increase of 0.1 in S means the remaining gap is multiplied by 10^-0.1, or reduced by about 20.6%. This is a descriptive relative-gap scale, not evidence that discovery difficulty grows exponentially. The monotone transform preserves every per-epoch ranking of aggregate rewards; it cannot establish a row-delta advantage absent from the raw curve.

Bands are pointwise 95% t intervals over ten training seeds, transformed using the same monotone function. They do not include uncertainty from drawing a new validation set and are not simultaneous confidence bands. The paired-difference figure uses matched training seeds on the original scale.

The transform was chosen after inspecting results and is exploratory. No claims of statistical superiority or SOTA follow from this visualization. Final test results use validation-selected checkpoints and are different from final-epoch validation curves.
''')
