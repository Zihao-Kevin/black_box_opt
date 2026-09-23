import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';runs=json.loads((R/'training-summary.json').read_text())
styles={'qcv':('Online QCV','#0072B2'),'row_delta':('Online Q + row delta','#D55E00'),'rloo':('RLOO','#009E73'),'relax':('RELAX','#CC79A7'),'reinforce':('REINFORCE','#E69F00'),'otb':('OTB','#56B4E9'),'grpo':('GRPO','#777777')}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
for method,(label,color) in styles.items():
 values=np.array([[r['initial_validation']['mean_reward']]+[e['validation']['mean_reward'] for e in r['epochs']] for r in runs if r['method']==method]);assert values.shape==(10,33)
 mean=values.mean(0);half=2.262157*values.std(0,ddof=1)/np.sqrt(10)
 ax.plot(range(33),mean,label=label,color=color);ax.fill_between(range(33),mean-half,mean+half,color=color,alpha=.12)
ax.set(xlabel='Training epoch (1,024 training oracle calls per epoch)',ylabel='Validation mean reward (LB / makespan)',title='JobShop · 10 new seeds · entropy β = 0.1 · 128 validation instances',xlim=(0,32));ax.grid(alpha=.2);ax.legend(ncol=2,loc='lower right',frameon=False)
for ext in ['png','svg','pdf']:fig.savefig(R/f'validation-learning-curves.{ext}',dpi=180)
