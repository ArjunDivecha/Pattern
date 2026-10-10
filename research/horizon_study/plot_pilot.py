"""Plot the saved primary n-tile diagnostic; no training or data edits."""
from pathlib import Path
import os,tempfile,sys
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='pattern-mpl-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
out=Path(sys.argv[1]);s=pd.read_csv(out/'deciles_summary.csv')
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(s.tile,s.mean_forward_return*100,color=['#bd5b37' if t==1 else '#214f79' if t==10 else '#b6c5d1' for t in s.tile])
ax.axhline(0,color='#555',lw=.8);ax.set_xticks(s.tile,[f'D{t}' for t in s.tile])
ax.set_ylabel('Mean 5-session forward return (%)');ax.set_title('Pilot: bottom-to-top decile returns — not a strategy finding')
ax.spines[['top','right']].set_visible(False)
fig.text(.12,.015,'Resolved dates only; counts vary by decile. One seed, two epochs, 30 signal dates.',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'deciles.png',dpi=150)
