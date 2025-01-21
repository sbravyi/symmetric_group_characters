import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'DATA'
FIG_DIR = SCRIPT_DIR.parent / 'FIGS'
path = DATA_DIR# change for different datasets

# input data file
time_data = pd.DataFrame(pd.read_pickle(DATA_DIR/'kostka_short.delete'))

fig,ax = plt.subplots(figsize=(10, 8))

g = sns.boxplot(data=time_data, x='n', y='Runtime', hue='Algorithm', 
                log_scale=True, linewidth=0.8, widths=0.35, 
                showfliers=False, ax=ax)
plt.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.legend(ncol=len(time_data.columns))
plt.ylabel('Runtime (seconds)')

file_name = FIG_DIR/'kostka_times.pdf'
plt.savefig(file_name)
