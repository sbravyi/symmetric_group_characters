import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

start = 10
stop = 40
step = 4
relerr = 1e-12
path = './DATA/kostka__short_' # change for different datasets

# plot run times of MPS vs kostka
xarr = []
yarr_mps = []
yarr_sage = []

time_data = {
        'n' : [],
        'Algorithm': [],
        'Runtime': []
        }

# loop through files and convert to dataframe
for n in range(start, stop, step):
    f_name = path+str(n)+'_'+str(relerr)+'.dat'
    with open(f_name, "r") as f:
        for line in f:
            run_data = json.loads(line.strip()) # data from an individual run
            n = sum(run_data[0])
            
            # MPS data
            time_data['n'].append(n)
            time_data['Algorithm'].append('MPS')
            time_data['Runtime'].append(run_data[1])
            
            # Sage data
            time_data['n'].append(n)
            time_data['Algorithm'].append('SAGE')
            time_data['Runtime'].append(run_data[2])

time_data = pd.DataFrame(time_data)

fig,ax = plt.subplots(figsize=(10, 8))

g = sns.boxplot(data=time_data, x='n', y='Runtime', hue='Algorithm', 
                log_scale=True, linewidth=0.8, widths=0.35, 
                showfliers=False, ax=ax)
plt.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.legend(ncol=len(time_data.columns))
plt.ylabel('Runtime (seconds)')
plt.savefig("./FIGS/kostka_"+str(relerr)+".pdf")
