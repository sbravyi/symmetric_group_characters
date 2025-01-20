import numpy as np
import pickle
import matplotlib.pyplot as plt

file_name = 'mps_data.dat'

with open('./DATA/'+file_name, 'rb') as fp:
	result_mps = pickle.load(fp)

num_trials = len(result_mps)

# minimum n to include in the plot
n_min = 10
result_mps = [result_mps[i] for i in range(num_trials) if np.sum(result_mps[i]['Mu'])>=n_min]
num_trials = len(result_mps)


partitions = [str(np.sum(result_mps[i]['Mu'])) for i in range(num_trials)]



data = {}
data['D'] = [result_mps[i]['D'] for i in range(num_trials)]
data['upper bound'] = [result_mps[i]['Dupper'] for i in range(num_trials)]


x = np.arange(num_trials)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

plt.figure(figsize=(10, 8))
plt.rcParams['font.size'] = '20'
ax = plt.gca()


color = {}
color['D'] = 'C0'
color['upper bound'] = 'C2'
for name, D in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, D, width, label=name,color=color[name])
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MPS bond dimension $D$')
ax.set_xticks(x + width, partitions)
ax.legend(loc='upper left', ncols=3)
plt.yscale('log')
ax.set_xlabel('n')
ax.grid(True)


plt.show()
plt.savefig("./FIGS/char_dim.pdf")

