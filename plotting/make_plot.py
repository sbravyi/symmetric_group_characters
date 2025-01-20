import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import get_partitions
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'DATA'
FIG_DIR = SCRIPT_DIR.parent / 'FIGS'

path = DATA_DIR  # data directory

file_name = 'mps_data.dat'
with open(path / file_name, 'rb') as fp:
	result_mps = pickle.load(fp)

file_name = 'sage_data.dat'
with open(path / file_name, 'rb') as fp:
	result_sage = pickle.load(fp)

num_trials = len(result_mps)
assert(len(result_sage)==num_trials)

# GAP runtimes (in milliseconds)
runtime_gap = {}
runtime_gap[10] = 25
runtime_gap[12] = 58
runtime_gap[14] = 115
runtime_gap[16] = 234
runtime_gap[18] =  443
runtime_gap[20] = 874
runtime_gap[22] = 1558
runtime_gap[24] =  2757
runtime_gap[26] =  4811
runtime_gap[28] = 8340
runtime_gap[30] =  14023


# minimum n to include in the plot
n_min = 10
result_mps = [result_mps[i] for i in range(num_trials) if np.sum(result_mps[i]['Mu'])>=n_min]
result_sage = [result_sage[i] for i in range(num_trials) if np.sum(result_sage[i]['Mu'])>=n_min]
num_trials = len(result_mps)
assert(num_trials==len(result_sage))


data = {}
data['MPS'] = [result_mps[i]['MPS_runtime_part1']+result_mps[i]['MPS_runtime_part2'] for i in range(num_trials)]
data['GAP'] = [runtime_gap[i]/1000 for i in runtime_gap]
data['SAGE'] = [result_sage[i]['sage_runtime'] for i in range(num_trials)]


# compute maximum approximation error
err_max_full = 0
for i in range(num_trials):
	Mu = result_mps[i]['Mu']
	assert(Mu==result_sage[i]['Mu'])
	n = np.sum(Mu)
	print('n=',n)
	Pn = get_partitions(n)
	err_max = 0
	for Lambda in Pn:
		chi_mps = result_mps[i]['table_mps'][Lambda]
		chi_sage = result_sage[i]['table'][Lambda]
		err = np.abs(chi_mps-chi_sage)
		err_max = max(err,err_max)
	print('Mu=',Mu,'maximum approximation error=',err_max)
	err_max_full = max(err_max_full,err_max)
print('Full maximum approximation error = ',err_max_full)


x = np.arange(num_trials)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

assert(len(x)==len(data['MPS']))
assert(len(x)==len(data['SAGE']))
assert(len(x)==len(data['GAP']))

plt.rcParams['font.size'] = '20'
plt.figure(figsize=(10, 8))
ax = plt.gca()

for name, runtime in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, runtime, width, label=name)
    multiplier += 1


partitions = [str(np.sum(result_mps[i]['Mu'])) for i in range(num_trials)]


ax.set_ylabel('Runtime (seconds)')
ax.set_xticks(x + width, partitions)
ax.legend(loc='upper left', ncols=3)
plt.yscale('log')
ax.set_xlabel('n')
ax.grid(True)
plt.show()
plt.savefig(FIG_DIR/"char.pdf")
