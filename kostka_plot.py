import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

start = 10
stop = 40
step = 4
relerr = 1e-12
path = './DATA/kostka__short_'

# plot run times of MPS vs kostka
xarr = []
yarr_mps = []
yarr_sage = []

# loop through files
for n in range(start, stop, step):
    f_name = path+str(n)+'_'+str(relerr)+'.dat'
    with open(f_name, "r") as f:
        for line in f:
            run_data = json.loads(line.strip()) # data from an individual run
            n = sum(run_data[0])
            xarr.append(n)
            yarr_mps.append(run_data[1])
            yarr_sage.append(run_data[2])
            


           
#sns.barplot(x=[xv - 0 for xv in xarr], y=np.log(yarr_mps), capsize=.1, errorbar="sd")
#sns.barplot(x=[xv + 1 for xv in xarr], y=np.log(yarr_sage), capsize=.1, errorbar="sd", color="red")
# plt.xlabel("n")
# plt.ylabel("Time (s)")

#sns.stripplot(x=[xv + 1 for xv in xarr], y=np.log(yarr_sage), color="0", alpha=.2)
#sns.stripplot(x=[xv - 0 for xv in xarr], y=np.log(yarr_mps), color="1", alpha=.2) 

#sns.violinplot(x=xarr, y = yarr_mps, log_scale=True, alpha=0.25)
#sns.violinplot(x=xarr, y=yarr_sage, log_scale=True, alpha=0.5)

sns.boxplot(x=xarr, y=yarr_sage)
sns.boxplot(x=xarr, y=yarr_mps)
plt.yscale("log")
            
            
            