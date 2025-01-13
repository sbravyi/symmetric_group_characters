from sage.all import symmetrica
import numpy as np
import time
import pickle
from character_builder import get_partitions



result = []

# partitions for the timing test
SelectMu = []
for m in range(2,16):
    SelectMu.append([2]*m)


for Mu in SelectMu:
    n = np.sum(Mu)

    # compute all partitions of n
    Pn = get_partitions(n)

    print('n=',n)
    print('Number of partitions=',len(Pn))

    result_entry = {}
    result_entry['num_partitions'] = len(Pn)

    print('Mu=',Mu)


    result_entry['Mu'] = Mu


    t = time.time()
    table = {}
    for Lambda in Pn:
        table[Lambda] =  symmetrica.charvalue(Lambda,Mu)
    sage_runtime = time.time() - t
    print('sage runtime=',"{0:.5f}".format(sage_runtime))
    result_entry['sage_runtime'] = sage_runtime
    result_entry['table'] = table
    result.append(result_entry)
    print('###################################')



file_name = './DATA/sage_data.dat'
with open(file_name, 'wb') as fp:
    pickle.dump(result, fp, protocol=4)
print('Done')

print('file_name=',file_name)

