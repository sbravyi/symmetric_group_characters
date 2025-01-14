import numpy as np
import time
import pickle
from character_builder import get_partitions
from kostka_builder import KostkaBuilder
from sage.all import * 

n = 15

# compute all partitions of n
Pn = [list(p) for p in list(get_partitions(n))]
for p in Pn:
    p.reverse()
Pn = [tuple(p) for p in Pn]

print('n=',n)
print('Number of partitions=',len(Pn))

#Mu = rm.choice(Pn)
Mu = [1]*n # seems to be the slowest run time
print('Weight Mu=',Mu)

# compute all kostas of weight Mu using the MPS algorithm
t = time.time()
builder = KostkaBuilder(Mu)
print("done building")
table_mps = {}
for Lambda in Pn:
    table_mps[Lambda] = builder.get_kostka(Lambda)
elapsed = time.time() - t
print('MPS runtime=',"{0:.2f}".format(elapsed))

def kos(Mu):
    assert(np.sum(Mu) == n)
    tt = time.time()
    build = KostkaBuilder(Mu)
    table = {}
    for Lambda in Pn:
        table[Lambda] = build.get_kostka(Lambda)
    return table, time.time()-tt

# compute all kostkas of weight Mu using sage
t = time.time()
table_sage = {}
for Lambda in Pn:
    table_sage[Lambda] = symmetrica.kostka_number(Lambda, Mu) # symmetrica is a sage package
elapsed = time.time() - t
print('Sage runtime=',"{0:.2f}".format(elapsed))

#check correctness of the MPS algorithm
err_max = 0
for Lambda in Pn:
   err_max = max(err_max, np.abs(table_mps[Lambda]-table_sage[Lambda]))
print('maximum approximation error=',err_max)