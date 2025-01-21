import time
from sage.all import *
from character_building.kostka_builder import KostkaBuilder
from utils import get_partitions
import random as rm
import numpy as np
from pathlib import Path
import pickle

# Code to time and compare the MPS Kostka algorithm to symmetrica
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'DATA'

path = DATA_DIR  # data directory
file_name = path/'kostka_short.dat'

start = 38
stop =  40# non inclusive
step = 4
relerr = 1e-12
its = 29 # number of iterations per size

def trial_mps(Mu, Pn):
    table_mps = {}
    t = time.time()
    builder = KostkaBuilder(Mu, relerr)
    for Lambda in Pn:
        table_mps[Lambda] = builder.get_kostka(Lambda)
    return time.time() - t, table_mps

def trial_sage(Mu, Pn):
    table_sage = {}
    t = time.time()
    for Lambda in Pn:
        table_sage[Lambda] = symmetrica.kostka_number(Lambda, Mu)
    return time.time()-t, table_sage

results =  [] # array of dictionaries

for n in range(start, stop, step):
    arr = ()
    Pn = get_partitions(n)
    
    print('Running Kostkas for partitions of length '+str(n))
    
    
    # run time trials
    for i in range(its):
        print("Iteration: "+str(i), end='\r')
        Mu = rm.choice(Pn) # random Mu
        while len(Mu) > int(n/3): # require Mu to be "short"
            Mu = rm.choice(Pn)
        
        elapsed_mps, table_mps = trial_mps(Mu, Pn)
        elapsed_sage, table_sage = trial_sage(Mu, Pn)
        
        # check for errors
        max_error = 0
        num_error = 0
        if table_mps and table_sage:
            for Lambda in Pn:
                tmp = np.abs(table_mps[Lambda] - table_sage[Lambda])
                if tmp > max_error:
                    max_error = tmp
                    if tmp >= 0.5:
                        num_error += 1
         
        # MPS data
        results.append({
            'n': n,
            'Algorithm': 'MPS',
            'Runtime': elapsed_mps,
            'Mu': Mu,
            'Errors': num_error,
            'Max error': max_error,
            'Relerr': relerr
            })
        
        # SAGE data
        results.append({
            'n': n,
            'Algorithm': 'SAGE',
            'Runtime': elapsed_sage,
            'Mu': Mu,
            'Errors': 0,
            'Max error': 0,
            'Relerr': 0
            })
        
    print('################################################')

with open(file_name, 'wb') as fp:
    pickle.dump(results, fp)
print('Done')
print('file_name=',file_name)

