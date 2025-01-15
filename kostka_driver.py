import time
from sage.all import *
from kostka_builder import KostkaBuilder, partitions
import random as rm
import numpy as np
import json

# Code to time and compare the mps Kostka algorithm to symmetrica

path = './DATA/kostka_' # path prefix

start = 10
stop =  35# non inclusive
step = 4
relerr = 1e-12
its = 100 # number of iterations per size

# collects data for size n
def run(n):
    f_name = path+str(n)+'_'+str(relerr)+'.dat'
    with open(f_name, "a") as f:
        # create all partitions of n
        Pn = [list(p) for p in partitions(n)]
        for p in Pn:
            p.reverse()
        Pn = [tuple(p) for p in Pn]
        
        # run time trials
        for i in range(its):
            Mu = rm.choice(Pn) # random Mu
            table_mps = {}
            table_sage = {}
            
            # run MPS algorithm
            t = time.time()
            builder = KostkaBuilder(Mu, relerr)
            for Lambda in Pn:
                table_mps[Lambda] = builder.get_kostka(Lambda)
            elapsed_mps = time.time() - t
            
            # call SAGE
            t = time.time()
            for Lambda in Pn:
                table_sage[Lambda] = symmetrica.kostka_number(Lambda, Mu)
            elapsed_sage = time.time()-t
            
            # check for errors
            max_error = 0
            num_error = 0
            for Lambda in Pn:
                tmp = np.abs(table_mps[Lambda] - table_sage[Lambda])
                if tmp > max_error:
                    max_error = tmp
                if tmp >= 0.5:
                    num_error += 1
            
            # write to file
            json.dump([Mu, elapsed_mps, elapsed_sage, max_error, num_error], f)
            f.write('\n')

for n in range(start, stop, step):
    run(n)