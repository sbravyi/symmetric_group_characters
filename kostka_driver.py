import time
from sage.all import *
from kostka_builder import KostkaBuilder, partitions
import random as rm
import numpy as np
import json
import multiprocessing
import signal
import functools

# Code to time and compare the mps Kostka algorithm to symmetrica

path = './DATA/kostka__short_' # path prefix

start = 38
stop =  40# non inclusive
step = 4
relerr = 1e-12
its = 30 # number of iterations per size
#max_time = 1 # seconds to timeout after

def timeout(seconds=5, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                return default

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator

#@timeout(seconds=max_time, default = (-1, {}))
def trial_mps(Mu, Pn):
    table_mps = {}
    t = time.time()
    builder = KostkaBuilder(Mu, relerr)
    for Lambda in Pn:
        table_mps[Lambda] = builder.get_kostka(Lambda)
    return time.time() - t, table_mps

#@timeout(seconds=max_time, default = (-1, {}))    
def trial_sage(Mu, Pn):
    table_sage = {}
    t = time.time()
    for Lambda in Pn:
        table_sage[Lambda] = symmetrica.kostka_number(Lambda, Mu)
    return time.time()-t, table_sage
        

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
            while len(Mu) > int(n/3):
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
            
            # write to file
            json.dump([Mu, elapsed_mps, elapsed_sage, max_error, num_error], f)
            f.write('\n')

for n in range(start, stop, step):
   run(n)