import numpy as np
import time
import pickle
from character_building.character_builder import CharacterBuilder
from utils import get_partitions
from pathlib import Path


# stores all computation results and runtime
result = []

# partitions for the runtime test: (n/2) cycles of length=2
SelectMu = []
for m in range(2,16):
    SelectMu.append([2]*m)

for Mu in SelectMu:

    n = np.sum(Mu)
    Pn = get_partitions(n)


    print('n=',n)
    print('Number of partitions=',len(Pn))

    result_entry = {}
    result_entry['num_partitions'] = len(Pn)

    print('Mu=',Mu)
    result_entry['Mu'] = Mu


    t = time.time()
    builder = CharacterBuilder(Mu)
    # runtime for computing the MPS
    runtime_part1 = time.time() - t

    D = builder.get_bond_dimension()
    print('Maximum MPS bond dimension =',D)
    result_entry['D'] = D

    # analytic upper bound on the bond dimension 
    Dupper = np.prod([Mu[i]+2 for i in range(len(Mu))])
    #print('Upper bound on MPS bond dimension =',Dupper)
    result_entry['Dupper'] = Dupper

    # compute all characters of Mu
    table_mps = {}
    t = time.time()
    for Lambda in Pn:
        table_mps[Lambda] = builder.get_character(Lambda)
    # runtime for computing the characters
    runtime_part2 = time.time() - t
    print('MPS runtime (part 1)=',"{0:.2f}".format(runtime_part1))
    print('MPS runtime (part 2)=',"{0:.2f}".format(runtime_part2))
    result_entry['MPS_runtime_part1'] = runtime_part1
    result_entry['MPS_runtime_part2'] = runtime_part2
    result_entry['table_mps'] = table_mps
    result.append(result_entry)
    print('###################################')


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'DATA'

path = DATA_DIR  # data directory

file_name = path/'mps_data.dat'


with open(file_name, 'wb') as fp:
    pickle.dump(result, fp)
print('Done')

print('file_name=',file_name)

