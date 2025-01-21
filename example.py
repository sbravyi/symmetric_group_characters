import numpy as np
import random
from ChiMPS.character_builder import CharacterBuilder
from utils import get_partitions

# compute characters of the symmetric group S_n using the MPS algorithm
n = 8

Pn = get_partitions(n)

Mu = random.choice(Pn)
print('n=', n)
print('Conjugacy class Mu=', Mu)

# compute MPS that encodes all characters of Mu
builder = CharacterBuilder(Mu)

# compute all characters of Mu
for Lambda in Pn:
    chi = builder.get_character(Lambda)
    print('irrep Lambda=', Lambda, 'character=', chi)
