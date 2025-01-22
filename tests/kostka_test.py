"""
Validation of skew and standard Kostka numbers
Tested m < 8 and >= 8 (caching is handled different in these cases)
"""

from kostka_builder import KostkaBuilder
from utils import get_partitions
from sage.all import *
import random as rm

P7 = utils.get_partitions(7)
P8 = utils.get_partitions(8)

# Standard Kostka Numbers

Mu7 = rm.choice(P7)
bd_7 = KostkaBuilder(Mu7)
table_7 = {}
errors_7 = 0
for Lambda in P7:
    table_7[Lambda] = bd_7.get_kostka(Lambda)
    if table_7[Lambda] != symmetrica.kostka_number(Lambda, Mu7):
        errors_7 += 1
print("Errors for Kostkas with Mu=" + str(Mu7) + ": ", errors_7)


Mu8 = rm.choice(P8)
bd_8 = KostkaBuilder(Mu8)
table_8 = {}
errors_8 = 0
for Lambda in P8:
    table_8[Lambda] = bd_8.get_kostka(Lambda)
    if table_8[Lambda] != symmetrica.kostka_number(Lambda, Mu8):
        errors_8 += 1
print("Errors for Kostkas with Mu=" + str(Mu8) + ": ", errors_8)


# Skew Kostka Numbers:

# correct skew Kostka numbers for
#   Mu = (2,1,1)
#   Nu = (2,1)
skew_test_7 = {
    (7,): 0,
    (6, 1): 1,
    (5, 1, 1): 3,
    (4, 1, 1, 1): 3,
    (3, 1, 1, 1, 1): 1,
    (2, 1, 1, 1, 1, 1): 0,
    (1, 1, 1, 1, 1, 1, 1): 0,
    (2, 2, 1, 1, 1): 1,
    (3, 2, 1, 1): 5,
    (4, 2, 1): 7,
    (2, 2, 2, 1): 2,
    (3, 3, 1): 4,
    (5, 2): 3,
    (3, 2, 2): 4,
    (4, 3): 3,
}

bd_skew_7 = KostkaBuilder((2, 1, 1), Nu=(2, 1))
table_skew_7 = {}
errors_skew_7 = 0
for Lambda in P7:
    table_skew_7[Lambda] = bd_skew_7.get_kostka(Lambda)
    if table_skew_7[Lambda] != skew_test_7[Lambda]:
        errors_skew_7 += 1
print("Errors for skew Kostkas of size n=7: ", errors_skew_7)

# correct skew Kostka numbers for
#   Mu = (2,1,1)
#   Nu = (2,1,1)
skew_test_8 = {
    (8,): 0,
    (7, 1): 0,
    (6, 1, 1): 1,
    (5, 1, 1, 1): 3,
    (4, 1, 1, 1, 1): 3,
    (3, 1, 1, 1, 1, 1): 1,
    (2, 1, 1, 1, 1, 1, 1): 0,
    (1, 1, 1, 1, 1, 1, 1, 1): 0,
    (2, 2, 1, 1, 1, 1): 1,
    (3, 2, 1, 1, 1): 5,
    (4, 2, 1, 1): 7,
    (2, 2, 2, 1, 1): 2,
    (3, 3, 1, 1): 4,
    (5, 2, 1): 3,
    (3, 2, 2, 1): 5,
    (4, 3, 1): 3,
    (6, 2): 0,
    (4, 2, 2): 3,
    (2, 2, 2, 2): 1,
    (3, 3, 2): 2,
    (5, 3): 0,
    (4, 4): 0,
}

bd_skew_8 = KostkaBuilder((2, 1, 1), Nu=(2, 1, 1))
table_skew_8 = {}
errors_skew_8 = 0
for Lambda in P8:
    table_skew_8[Lambda] = bd_skew_8.get_kostka(Lambda)
    if table_skew_8[Lambda] != skew_test_8[Lambda]:
        errors_skew_8 += 1
print("Errors for skew Kostkas of size n=8: ", errors_skew_8)
