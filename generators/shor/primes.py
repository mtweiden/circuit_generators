primes_to_bits = {
    3 : 2,
    5 : 2,
    7 : 3,
    11 : 4,
    23 : 5,
    47 : 6,
    113 : 7,
    227 : 8,
    491 : 9,
    787 : 10,
    1571 : 11,
    3049 : 12,
    6101 : 13,
    11839 : 14,
    22067 : 15,
    45821 : 16,
    91297 : 17,
    185233 : 18,
    367261 : 19,
    754717 : 20,
    1662803 : 21,
    3250619 : 22,
    6711713 : 23,
    10000019 : 24,
    22335757 : 25,
    44721359 : 26,
    81100289 : 27,
    169131961 : 28,
    355555553 : 29,
    700000001 : 30
}

bits_to_primes = {v:k for k,v in primes_to_bits.items()}

primes = list(primes_to_bits.keys())
bits = list(primes_to_bits.values())

import itertools
from random import shuffle, seed
seed(123)
combos = list(itertools.combinations(bits, 2))
shuffle(combos)

values_of_N = []
max_n = 49
for num_bits_n in range(4, max_n):
    flag = False
    for a,b in combos:
        if a + b == num_bits_n:
            flag = True
            x,y = bits_to_primes[a],bits_to_primes[b]
            values_of_N.append(x*y)
            break
    if flag == False:
        raise RuntimeError(f'Nothing found for {num_bits_n}')
print(f'Number of circuits: {len(values_of_N)}')
print(f'Max width {max_n*2 + 3}')

import pickle
with open('values_for_N.pickle','wb') as f:
    pickle.dump(values_of_N, f)