import numpy as np
#import cupy as cp

#from cucounter import Counter

def test_test():
    assert True

"""
def test_counter_with_numpy():
    keys = np.array([0, 1, 2, 5, 10, 42], dtype=np.uint64)
    kmers = np.array([42, 0, 0, 0, 1, 5, 2, 2, 0, 1, 10, 42], dtype=np.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers)
    counts = counter[keys]

    assert np.all(counts == np.array([4, 2, 2, 1, 1, 2], dtype=np.uint64))

def test_counter_with_cupy():
    keys = cp.array([0, 1, 2, 5, 10, 42], dtype=cp.uint64)
    kmers = cp.array([42, 0, 0, 0, 1, 5, 2, 2, 0, 1, 10, 42], dtype=cp.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers)
    counts = counter[keys]

    assert cp.all(counts == cp.array([4, 2, 2, 1, 1, 2], dtype=cp.uint64))
"""
