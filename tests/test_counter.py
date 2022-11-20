import numpy as np
import cupy as cp

from cucounter import Counter

def test_counter_with_numpy():
    keys = np.array([2, 0, 1, 5, 4, 10, 42], dtype=np.uint64)
    kmers = np.array([42, 0, 0, 0, 1, 5, 2, 2, 0, 1, 10, 42], dtype=np.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers)
    counts = counter[keys]

    assert np.all(counts == np.array([2, 4, 2, 1, 0, 1, 2], dtype=np.uint32))

def test_counter_with_cupy():
    keys = cp.array([2, 0, 1, 5, 4, 10, 42], dtype=np.uint64)
    kmers = cp.array([42, 0, 0, 0, 1, 5, 2, 2, 0, 1, 10, 42], dtype=cp.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers)
    counts = counter[keys]

    assert cp.all(counts == cp.array([2, 4, 2, 1, 0, 1, 2], dtype=cp.uint32))

def test_revcomps_with_numpy():
    keys = np.array([0, 0x3FFFFFFFFFFFFFFF], dtype=np.uint64)
    kmers = np.array([0, 0x3FFFFFFFFFFFFFFF], dtype=np.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers, count_revcomps=True, kmer_size=31)
    counts = counter[keys]

    assert np.all(counts == np.array([2, 2], dtype=np.uint32))

def test_revcomps_with_cupy():
    keys = cp.array([0, 0x3FFFFFFFFFFFFFFF], dtype=cp.uint64)
    kmers = cp.array([0, 0x3FFFFFFFFFFFFFFF], dtype=cp.uint64)

    counter = Counter(keys=keys)
    counter.count(kmers, count_revcomps=True, kmer_size=31)
    counts = counter[keys]

    assert cp.all(counts == cp.array([2, 2], dtype=cp.uint32))
