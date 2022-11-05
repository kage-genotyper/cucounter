import numpy as np
import cupy as cp

from cucounter_C import HashTable

class Counter(HashTable):
    def __init__(self, keys, capacity: int = 0, capacity_factor: float = 1.8):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"
        assert capacity_factor > 1.0, "capacity_factor must be greater than 1.0"

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * capacity_factor) 

        assert capacity > keys.size, "Capacity must be greater than size of keyset"

        if isinstance(keys, np.ndarray):
            super().__init__(keys, capacity)
        elif isinstance(keys, cp.ndarray):
            super().__init__(keys.data.ptr, keys.size, capacity)

    def count(self, keys, count_revcomps=False, kmer_size=32):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"
        assert kmer_size > 0 and kmer_size <= 32, "kmer size must be 32 >= size > 0"

        if isinstance(keys, np.ndarray):
            super().count(keys, count_revcomps, kmer_size)
        elif isinstance(keys, cp.ndarray):
            super().count(keys.data.ptr, keys.size, count_revcomps, kmer_size)

    def __getitem__(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().get(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 
        
