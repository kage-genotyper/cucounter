import numpy as np
import cupy as cp

from cucounter_C import HashTable


class Counter(HashTable):
    def __init__(self, keys, capacity: int = 0):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        # Dynamically determine hashtable capacity if not provided
        if capacity == 0:
            capacity = int(keys.size * 1.4)

        if isinstance(keys, np.ndarray):
            super().__init__(keys, capacity)
        elif isinstance(keys, cp.ndarray):
            super().__init__(keys.data.ptr, keys.size, capacity)

    def count(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        if isinstance(keys, np.ndarray):
            super().count(keys)
        elif isinstance(keys, cp.ndarray):
            super().count(keys.data.ptr, keys.size)

    def __getitem__(self, keys):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "Invalid key type"

        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().get(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 
        
