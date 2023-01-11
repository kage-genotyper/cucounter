import numpy as np
import cupy as cp

from cucounter_backend import HashTable

class Counter(HashTable):
    def __init__(self, keys, capacity: int = 0, capacity_factor: float = 1.7331):
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "keys must be numpy or cupy ndarray"
        assert keys.dtype == np.uint64, "keys.dtype must be uint64"
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
        """
        Count a set of keys. 
        Duplicate keys are counted equally as many times as they are observed. 
        If count_revcomps == True, each individual key and its reverse complement will be counted
        as two separate keys. When computing the reverse complement of a key, the kmer_size
        argument determines the size of the kmer.

        Args:
            keys (numpy.ndarray or cupy.ndarray): the keys to increment in the hashtable.
            count_revcomps (bool): whether to count each key's reverse complement.
            kmer_size (int): the assumed kmer size.
        """
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "keys must be numpy or cupy ndarray"
        assert keys.dtype == np.uint64, "keys.dtype must be uint64"
        assert kmer_size > 0 and kmer_size <= 32, "kmer size must be 32 >= size > 0"

        if isinstance(keys, np.ndarray):
            super().count(keys, count_revcomps, kmer_size)
        elif isinstance(keys, cp.ndarray):
            super().count(keys.data.ptr, keys.size, count_revcomps, kmer_size)

    def __getitem__(self, keys):
        """
        Look up a set of keys to get their corresponding counts from the hashtable.
        Each input key's index will correspond to the output count's indices.

        Args:
            keys (numpy.ndarray or cupy.ndarray): the keyset to lookup in the hashtable.
        Returns:
            counts (numpy.ndarray or cupy.ndarray): the counts corresponding to the input keyset.
                counts will match the input key's type when determining whether to return a
                numpy.ndarray or cupy.ndarray.
        """
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "keys must be numpy or cupy ndarray"
        assert keys.dtype == np.uint64, "keys.dtype must be uint64"

        if isinstance(keys, np.ndarray):
            return super().get(keys)
        elif isinstance(keys, cp.ndarray):
            counts = cp.zeros_like(keys, dtype=np.uint32)
            super().get(keys.data.ptr, counts.data.ptr, keys.size)
            return counts 

    def get_probe_lengths(self, keys):
        """
        Determines the number of (linear probing) probes needed before termination for each key 
        in the input keyset. Termination for a key is reached when the probing scheme finds the
        corresponding key in the hashtable or when an empty position is found, indicating that the
        key is not present in the hashtable.
        Currently does not support checking probing lengths for reverse complements.

        Args:
            keys (numpy.ndarray or cupy.ndarray): the keyset to check probing lengths for.
        Returns:
            probe_lengths (numpy.ndarray or cupy.ndarray): the number of probes performed before
                terminating for each key. 
                probe_lengths will match the input key's type when determining whether to 
                return a numpy.ndarray or cupy.ndarray.
        """
        assert isinstance(keys, (np.ndarray, cp.ndarray)), "keys must be numpy or cupy ndarray"
        assert keys.dtype == np.uint64, "keys.dtype must be uint64"

        if isinstance(keys, np.ndarray):
            return super().get_probe_lengths(keys)
        elif isinstance(keys, cp.ndarray):
            probe_lengths = cp.zeros_like(keys, dtype=np.uint32)
            super.get_probe_lengths(keys.data.ptr, probe_lengths.data.ptr, keys.size)
            return probe_lengths

    def __repr__(self):
        return super().to_string(True)

    def __str__(self):
        return super().to_string(False)
