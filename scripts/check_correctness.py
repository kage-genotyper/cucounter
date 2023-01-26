import time
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from cucounter import Counter as CuCounter 

def get_arguments():
    parser = argparse.ArgumentParser("Script checking that counts computed by cucounter.Counter and npstructures.Counter are equal.")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    parser.add_argument("-cucounter_capacity", type=int, default=0)
    parser.add_argument("-cucounter_capacity_factor", type=float, default=1.7331)
    return parser.parse_args()


fasta_filename = "data/fa/testreads20m.fa"
keys_filename = "data/npy/uniquekmers.npy"
args = get_arguments()

def check(fasta_filename, keys_filename, xp, counter_size, chunk_size, 
        cucounter_capacity, cucounter_capacity_factor):
    keys = np.load(keys_filename)[:counter_size]
    keys = xp.asanyarray(keys)

    nps_counter = nps.Counter(keys=keys)
    cu_counter = CuCounter(keys=keys, 
            capacity=cucounter_capacity, capacity_factor=cucounter_capacity_factor)

    for i, chunk in enumerate(bnp.open(fasta_filename, chunk_size=chunk_size), start=1):
        kmers = bnp.kmers.fast_hash(chunk.sequence, 31, bnp.encodings.ACTGEncoding)

        nps_counter.count(kmers.ravel())
        cu_counter.count(kmers.ravel())

        print(f"Counting kmer chunks ... {i}\r", end="")
    cp.cuda.runtime.deviceSynchronize()
    print(f"Counting kmer chunks ... {i}")

    nps_counts = nps_counter[keys.ravel()]
    cu_counts = cu_counter[keys.ravel()]

    assert isinstance(nps_counts, xp.ndarray)
    assert isinstance(cu_counts, xp.ndarray)
    xp.testing.assert_array_equal(nps_counts, cu_counts)
    print("Assert passed")

if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp

    print(f"backend array module      : {array_module.__name__}")
    print(f"counter size              : {args.counter_size}")
    print(f"chunk size                : {args.chunk_size}")
    print(f"cucounter capacity        : {args.cucounter_capacity}")
    print(f"cucounter capacity factor : {args.cucounter_capacity_factor}")

    check(fasta_filename, keys_filename, array_module, args.counter_size, 
            args.chunk_size, args.cucounter_capacity, args.cucounter_capacity_factor)
