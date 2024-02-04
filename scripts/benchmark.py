import time
import importlib
import shutil
import pickle
import argparse

import numpy as np
import cupy as cp

import npstructures as nps
import bionumpy as bnp

from cucounter import Counter as CuCounter


def get_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking script for reading a fasta file and achieving a frequency count for kmers")
    parser.add_argument("-backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("-counter", choices=["nps", "cu"], required=True)
    parser.add_argument("-counter_size", type=int, required=True)
    parser.add_argument("-chunk_size", type=int, required=True)
    parser.add_argument("-cucounter_capacity", type=int, default=0)
    parser.add_argument("-cucounter_capacity_factor", type=float, default=1.7337)
    parser.add_argument("-count_revcomps", action="store_true")
    parser.add_argument("-kmer_size", type=int, default=31)

    return parser.parse_args()


args = get_arguments()

fasta_filename = "data/fa/testreads20m.fa"
#keys_filename = "data/npy/uniquekmers.npy"
keys_filename = "../kmer-counting-experimentation/data/npy/uniquekmersACGT.npy"


def pipeline(fasta_filename, keys_filename, xp, counter_type, counter_size, chunk_size, 
        cucounter_capacity, cucounter_capacity_factor, count_revcomps, kmer_size):
    shsize = shutil.get_terminal_size().columns
    print(">> INFO")
    print(f"BACKEND_ARRAY_MODULE       : {xp.__name__}")
    print(f"COUNTER_TYPE               : {'Counter' if counter_type == nps.Counter else 'CuCounter'}")
    print(f"COUNTER_SIZE               : {counter_size}")
    print(f"CHUNK_SIZE                 : {chunk_size}")
    print(f"CUCOUNTER_CAPACITY         : {cucounter_capacity}")
    print(f"CUCOUNTER_CAPACITY_FACTOR  : {cucounter_capacity_factor}")
    print(f"COUNT_REVCOMPS             : {count_revcomps}")
    print(f"KMER_SIZE                  : {kmer_size}")
        
    keys = np.load(keys_filename)[:counter_size]
    keys = xp.asanyarray(keys)

    t = time.time()
    if counter_type == CuCounter:
        counter = counter_type(keys, capacity=cucounter_capacity, capacity_factor=cucounter_capacity_factor)
    else:
        counter = counter_type(keys)
    cp.cuda.runtime.deviceSynchronize()
    counter_init_t = time.time() - t

    chunk_creation_t = 0
    chunk_hashing_t = 0
    chunk_counting_t = 0

    t_ = time.time()

    for i, chunk in enumerate(bnp.open(fasta_filename, chunk_size=chunk_size), start=1):
        t = time.time()
        kmers = bnp.kmers.fast_hash(chunk.sequence, kmer_size, bnp.encodings.ACTGEncoding)
        cp.cuda.runtime.deviceSynchronize()
        chunk_hashing_t += time.time() - t

        t = time.time()
        counter.count(kmers.ravel(), count_revcomps=count_revcomps, kmer_size=kmer_size)
        cp.cuda.runtime.deviceSynchronize()
        chunk_counting_t += time.time() - t

        print(f"PROCESSING CHUNK: {i}", end="\r")
    print(f"PROCESSING CHUNK: {i}")

    total_t = time.time() - t_
    chunk_creation_t = total_t - (chunk_hashing_t + chunk_counting_t)

    print(">> TIMES")
    print(f"COUNTER_INIT_TIME      : {round(counter_init_t, 3)} seconds")
    print(f"CHUNK_CREATION_TIME    : {round(chunk_creation_t, 3)} seconds")
    print(f"CHUNK_HASHING_TIME     : {round(chunk_hashing_t, 3)} seconds")
    print(f"CHUNK_COUNTING_TIME    : {round(chunk_counting_t, 3)} seconds")
    print(f"TOTAL_FA2COUNTS_TIME   : {round(total_t, 3)} seconds")

    counts = counter[keys]
    print(counts[:10])


if __name__ == "__main__":
    if args.backend == "cupy":
        nps.set_backend(cp)
        bnp.set_backend(cp)

    array_module = np if args.backend == "numpy" else cp
    counter_type = nps.Counter if args.counter == "nps" else CuCounter
    
    time_data = pipeline(
            fasta_filename=fasta_filename, 
            keys_filename=keys_filename, 
            xp=array_module, 
            counter_type=counter_type, 
            counter_size=args.counter_size, 
            chunk_size=args.chunk_size,
            cucounter_capacity=args.cucounter_capacity,
            cucounter_capacity_factor=args.cucounter_capacity_factor,
            count_revcomps=args.count_revcomps,
            kmer_size=args.kmer_size)

