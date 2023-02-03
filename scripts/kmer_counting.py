import sys
import time
import numpy as np
import cupy as cp

import bionumpy as bnp

from cucounter import Counter

FASTA = "/home/jorgen/data/fa/hg002_simulated_reads_15x.fa"
KEYS = "/home/jorgen/data/npy/uniquekmersACGT.npy"
NUM_KEYS = 81231751
KMER_SIZE = 31


bnp.set_backend(cp)

counter_keys = np.load(KEYS)[:NUM_KEYS]

TARGET_LOAD_FACTOR = None
CHUNK_SIZE = 10000000 
counter = Counter(keys=counter_keys, target_load_factor=TARGET_LOAD_FACTOR)

chunk_iter = bnp.io.parser.NumpyFileReader(
        open(FASTA, "rb"), 
        bnp.TwoLineFastaBuffer).read_chunks(min_chunk_size=CHUNK_SIZE)

t1 = time.perf_counter()

for i, chunk in enumerate(chunk_iter, start=1):
    kmers = bnp.sequence.get_kmers(bnp.as_encoded_array(chunk.get_data().sequence, bnp.DNAEncoding), KMER_SIZE).ravel().raw().astype(np.uint64)
    counter.count(kmers, count_revcomps=True, kmer_size=KMER_SIZE)
    print(f"counted {i} chunks", end="\r")

print(f"counted {i} chunks")
t2 = time.perf_counter()
elapsed = t2 - t1
print(f"Total time elapsed: {round(elapsed, 4)} seconds")
