import time
import numpy as np
import cupy as cp

import bionumpy as bnp

from cucounter import Counter

FASTA = "/home/jorgen/data/fa/hg002_simulated_reads_15x.fa"
KEYS = "/home/jorgen/data/npy/uniquekmersACGT.npy"
NUM_KEYS = 30000000
CHUNK_SIZE = 10000000
KMER_SIZE = 31

#bnp.set_backend(cp)

counter_keys = np.load(KEYS)#[:NUM_KEYS]
counter = Counter(keys=counter_keys, capacity=int(NUM_KEYS*1.2), target_load_factor=0.45)
capacity = counter.capacity
load_factor = counter.load_factor
print(f"target_load_factor={0.45}")
print(f"num_keys={NUM_KEYS}")
print(f"capacity={capacity}")
print(f"load_factor={load_factor}")
exit(0)

t1 = time.perf_counter()

#for i, chunk in enumerate(bnp.open(FASTA).read_chunks(min_chunk_size=CHUNK_SIZE), start=1):
chunk_iter = bnp.io.parser.NumpyFileReader(open(FASTA, "rb"), bnp.TwoLineFastaBuffer).read_chunks(min_chunk_size=CHUNK_SIZE)
for i, chunk in enumerate(chunk_iter, start=1):
    kmers = bnp.sequence.get_kmers(bnp.as_encoded_array(chunk.get_data().sequence, bnp.DNAEncoding), KMER_SIZE).ravel().raw().astype(np.uint64)
    counter.count(kmers, count_revcomps=False, kmer_size=KMER_SIZE)
    
    print(f"counted {i} chunks", end="\r")
    if i >= 300:
        break
print(f"counted {i} chunks")

t2 = time.perf_counter()
elapsed = t2 - t1
print(f"Elapsed: {round(elapsed, 4)} seconds")
