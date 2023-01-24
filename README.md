## CuCounter: A Python *k*mer frequency counter object based on a massively parallel CUDA hash table

## Installation
CuCounter requires NumPy and CuPy. It also currently only supports Nvidia GPUs with CUDA.
CuCounter is currently not on PyPI, so in order to install CuCounter:
* clone the CuCounter repository
* use pip to install all necessary dependencies as well as CuCounter from inside the cloned repository
```Bash
git clone https://github.com/jorgenwh/cucounter.git
cd cucounter
pip install -r requirements.txt
pip install .
```

## Usage
All of CuCounter's methods (including its constructor) will accept either NumPy or CuPy arrays.
CuPy arrays are preferred as it circumvents having to copy memory back and fourth between the host and device.
NumPy is used in the example below, but the same code would work if NumPy had been replaced with CuPy.
```Python
from cucounter import Counter
import numpy as np

# Create a static set of 100 million unique 64-bit encoded kmers as keys for the counter
unique_kmers = np.arange(100000000, dtype=np.uint64)

# Create counter object
counter = Counter(keys=unique_kmers)

# Create a chunk of 200 million kmers to count
kmers = np.random.randint(low=0, high=0xFFFFFFFFFFFFFFFF, size=(200000000,), dtype=np.uint64)

# Count the observed kmer frequencies. Kmers not present in the original key set are ignored
counter.count(kmers)

# Fetch the observed frequencies for the original key set
counts = counter[unique_kmers] 

counts.dtype # np.uint32
counts.shape # (100000000,)
```

CuCounter also supports counting the reverse complements of *k*mers aswell as the original *k*mer.
```Python
counter.count(kmers, count_revcomps=True)
```
