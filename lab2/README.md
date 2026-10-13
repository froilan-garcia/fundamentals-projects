# Lab 2 – Distributed protein matching in Python (MPI)

Search a pattern typed on the keyboard in every protein sequence of `proteins.csv`, then show the 10 proteins with the most occurrences (ties broken by highest hydrofob) and the protein with the maximum.

- `serial-proteins.py`: serial version.
- `mpi-proteins.py`: parallel version with MPI (`mpi4py`). Each rank reads only its own byte range of the file (aligned to whole lines), computes its local top 10, and rank 0 gathers and merges them.

## Usage

Generate the dataset with `proteins-generator.py` (in `../lab1`), then run from the folder that contains `proteins.csv`:

```
python proteins-generator.py <numrows> <seed>
python serial-proteins.py
mpiexec -n <processes> python mpi-proteins.py
```

Without `-n`, `mpiexec` launches one process per logical core.

## Results

Dataset: **10,000,000 proteins**, generated with **seed = 67**. Pattern: `AB`. Run with `mpiexec` and no `-n`, which gave 16 processes on a 16-logical-core machine.

| Version | Processes | Execution time | Protein with max occurrences |
|---|---|---|---|
| Serial | 1 | 19.41 s | 3556501 |
| MPI | 16 | 3.17 s | 3556501 |
| MPI | 16 | 3.24 s | 3556501 |

**Speedup (16 processes): ~6.1x** (19.41 / 3.17).



##############################3

## Results (Miguel)

**Dataset: **10,000,000 proteins**, generated with **seed = 67**. Pattern: `AB`.**

* Para `serial-proteins.py` & `mpi-proteins.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 16.6311 s | 3556501 |
| MPI | 4 | 7.3173 s | 3556501 |
| MPI | 8 | 4.8929 s | 3556501 |
| MPI | 12 | 4.0126 s | 3556501 |
| MPI | 16 | 3.4687 s | 3556501 |


* Para `serial-proteins-v2.py` & `mpi-proteins-v2.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 23.4912 s | 3556501 |
| MPI | 4 | 9.5361 s | 3556501 |
| MPI | 8 | 5.1703 s | 3556501 |
| MPI | 12 | 3.7011 s | 3556501 |
| MPI | 16 | 3.2704 s | 3556501 |

Protein with max occurrences -> ID: 3556501 | Occurrences: 17 | Hydrofob: 91

**Speedup (4 processes): 2.46**
**Speedup (8 processes): 4.54**
**Speedup (12 processes): 6.35**
**Speedup (16 processes): 7.18**

######

**GUSTAFSSON LAW**

* `mpi-proteins-v2.py`

**Dataset 1M:**
MPI 1 proceso ->  3.6487 seconds
Protein with max occurrences: 316614

**Dataset 4M:** 
Serial         -> 8.9218 seconds
MPI 4 procesos -> 3.7895 seconds
Protein with max occurrences -> ID: 3556501 | Occurrences: 17 | Hydrofob: 91

**Dataset 8M:** 
Serial        -> 16.3171 seconds
MPI 8 procesos -> 4.1838 seconds
Protein with max occurrences: -> ID: 3556501 | Occurrences: 17 | Hydrofob: 91

**Dataset 12M:**
Serial         -> 23.7431 seconds
MPI 12 procesos -> 4.4306 seconds
Protein with max occurrences -> ID: 3556501 | Occurrences: 17 | Hydrofob: 91

**Dataset 16M:** 
Serial         -> 36.5458 seconds
MPI 16 procesos -> 5.1546 seconds
Protein with max occurrences -> ID: 3556501 | Occurrences: 17 | Hydrofob: 91


* `mpi-proteins.py`

**Dataset 1M:** 
Serial        -> 1.5964 second
MPI 1 proceso -> 2.7463 seconds
Protein with max occurrences: 316614

**Dataset 4M:** 
Serial         -> 6.2027 seconds
MPI 4 procesos -> 3.6029 seconds
Protein with max occurrences: 3556501

**Dataset 8M:**
Serial        -> 12.2075 seconds
MPI 8 procesos -> 3.6757 seconds
Protein with max occurrences: 3556501

**Dataset 12M:**
Serial         -> 18.1499 seconds
MPI 12 procesos -> 4.3628 seconds
Protein with max occurrences: 3556501

**Dataset 16M:** 
Serial         -> 24.8704 seconds
MPI 16 procesos -> 6.5911 seconds
Protein with max occurrences: 3556501