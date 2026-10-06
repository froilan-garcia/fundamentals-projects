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
