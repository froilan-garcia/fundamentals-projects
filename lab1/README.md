# Lab 1 – K-Means parallelization in Python

K-Means clustering (on `enzyme` and `hydrofob`) over the proteins dataset, in serial, `multiprocessing` and `threading` versions, comparing execution times and speedup. The full analysis is in `lab1_report.pdf`.

## Files

- `lab1-proteins-serial.py`: serial version.
- `lab1-proteins-mp.py`: multiprocessing version, one process per `k` when computing the inertia. This is the fastest of the multiprocessing variants.
- `lab1-proteins-th.py`: threading version. `Different Number of Threads/` has the same script with 4, 8, 12 and 16 threads.
- `kmeans_scratch.py`: K-Means implementation used by the scripts.
- `Other Multiprocessing Implementations/`: alternative parallelizations that turned out slower (centroid update, point assignment, optimal `k`, preprocessing).

## Usage

```
python proteins-generator.py 2000000 <seed>
python lab1-proteins-serial.py
python lab1-proteins-mp.py
python lab1-proteins-th.py
```

## Results

Dataset: 2,000,000 proteins, seed = 123, on a 20-core machine.

| Version | Time | Speedup |
|---|---|---|
| Serial | 162.8 s | 1.0x |
| Multiprocessing | 43.1 s | 3.8x |
| Threading (12 threads) | 35.4 s | 4.6x |
