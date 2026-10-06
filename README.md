# Fundamentals Projects

Labs for *Technological Fundamentals in the Big Data World* (Master in Big Data, UC3M). Both labs work on the synthetic proteins dataset produced by `proteins-generator.py`.

| Lab | Topic | Parallelism |
|---|---|---|
| [lab1](lab1/) | K-Means clustering of proteins | `multiprocessing` and `threading` |
| [lab2](lab2/) | Pattern matching in protein sequences | MPI (`mpi4py`) |

Datasets (`proteins.csv`) are not committed. Generate them with:

```
python proteins-generator.py <numrows> <seed>
```
