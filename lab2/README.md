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

**Dataset: **50,000 proteins**, generated with **seed = 123**. Pattern: `ABCD`.**

* Para `serial-proteins-v1.py` & `mpi-proteins-v1.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 0.1742 s | 33058 |
| MPI | 4 | 0.4222 s | 33058 |
| MPI | 16 | 0.6565 s | 33058 |

Protein with max occurrences -> ID: 33058 | Occurrences: 5 | Hydrofob: 204

 
**Speedup (4 processes):  0.413**
**Speedup (16 processes): 0.265**

- El tamaño del dataset es trivial. La versión serial es capaz de procesar todo ese texto en apenas 0.17 segundos. La carga de computación real es casi inexistente.
- El castigo de la arquitectura (Overhead): En las versión paralela, el proceso Master tiene que arrancar el entorno MPI, leer el archivo completo, calcular los cortes, transformar los trozos a diccionarios nativos, serializarlos y transmitirlos por la red virtual local (scatter), para finalmente recoger los resultados (gather). Todo ese esfuerzo logístico tarda unos 0.40 segundos. Estás gastando más tiempo repartiendo el trabajo que ejecutándolo.
- Al saltar a 16 procesos, el tiempo empeora aún más (0.65s). El Master ahora tiene el cuádruple de trabajo organizativo: tiene que fragmentar los datos en 16 partes, gestionar 16 canales de red y coordinar a 16 trabajadores, todo para repartir una tarea que solo dura 0.17 segundos. 
Esto explica los speedups<1

* Para `serial-proteins.py` & `mpi-proteins.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 0.1165 s | 33058 |
| MPI | 4 | 0.0280 s | 33058 |
| MPI | 16 | 0.0310 s | 33058 |


**Speedup (4 processes):  4.161**
**Speedup (16 processes): 3.758**

- Lograr un speedup superior al número de procesadores físicos (4.16 > 4) es un fenómeno real conocido como aceleración superlineal. Al dividir el archivo en 4 trozos más pequeños, cada núcleo procesa un fragmento de datos que cabe perfectamente en la **memoria caché L1/L2** de tu procesador, mientras que la versión secuencial satura la caché y tiene que ir constantemente a la RAM.
- El rendimiento empeora al usar 16 procesos (el *speedup* baja a 3.75) por un fenómeno llamado **I/O Contention**. El archivo de 50.000 proteínas pesa apenas unos megabytes. Si lanzas a 16 procesos a intentar leer del disco duro de tu ordenador exactamente en el mismo milisegundo, el controlador del disco se satura gestionando 16 punteros de lectura simultáneos para un archivo minúsculo. Además, el tiempo que tarda MPI en arrancar 16 *workers* penaliza cuando la tarea dura solo 0.03 segundos.

**Dataset: **5,000,000 proteins**, generated with **seed = 123**. Pattern: `ABCD`.**

* Para `serial-proteins-v1.py` & `mpi-proteins-v1.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 18.4197 s | 33058 |
| MPI | 4 | 27.7630 s | 33058 |
| MPI | 16 | 27.8387 s | 33058 |

Protein with max occurrences -> ID: 33058 | Occurrences: 5 | Hydrofob: 204
 
**Speedup (4 processes):  0.663**
**Speedup (16 processes): 0.662**

* Para `serial-proteins.py` & `mpi-proteins.py`

| Version | Processes | Execution time | Protein with max occurrences (ID) |
|---|---|---|---|
| Serial | 1 | 10.5279 s | 33058 |
| MPI | 4 | 2.3054 s | 33058 |
| MPI | 16 | 0.8707 s | 33058 |

**Speedup (4 processes):  4.567**
**Speedup (16 processes): 12.091**