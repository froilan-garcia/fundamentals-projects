import os, heapq   #functions for file size and top
#no time functions needed, as MPI has its own timer
from mpi4py import MPI #dist. python librarty -> remember the mpiexec <-


comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

pattern = comm.bcast(input("Pattern to search: ").upper() if rank == 0 else None, root=0).encode() # 1 pattern for all workers, broadcasted from rank 0 to all ranks
start = MPI.Wtime()
fsize = os.path.getsize("proteins.csv") # we get the file size to split the work between ranks
begin = fsize * rank // size
end = fsize * (rank + 1) // size
with open("proteins.csv", "rb") as f:  # each rank reads only its own byte range, aligned to whole lines
    f.seek(max(begin - 1, 0)) #each rank puts the file pointer to its corresponding byte range, but we need to make sure we don't start in the middle of a line, so we go back one byte and read until the next newline
    f.readline()  # skip partial line (rank 0 skips the header) now every rank is aligned to a whole line, and we can read until the end of our byte range
    data = f.read(max(end - f.tell(), 0)) #we read the corresponding byte range
    data += f.readline() if data and not data.endswith(b"\n") else b"" #but we need to make sure we don't stop in the middle of a line, so we read until the next newline

local = [(line.count(pattern, line.rfind(b",") + 1), line) for line in data.splitlines()]
#now we count the occurrences of the pattern in each line, but we only want to count in the sequence part, which is after the last comma,
#so we use rfind to find the last comma and slice the line accordingly. We also keep the whole line for later use.

key = lambda m: (-m[0], -m[1])  # more matches, then more hydrofob

local = [(n, int(p[2]), int(p[0])) for n, line in local if n for p in [line.split(b",", 3)]] # now we collec the occurrences, hydrofob and protid for each line that has at least one match
#We also split the line into its parts using split with a maxsplit of 3, so we only get the first three parts (protid, enzyme and hydrofob) and we get only protid and hydrofob.

parts = comm.gather(heapq.nsmallest(10, local, key=key), root=0)  # only each rank's top 10 travels


if rank == 0:
    top = heapq.nsmallest(10, (m for p in parts for m in p), key=key) # we and get the top 10 from the merged list of top tens

    print(f"Execution time ({size} processes): {MPI.Wtime() - start:.4f} s", flush=True)
    print("Protein with max occurrences:", top[0][2] if top else "no matches", flush=True)  # flush: mpiexec buffers stdout

    import matplotlib.pyplot as plt #plots
    plt.bar([str(m[2]) for m in top], [m[0] for m in top])
    plt.xlabel("Protein id"); plt.ylabel("Occurrences")
    plt.title(f"Top 10 proteins for '{pattern.decode()}'")
    plt.show()
