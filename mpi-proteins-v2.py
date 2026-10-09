import os
import heapq
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD # Get the global communicator
    rank = comm.Get_rank() # Get the process ID
    size = comm.Get_size() # Total number of processes
    pattern = comm.bcast(input("Enter pattern to search: ").upper().encode() if rank == 0 else None, root=0)

    start = MPI.Wtime()
    fsize = os.path.getsize("proteins.csv") # we get the file size to split the work between ranks
    begin = fsize * rank // size
    end = fsize * (rank + 1) // size

    local_matches = []
    with open("proteins.csv", "rb") as f: # each rank reads only its own byte range, aligned to whole lines
        if rank != 0:
            f.seek(max(begin - 1, 0)) #each rank puts the file pointer to its corresponding byte range, but we need to make sure we don't start in the middle of a line, so we go back one byte and read until the next newline
            f.readline()  # skip partial line (rank 0 skips the header) now every rank is aligned to a whole line, and we can read until the end of our byte range
        else:
            f.readline()  # rank 0 skips header
        
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            
            # look for occurrences
            last_comma_idx = line.rfind(b",")
            if last_comma_idx != -1:
                n_occurrences = line.count(pattern, last_comma_idx + 1)
                if n_occurrences > 0:
                    parts = line.split(b",", 3)
                    protid = int(parts[0])
                    hydro = int(parts[2])
                    local_matches.append((n_occurrences, hydro, protid))

    key = lambda m: (-m[0], -m[1]) # more matches, then more hydrofob
    local_top_10 = heapq.nsmallest(10, local_matches, key=key) # only each rank's top 10 travels
    gathered_tops = comm.gather(local_top_10, root=0) # gather

    if rank == 0:
        flat_tops = [match for sublist in gathered_tops for match in sublist]
        top_10 = heapq.nsmallest(10, flat_tops, key=key)

        exec_time = MPI.Wtime() - start
        print(f"\nExecution time (MPI - {size} processes): {exec_time:.4f} seconds", flush=True)

        if top_10:
            best_match = top_10[0]
            print(f"Protein with max occurrences\n ID: {best_match[2]} | Occurrences: {best_match[0]} | Hydrofob: {best_match[1]}", flush=True)
            
            # print a barchart
            prot_ids = [str(m[2]) for m in top_10]
            occurrences = [m[0] for m in top_10]

            plt.figure(figsize=(10, 6))
            plt.bar(prot_ids, occurrences, color='coral', edgecolor='black')
            plt.xlabel('Protein ID')
            plt.ylabel('Number of Occurrences')
            plt.title(f'Top 10 Proteins with pattern "{pattern.decode()}" (MPI)')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo occurrences found for the given pattern.", flush=True)

if __name__ == '__main__':
    main()