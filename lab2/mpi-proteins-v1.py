import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI #dist. python librarty -> remember the mpiexec <-

def main():
    comm = MPI.COMM_WORLD  # Get the global communicator
    rank = comm.Get_rank() # Get the process ID
    size = comm.Get_size() # Total number of processes

    pattern = None
    if rank == 0:
        pattern = input("Enter the pattern to search for: ").upper()
        start = time.time()
        df = pd.read_csv('proteins.csv', usecols=['protid', 'hydrofob', 'sequence'])

        # divide the data into equal blocks using NumPy to distribute them
        chunk_size = len(df) // size + 1
        chunks = [df.iloc[i * chunk_size : (i + 1) * chunk_size].to_dict('list') for i in range(size)]
    else:
        chunks = None

    # broadcast: sends the text pattern to all workers
    pattern = comm.bcast(pattern, root=0)
    # scatter: distributes the dictionaries to each process
    local_data = comm.scatter(chunks, root=0)
    local_df = pd.DataFrame(local_data)
    # each process (worker) counts the occurrences in parallel within its own block
    local_df['occurrences'] = local_df['sequence'].str.count(pattern)
    local_matches = local_df[local_df['occurrences'] > 0]
    # gather: the main process collects the filtered sub-DataFrames from all of them
    gathered_matches = comm.gather(local_matches, root=0)
    
    if rank == 0:
        df_final = pd.concat(gathered_matches)
        
        # sort first by ocurrences, then by hydrofob
        df_sorted = df_final.sort_values(by=['occurrences', 'hydrofob'], ascending=[False, False])
        
        exec_time = time.time() - start
        print(f"\nExecution time (MPI - {size} processes): {exec_time:.4f} seconds")
        
        if not df_sorted.empty:
            # prints the id of the protein with max occurrences
            top_protein = df_sorted.iloc[0]
            print(f"Protein with max occurrences\n ID: {int(top_protein['protid'])} | Occurrences: {int(top_protein['occurrences'])} | Hydrofob: {top_protein['hydrofob']}")
            
            top_10 = df_sorted.head(10)
            plt.figure(figsize=(10, 6))
            plt.bar(top_10['protid'].astype(str), top_10['occurrences'], color='coral', edgecolor='black')
            plt.xlabel('Protein ID')
            plt.ylabel('Number of Occurrences')
            plt.title(f'Top 10 Proteins with pattern "{pattern}" (MPI)')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo occurrences found for the given pattern.")

if __name__ == '__main__':
    main()