import time
import heapq
import matplotlib.pyplot as plt

def main():
    pattern = input("Enter the pattern to search for: ").upper()
    start = time.time()
    matches = []  # (occurrences, hydrofob, protid) for every protein with at least one match
    
    with open("proteins.csv", "r") as f:
        next(f)  # skip header
        for line in f:
            protid, _, hydro, seq = line.rstrip("\n").split(",")
            n = seq.count(pattern)
            if n > 0:
                matches.append((n, int(hydro), int(protid)))
    
    # sort first by ocurrences (-m[0]), then by hydrofob (-m[1])
    top_10 = heapq.nsmallest(10, matches, key=lambda m: (-m[0], -m[1])) # more matches, then more hydrofob funcion tocha la vd
    
    exec_time = time.time() - start
    print(f"\nExecution time: {exec_time:.4f} seconds")
    
    if top_10:
        # prints the id of the protein with max occurrences (max hydrofob if tied)
        best_match = top_10[0]
        print(f"Protein with max occurrences:\n ID: {best_match[2]} | Occurrences: {best_match[0]} | Hydrofob: {best_match[1]}")
        # print a barchart of occurrences for the 10 proteins with more matches
        prot_ids = [str(m[2]) for m in top_10]
        occurrences = [m[0] for m in top_10]
        
        plt.figure(figsize=(10, 6))
        plt.bar(prot_ids, occurrences, color='skyblue', edgecolor='black')
        plt.xlabel('Protein ID')
        plt.ylabel('Number of Occurrences')
        plt.title(f'Top 10 Proteins with pattern "{pattern.decode()}"')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo occurrences found for the given pattern.")

if __name__ == '__main__':
    main()