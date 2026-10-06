import time, heapq
import matplotlib.pyplot as plt

pattern = input("Pattern to search: ").upper()
start = time.time()
matches = []  # (occurrences, hydrofob, protid) for every protein with at least one match
with open("proteins.csv") as f:
    next(f)  # skip header
    for line in f:
        protid, _, hydro, seq = line.rstrip("\n").split(",")
        n = seq.count(pattern)
        if n:
            matches.append((n, int(hydro), int(protid)))
top = heapq.nsmallest(10, matches, key=lambda m: (-m[0], -m[1]))  # more matches, then more hydrofob funcion tocha la vd
print(f"Execution time: {time.time() - start:.4f} s")
print("Protein with max occurrences:", top[0][2] if top else "no matches")
plt.bar([str(m[2]) for m in top], [m[0] for m in top])
plt.xlabel("Protein id"); plt.ylabel("Occurrences"); plt.title(f"Top 10 proteins for '{pattern}'")
plt.show()
