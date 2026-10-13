import time, heapq
import matplotlib.pyplot as plt


def main():
    pattern = input("Pattern to search: ").upper().encode() # bytes, the file is read in binary mode
    start = time.time()
    matches = []  # (occurrences, hydrofob, protid) for every protein with at least one match
    with open("proteins.csv", "rb") as f:
        next(f)  # skip header
        for line in f:
            n = line.count(pattern, line.rfind(b",") + 1) # count only in the sequence (after the last comma)
            if n:
                p = line.split(b",", 3) # split only the lines that match
                matches.append((n, int(p[2]), int(p[0])))
    top = heapq.nsmallest(10, matches, key=lambda m: (-m[0], -m[1]))  # more matches, then more hydrofob
    print(f"Execution time: {time.time() - start:.4f} s")
    print("Protein with max occurrences:", top[0][2] if top else "no matches")
    plt.rcParams["font.size"] = 14  
    plt.figure(figsize=(10, 6))
    plt.bar([str(m[2]) for m in top], [m[0] for m in top], color="skyblue", edgecolor="black")
    plt.xlabel("Protein ID"); plt.ylabel("Number of Occurrences"); plt.title(f"Top 10 proteins for '{pattern.decode()}'")
    plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", linestyle="--", alpha=0.5); plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
