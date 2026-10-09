import time
import pandas as pd
import matplotlib.pyplot as plt

def main():
    pattern = input("Enter the pattern to search for: ").upper()
    start = time.time()

    df = pd.read_csv('proteins.csv', usecols=['protid', 'hydrofob', 'sequence']) # 'hydrofob' to solve ties
    
    # look for occurrences of the pattern string inside each protein sequence.
    df['occurrences'] = df['sequence'].str.count(pattern)
    df_matches = df[df['occurrences'] > 0]
    
    # sort by ocurrence first, then by hydrofob
    df_sorted = df_matches.sort_values(by=['occurrences', 'hydrofob'], ascending=[False, False])
    
    exec_time = time.time() - start
    print(f"\nExecution time (serial): {exec_time:.4f} seconds")
    
    if not df_sorted.empty:
        # prints the id of the protein with max occurrences
        top_protein = df_sorted.iloc[0]
        print(f"Protein with max occurrences:\n ID: {int(top_protein['protid'])} | Occurrences: {int(top_protein['occurrences'])} | Hydrofob: {top_protein['hydrofob']}")
        
        # prints a barchart of occurrences for the top 10 proteins
        top_10 = df_sorted.head(10)
        
        plt.figure(figsize=(10, 6))
        plt.bar(top_10['protid'].astype(str), top_10['occurrences'], color='skyblue', edgecolor='black')
        plt.xlabel('Protein ID', fontsize=12)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.title(f'Top 10 Proteins with pattern "{pattern}" (Serial)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo occurrences found for the given pattern.")

if __name__ == '__main__':
    main()