import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kmeans_scratch_mp2 import KMeansCustom  # Import custom K-Means class
import multiprocessing as mp

# IMPLEMENTATION USING MULTIPROCESSING
# The multiprocessing affects only the kmeans_scratch_mp2 class.

# This function transforms the feature "sequence" into "sequence length".
def preprocess_dataset(df):
    # In order to work with the feature sequence, we work with length sequence.
    df["seq_length"] = df["sequence"].str.len()
    X_raw = df[["enzyme", "hydrofob"]].to_numpy()
   
    
    # Min - Max [0,1] normalization
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_norm = (X_raw - X_min) / (X_max - X_min)
    
    # Add normalized columns to the dataframe to draw the graphs
    df["enzyme_norm"] = X_norm[:, 0]
    df["hydrofob_norm"] = X_norm[:, 1]
    return df, X_norm

def compute_wcss(X, k_range, seed=123):
    # Function that computes inertia for a range of k values using the custom KMeans implementation.
    wcss_list = []
    for k in k_range:
        km = KMeansCustom(k=k, seed=seed)
        km.fit(X)
        wcss_list.append(km.inertia_)
    return wcss_list
# In order to find the optimal number of clusters (k), we are going to use elbow method, which consists of obtaining
# the value of k where inertia convexity changes. For doing that, we are going to find the maximum distance from the line that
# connects the first and last points of the inertia curve. This point is the optimal k. 

# This function finds the optimal k based on the elbow graph.
# Reminder: The optimal k is the one which lies furthest away 
# from the line joining the first and the last ks.
def find_optimal_k_elbow(k_values, wcss_list):
    # Finds optimal k using normalized perpendicular distance to the secant line
    k_arr = np.array(k_values, dtype=float)
    wcss_arr = np.array(wcss_list, dtype=float)

    # Min-Max normalization to standard scale [0, 1]
    x = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    y = (wcss_arr - wcss_arr.min()) / (wcss_arr.max() - wcss_arr.min())

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    distances = []
    for i, j in zip(x, y):
        p = np.array([i, j])
        # Perpendicular distance formula
        dist = np.abs((p2[0]-p1[0])*(p1[1]-p[1])-(p1[0]-p[0])*(p2[1]-p1[1])) / np.linalg.norm(p2 - p1)
        distances.append(dist)

    optimal_idx = np.argmax(distances)
    return k_values[optimal_idx], distances

# This function finds the cluster with the highest sequence length 
# and returns its id, max sequence length, and average sequence length.
def get_cluster_highest_seq_length(df):
    # Sequence analysis: Finds cluster with the longest sequence.
    # Tie-breaker rule: if several have maximum length, pick the one with maximum Hydrofob.

    sorted_df = df.sort_values(by=["seq_length", "hydrofob"], ascending=[False, False])
    max_seq_row = sorted_df.iloc[0]

    target_cluster_id = int(max_seq_row["cluster"])
    target_cluster_df = df[df["cluster"] == target_cluster_id]
    avg_seq_length = target_cluster_df["seq_length"].mean()

    return target_cluster_id, max_seq_row["protid"], max_seq_row["seq_length"], avg_seq_length

# This function is used to plot the green line in the elbow plot
def project_point_to_segment(x1, y1, x2, y2, x0, y0):
    # Project (x0, y0) on the line defined by [(x1, y1), (x2, y2)]

    AB = np.array([x2 - x1, y2 - y1])
    AP = np.array([x0 - x1, y0 - y1])
    denom = np.dot(AB, AB)
    if denom == 0:
        return x1, y1
    t = np.dot(AP, AB) / denom
    t = max(0, min(1, t))
    x_proj = x1 + t * (x2 - x1)
    y_proj = y1 + t * (y2 - y1)
    return x_proj, y_proj

# This funtion draws the elbow plot (the first plot)
def plot_elbow(ax, k_range, wcss_list, optimal_k):
    k_arr = np.array(k_range, dtype=float)
    wcss_arr = np.array(wcss_list, dtype=float)

    # Normalisation for calculating the perpendicular distance to the line connecting the first and last points of the inertia curve.
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    wcss_norm = (wcss_arr - wcss_arr.min()) / (wcss_arr.max() - wcss_arr.min())

    opt_idx = k_range.index(optimal_k)

    # We use auxiliar function defined before in order to make the projections
    xn_proj, yn_proj = project_point_to_segment(k_norm[0], wcss_norm[0], k_norm[-1], wcss_norm[-1], k_norm[opt_idx], wcss_norm[opt_idx])

    # We revert normalisaton to get the values of the projected point
    x_proj = xn_proj * (k_arr.max() - k_arr.min()) + k_arr.min()
    y_proj = yn_proj * (wcss_arr.max() - wcss_arr.min()) + wcss_arr.min()

    # Plot 
    ax.plot(k_range, wcss_list, "bo-", label="Inertia")
    ax.plot([k_range[0], k_range[-1]], [wcss_list[0], wcss_list[-1]], "r--", label="Reference Line")

    # Line which joins optimal k and projected point
    ax.plot([optimal_k, x_proj], [wcss_list[opt_idx], y_proj], color="g", linestyle="--", linewidth=2, label=f"Perpendicular Dist (k={optimal_k})")
    ax.set_title("Elbow Graph")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.legend(loc="upper right")
    ax.grid(True)

# This function draws the cluster plot (the second plot)
def plot_clusters(ax, df, final_km, optimal_k):
    # Gráfica de dispersión usando las variables normalizadas
    sns.scatterplot(
        data=df, 
        x="enzyme_norm", 
        y="hydrofob_norm", 
        hue="cluster", 
        palette="viridis", 
        alpha=0.6, 
        ax=ax, 
        s=20
    )
    # Centroides en escala normalizada [0, 1]
    ax.scatter(
        final_km.centroids_[:, 0], 
        final_km.centroids_[:, 1], 
        c="red", 
        marker="X", 
        s=200, 
        label="Centroids"
    )
    ax.set_title(f"Clustering K-Means (k={optimal_k}) [Normalized]")
    ax.set_xlabel("Enzyme (Normalized)")
    ax.set_ylabel("Hydrofob (Normalized)")
    ax.legend(loc="upper right")

def plot_centroids_heatmap(ax, final_km, optimal_k):
    centroid_df = pd.DataFrame(
        final_km.centroids_, 
        columns=["Enzyme (Norm)", "Hydrofob (Norm)"], 
        index=[f"Cluster {i}" for i in range(optimal_k)]
    )
    sns.heatmap(centroid_df, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
    ax.set_title("Heatmap of Centroids")


# Multiprocessing execution

if __name__ == "__main__":
    # Start global timer
    start = time.time()
    print("Parallel (multiprocessing) execution started...")

    # Load and preprocess dataset
    t_start_preprocess = time.time()
    df = pd.read_csv("proteins.csv")
    df, X = preprocess_dataset(df)
    t_preprocess = time.time() - t_start_preprocess
    print("Successfully loaded 'proteins.csv'.")

    # Calculate optimal k using Elbow Method
    print("\nCalculating optimal k using the Elbow Method (multiprocessing)...")
    t_start_inertia = time.time()
    k_range = list(range(1, 16))
    wcss_list = compute_wcss(X, k_range, seed=123)
    t_inertia = time.time() - t_start_inertia
    t_start_optimal = time.time()
    optimal_k, _ = find_optimal_k_elbow(k_range, wcss_list)
    t_optimal = time.time() - t_start_optimal
    print(f"Optimal number of clusters (k) found: {optimal_k}")

    # Run K-Means with optimal k (serial)
    print(f"\nClustering data into {optimal_k} clusters (manual K-Means)...")
    t_start_final_fit = time.time()
    final_km = KMeansCustom(k=optimal_k, seed=123)
    final_fit = final_km.fit(X)
    df["cluster"] = final_fit.labels_
    t_final_fit = time.time() - t_start_final_fit
    print("Clustering complete.")

    # Sequence analysis
    t_start_sequence = time.time()
    print("\n--- Analysis of Cluster with Highest Total Sequence Length ---")
    target_cluster_id, max_protid, max_seq_len, avg_seq_len = get_cluster_highest_seq_length(df)
    print(f"Cluster ID with longest sequence : {target_cluster_id}")
    print(f"  - Protein ID                  : {max_protid}")
    print(f"  - Maximum sequence length     : {max_seq_len}")
    print(f"  - Average sequence length     : {avg_seq_len:.2f}")
    t_sequence = time.time() - t_start_sequence 

    total_execution_time = time.time() - start

    print("Generating plots...")

    # Build figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_elbow(axes[0], k_range, wcss_list, optimal_k)
    plot_clusters(axes[1], df, final_fit, optimal_k)
    plot_centroids_heatmap(axes[2], final_fit, optimal_k)
    plt.tight_layout()

    # Time statistics
    print("\n--- Time statistics ---")
    print(f"\n Time spent in preprocessing: {t_preprocess:.4f}  ({(t_preprocess / total_execution_time) * 100:.2f}%)")
    print(f"\n Time spent calculating the inertia for every k: {t_inertia:.4f} s ({(t_inertia / total_execution_time) * 100:.2f}%)")
    print(f"\n Time spent finding the optimal k: {t_optimal:.4f} s ({(t_optimal / total_execution_time) * 100:.2f}%)")
    print(f"\n Time spent calculating the final fit: {t_final_fit:.4f} s ({(t_final_fit / total_execution_time) * 100:.2f}%)")
    print(f"\n Time spent analyzing the cluster with highest sequence length: {t_sequence:.4f} s ({(t_sequence / total_execution_time) * 100:.2f}%)")
    print(f"\n Total program runtime (excluding ploting): {total_execution_time:.4f} seconds.")
    
    print("\nDisplaying plots. Close plot windows to exit.")
    plt.show()