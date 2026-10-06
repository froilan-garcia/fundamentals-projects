import time
import numpy as np
import multiprocessing as mp

# To parallelize the process of assigning points to the centroids, we create a function that does this on a subset of all points.
def assign_points_to_centroids(differences):
    distances = np.sum(differences**2, axis=2)
    return np.argmin(distances,axis=1), np.sum(np.min(distances, axis=1))


class KMeansCustom:
  
    # Implementation of K-Means algorithm from scracth in order to paralellize and optimize correctly

    # This algorithm is a no supervised learning algorithm that partitions a dataset into K distinct clusters, 
    # and mathematically it tries to minimize the sum of squared distances between data points and their respective cluster centroids.
    # This metric is called inertia. The algorithm iteratively tries to reduce inertia until a maximum number of iterations is reached 
    # or until the change in inertia is below a specified tolerance.


    def __init__(self, k=3, max_iter=300, tol=1e-4, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        #self.fit_time = 0.0
        # Notation: _suffix indicates that the attribute is set after fitting the model, following scikit-learn conventions.



    def fit(self, X):
        # Fits the K-Means algorithm to input data X of shape (N_rows, N_columns).
        X = np.asarray(X)  # Ensure input is a numpy array

        start_fit = time.time() # We are going to measure the time spent by the algorithm
        np.random.seed(self.seed)
        n_samples, n_features = X.shape # Dimensions of input data

        # RANDOM INICIALIZATION OF CENTROIDS 
        # Select k random distinct row indices without replacement
        initial_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids_ = X[initial_indices].copy()
        # Using .copy() creates an independent duplicate of the selected rows in memory so that updating the centroids 
        # during K-Means will not alter the original dataset X.
        
        prev_inertia = float("inf") # Initialize inertia to a very large value to ensure the first iteration does not fail.

        # OPTIMIZATION LOOP 
        for _ in range(self.max_iter):
            # Distance Calculation using numpy. Reshape X to (N, 1, D) and centroids to (1, K, D),
            # in order to get a (N, K, D) array of differences between each point and each centroid.
            diffs = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]

            n_processes = min(mp.cpu_count(), n_samples)
            block_size = max(1, n_samples // n_processes)
            chunk_starts = range(0, n_samples, block_size)
            chunks = [diffs[i:i + block_size] for i in chunk_starts]

            with mp.Pool(n_processes) as pool:
                results = pool.map(assign_points_to_centroids, chunks)

            labels = np.empty(n_samples, dtype=int)
            inertia = 0.0
            offset = 0

            for chunk_labels, chunk_inertia in results:
                n = len(chunk_labels)
                labels[offset:offset + n] = chunk_labels
                inertia += chunk_inertia
                offset += n

            # Centroid Update (we calculate the new mean of cluster points)
            new_centroids = np.zeros_like(self.centroids_)
            for c_id in range(self.k):
                cluster_points = X[labels == c_id] # Extract proteins assigned to cluser c_id
                if len(cluster_points) > 0:
                    new_centroids[c_id] = cluster_points.mean(axis=0)
                else:
                    # Empty clusters: retain previous position
                    new_centroids[c_id] = self.centroids_[c_id]
                    
            self.centroids_ = new_centroids
            # Convergence Check
            if abs(prev_inertia - inertia) <= self.tol:
                break

            prev_inertia = inertia
            

        self.labels_ = labels
        self.inertia_ = inertia
        #  self.fit_time = time.time() - start_fit
        return self