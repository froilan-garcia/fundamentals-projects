import time
import numpy as np


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

        # To calculte the distances (x-centroid)^2 efficiently inside the loop, we will do 
        # x^2 + centroid^2 - 2*x*centroid
        # This allows us to calculate x^2 only once and reduce the number of computations.
        x_squared = np.sum(X ** 2, axis = 1)

        # OPTIMIZATION LOOP 
        for _ in range(self.max_iter):
            # # Distance Calculation using numpy. Reshape X to (N, 1, D) and centroids to (1, K, D),
            # # in order to get a (N, K, D) array of differences between each point and each centroid.
            # diffs = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
            # # Squared Euclidean distance 
            # sq_distances = np.sum(diffs**2, axis=2)
            # # Assign each point to nearest centroid index
            # labels = np.argmin(sq_distances, axis=1)
            # # Calculate inertia 
            # inertia = np.sum(np.min(sq_distances, axis=1))

            # Distance calculation
            centroid_squared = np.sum(self.centroids_ ** 2, axis=1)
            sq_distances = (x_squared[:, None] + centroid_squared[None, :] - 2 * (X @ self.centroids_.T))
            sq_distances = np.maximum(sq_distances, 0)

            # Centroid asignation
            labels = np.argmin(sq_distances, axis=1)

            # Inertia caculation
            inertia = np.sum(np.min(sq_distances, axis=1))              

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