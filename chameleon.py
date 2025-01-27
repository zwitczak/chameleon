from sklearn.neighbors import NearestNeighbors
import networkx as nx
from tqdm import tqdm
from helpers.graphtools import recursive_partition
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from helpers.clustertools import merge_score

class Chameleon:
    def __init__(self, min_cluster_size=0.05, k_neighbors=2, alpha=0.5):
        """
        Initialize the Chameleon clustering algorithm.

        Parameters:
        - dataset (array-like): The dataset to cluster.
        - k_neighbors (int): The number of neighbors to consider when constructing the k-NN graph.
        - min_size (float): Minimum cluster size as a fraction of the dataset size.
        """
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size

        self.scheme = 'function_product'  # Clustering scheme
        self.alpha = alpha   # Balance factor for product scheme

        # Attributes for intermediate results
        self.graph = None         # k-NN graph
        self.clusters = []        # Initial partitions (Phase 1)
        self.final_clusters = []  # Final clusters after merging (Phase 2)

    def _build_knn_graph(self, verbose=False):
        """
        Build the k-NN graph.
        """
        epsilon = 1e-10  # Small value to avoid division by zero

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.dataset)
        distances, indices = nbrs.kneighbors(self.dataset)
        
        if verbose:
            print(f"Building k-NN graph (k = {self.k_neighbors})...")

        # Create a graph with nodes and edges based on k-nearest neighbors
        graph = nx.Graph()
        iterpoints = tqdm(enumerate(indices), total=len(self.dataset)) if verbose else enumerate(indices)

        for i, neighbors in iterpoints:
            for j, weight in zip(neighbors, distances[i]):
                if i != j:  # Avoid self-loops
                    graph.add_edge(int(i), int(j), weight=int(1/(weight+epsilon)*1e4))

        graph.graph['edge_weight_attr'] = 'weight'
        self.graph = graph

    def _partition_graph(self, verbose=False):
        """
        Partition the graph iteratively until no valid partitions are possible.
        """

        if verbose:
            print("Partitioning the graph...")

        min_size = math.ceil(self.min_cluster_size*len(self.dataset)) 

        self.clusters = recursive_partition(self.graph, min_size, verbose=verbose)
    
        if verbose:
            print(f"Partitioning complete. Number of clusters: {len(self.clusters)}")
        
    def _merge_clusters(self, verbose=False):
        merges = 0
        cluster_graphs = [self.graph.subgraph(cluster.nodes).copy() for cluster in self.clusters]

        pbar = tqdm(desc="Merging Clusters. Operations", disable=not verbose)
        pbar2 = tqdm(desc="Number of merges", disable=not verbose)

        while True:
            best_merge_score = float('-inf')
            best_pair = None

            def compute_merge_score(i, j):
                score = merge_score(self.graph, cluster_graphs[i], cluster_graphs[j], self.alpha)
                return (i, j, score)

            pairs = [(i, j) for i in range(len(cluster_graphs)) for j in range(i + 1, len(cluster_graphs))]
            scores = []
            pbar2.update(1)

            with ThreadPoolExecutor() as executor:
                for i, j, score in executor.map(lambda pair: compute_merge_score(*pair), pairs):
                    scores.append((i, j, score))
                    pbar.update(1)

            for i, j, score in scores:
                if score > best_merge_score:
                    best_merge_score = score
                    best_pair = (i, j)

            if verbose:
                print(f"Best merge score: {best_merge_score}. Best pair: {best_pair}")
            if best_pair is None or best_merge_score <= 0:
                if verbose:
                    print("No valid merges found or merge score <= 0. Stopping merging.")
                break

            # Merge the best pair
            i, j = best_pair
            new_cluster_nodes = set(cluster_graphs[i].nodes).union(cluster_graphs[j].nodes)
            new_cluster = self.graph.subgraph(new_cluster_nodes).copy()

            cluster_graphs.pop(j)
            cluster_graphs.pop(i)
            cluster_graphs.append(new_cluster)
            merges += 1

        pbar.close()
        pbar2.close()
        if verbose:
            print(f"Finished merging. Number of merges: {merges}")
        self.final_clusters = cluster_graphs
        
    def fit(self, dataset, verbose=False):
        """
        Fit the Chameleon algorithm.
        """
        self.dataset = dataset

        if self.scheme is None:
            raise ValueError('You must initialize the scheme before fitting the model.')

        # Phase 1: Construct the k-NN graph and partition the it
        self._build_knn_graph(verbose=verbose)
        self._partition_graph(verbose=verbose)

        # Phase 2: Merge clusters dynamically
        self._merge_clusters(verbose=verbose)

        # return self.final_clusters
        return self.final_clusters
    

    def format_clusters_to_df(self):
        """
        Format the clusters into a dataframe.
        """
        if self.final_clusters is None or self.dataset is None:
            raise ValueError('You must fit the model before formatting the clusters.')
        
        data = []
        for cluster_idx, G in enumerate(self.final_clusters):
            for node in G.nodes:
                pos_x, pos_y = self.dataset.loc[node, ['x', 'y']]
                data.append((node, cluster_idx, pos_x, pos_y))
    
        transformed_df = pd.DataFrame(data, columns=['node_idx', 'cluster_idx', 'pos_x', 'pos_y'])
        return transformed_df
        