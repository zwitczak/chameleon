import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
import pymetis as metis


class Chameleon:
    def __init__(self, dataset, min_size=0.05, k_neighbors=2):
        """
        Initialize the Chameleon clustering algorithm.

        Parameters:
        - dataset (array-like): The dataset to cluster.
        - k_neighbors (int): The number of neighbors to consider when constructing the k-NN graph.
        - min_size (float): Minimum cluster size as a fraction of the dataset size.
        """
        # Add small perturbation to avoid exact overlaps
        dataset = dataset.astype(float)
        perturbation = np.random.normal(loc=0.0, scale=0.01, size=dataset.shape)
        dataset += perturbation
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.min_cluster_size = int(min_size * len(self.dataset))

        self.graph = None  # k-NN graph

    def _build_knn_graph(self):
        """
        Build the k-NN graph.
        """
        epsilon = 1e-10  # Small value to avoid division by zero

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.dataset)
        distances, indices = nbrs.kneighbors(self.dataset)

        # Create a graph with nodes and edges based on k-nearest neighbors
        graph = nx.Graph()
        for i, neighbors in enumerate(indices):
            for j, weight in zip(neighbors, distances[i]):
                if i != j:  # Avoid self-loops
                    graph.add_edge(i, j, weight=1/(weight+epsilon))
                    # print(f"Edge added: ({i}, {j}) with weight {1/(weight+epsilon):.2f}")
        return graph
    
    def _partition_graph(self):
        """
        Partition the graph into two parts.
        """
        
        adjacency_list = [list(self.graph.neighbors(node)) for node in self.graph.nodes]
        _, partitions = metis.part_graph(adjacency_list, nparts=self.min_cluster_size)

        # Group nodes into clusters based on partitions
        self.clusters = {i: [] for i in set(partitions)}
        for node, part in enumerate(partitions):
            self.clusters[part].append(node)
    
    def fit(self):
        """
        Fit the Chameleon algorithm.
        """

        if self.scheme is None:
            raise ValueError('You must initialize the scheme before fitting the model.')

        # Phase 1: Construct the k-NN graph and partition the it
        self._build_knn_graph()
        self._partition_graph()

        # Phase 2: Merge clusters dynamically
        self._merge_clusters()

        return self.clusters

if __name__ == '__main__':
    # Example dataset
    dataset = np.array([[1, 8], [1,8], [3, 4], [5, 6], [7, 8], [4,4]])
    chameleon = Chameleon(dataset=dataset, min_size=0.1, k_neighbors=3)

    # Build the k-NN graph
    knn_graph = chameleon._build_knn_graph()

    # Draw the graph
    # plt.figure(figsize=(8, 6))
    # pos = {i: dataset[i] for i in range(len(dataset))}  # Use dataset points as positions
    # nx.draw(knn_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

    # # Draw edge weights (distances)
    # edge_labels = nx.get_edge_attributes(knn_graph, 'weight')
    # formatted_edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()}
    # nx.draw_networkx_edge_labels(knn_graph, pos, edge_labels=formatted_edge_labels, font_size=8)

    # plt.title("k-NN Graph")
    # plt.show()
