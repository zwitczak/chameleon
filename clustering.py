from sklearn.neighbors import NearestNeighbors 
import networkx as nx
import pymetis as metis


class Chameleon:
    def __init__(self, dataset, min_size = 0.05, k_neighbors=2):
        """
        Initialize the Chameleon clustering algorithm.

        Parameters:
        - dataset (array-like): The dataset to cluster.
        - k_neighbors (int): The number of neighbors to consider when constructing the knn graph.
        - min_cluster_size (int): The minimum number of samples in a cluster after Phase I.
        """
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.min_cluster_size = int(min_size * len(self.dataset))

        self.scheme = None  # Clustering scheme
        self.alpha = None   # Balance factor for product scheme
        self.min_rc = None  # Minimum relative closeness
        self.mic_ri = None  # Minimum relative interconnectivity

        # Attributes for intermediate results
        self.graph = None         # k-NN graph
        self.clusters = None      # Initial partitions (Phase 1)
        self.final_clusters = []  # Final clusters after merging (Phase 2)

    def set_scheme(self, scheme, **kwargs):
        """
        Set the merging scheme and its parameters.

        Parameters:
        - scheme (str): The merging scheme. Options are 'function_product' or 'values_limits'.
        - kwargs: Additional parameters for the scheme:
            * If 'function_product': Requires 'alpha' (float).
            * If 'values_limits': Requires 'rc' (float) and 'ri' (float).
        """

        if scheme not in ['function_product', 'values_limits']:
            raise ValueError("Invalid scheme. Choose either 'function_product' or 'values_limits'.")
        
        self.scheme = scheme

        if scheme == "function_product":
            if 'alpha' not in kwargs:
                raise ValueError("You must provide the 'alpha' parameter for the function product scheme.")
            self.alpha = kwargs['alpha']

        elif scheme == "values_limits":
            if 'rc' not in kwargs or 'ri' not in kwargs:
                raise ValueError("You must provide the 'rc' and 'ri' parameters for the values limits scheme.")
            self.rc = kwargs['rc']
            self.ri = kwargs['ri']

    def __calculate_relative_metric(self, cluster1, cluster2):
        """
        Calculate the relative closeness and relative interconnectivity between two clusters.

        Parameters:
        - cluster1 (set): The first cluster.
        - cluster2 (set): The second cluster.
        """
        pass

    def _build_knn_graph(self):
        """
        Build the knn graph.
        """

        self.graph = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.dataset)
        distances, indices = self.graph.kneighbors(self.dataset)

        # Create a graph with nodes and edges based on k-nearest neighbors
        graph = nx.Graph()
        for i, neighbors in enumerate(indices):
            for j, weight in zip(neighbors, distances[i]):
                if i != j: # Avoid self-loops
                    graph.add_edge(i, j, weight=1 / distances[i, j])

        return graph
    
    def _partition_graph(self):
        """
        Partition the graph into two parts.
        """
        subgraph = self._bisect_graph(self.graph)
        print(subgraph)
    
    def _bisect_graph(self, subgraph):
        """
        Bisect the graph into two parts such that the edge cut is minimized.
        """
        if len(subgraph.nodes) < 2:
            return [set(subgraph.nodes)]
        
        # Partition the graph using hMETIS
        (edgecuts, parts) = metis.part_graph(subgraph, 2)

        print(parts)

    
    def _initial_partitioning(self):
        """
        Perform the  partitioning of the dataset to find initial subclusters using hMETIS algorithm.
        Partitioning minimizes edge cut and maximizes internal density.
        Multi-level graph partitioning algorithms to partition the graph.
        Returns:
        - initial_subclusters (list): The initial subclusters.
        """

    def _merge_clusters(self):
        """
        Merge clusters based on the scheme.
        """

        pass

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

        return self.final_clusters
