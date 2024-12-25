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

        self.graph = None
        self.clusters = None
        self.scheme = None
        self.alpha = None
        self.min_rc = None
        self.mic_ri = None

    def init_function_product_scheme(self, alpha):
        """
        Initialize the function product scheme parameter.

        Parameters:
        - alpha (float): The balance factor between relative closeness and relative interconnectivity.
        """
        self.scheme = 'function_product'
        self.alpha = alpha

    def init_values_limits_scheme(self, rc, ri):
        """
        Initialize the values limit scheme parameters.

        Parameters:
        - rc (float): Minimum relative closeness for merging clusters.
        - ri (float): Minimum relative interconnectivity for merging clusters.
        """
        self.scheme = 'values_limits'
        self.rc = rc
        self.ri = ri

    def _build_knn_graph(self):
        """
        Build the knn graph.
        """

        self.graph = NearestNeighbors(n_neighbors=self.k_neighbors).fit(self.dataset)
        distances, indices = self.graph.kneighbors(self.dataset)

        # Create a graph with nodes and edges based on k-nearest neighbors
        graph = nx.Graph()
        for i, neighbors in enumerate(indices):
            for j, neighbor in enumerate(neighbors):
                if i != neighbor: 
                    graph.add_edge(i, neighbor, weight=1 / distances[i, j])

        return graph
    
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
        
        # Step 1: Construct the k-nearest neighbor graph
        self.graph = self._build_knn_graph()
        # Step 2: Initial partitioning into small clusters
        initial_subclusters = self._initial_partitioning()
        # Step 3: Merge clusters dynamically
        self._merge_clusters(initial_subclusters)
