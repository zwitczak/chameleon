from tqdm import tqdm
import metis

def recursive_partition(graph, min_cluster_size: int, verbose=False):
    """
    Partition a graph into a given number of parts using the METIS library.

    Parameters
    ----------
    graph : nx.Graph
        The graph to partition.
    num_parts : int
        The number of parts to partition the graph into.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    parts : list of int
        The partition of each node in the graph.
    """
    if len(graph.nodes) <= min_cluster_size:
        if verbose:
            print("Cluster size is below the minimum threshold. Stopping partition.")
        return [graph]

    edgecuts, parts = metis.part_graph(graph, nparts=2)

    # Group nodes by their partition
    partitions = [[], []]
    for node, part in zip(graph.nodes, parts):
        partitions[part].append(node)

    # Create subgraphs for each partition
    subgraph_1 = graph.subgraph(partitions[0]).copy()
    subgraph_2 = graph.subgraph(partitions[1]).copy()


    # Check if either subgraph is smaller than the minimum cluster size
    if len(subgraph_1.nodes) < min_cluster_size or len(subgraph_2.nodes) < min_cluster_size:
        if verbose:
            print("One of the partitions is below the minimum cluster size. Stopping further partitioning.")
        return [graph]
    
    # Recursively partition each subgraph
    clusters = []
    clusters.extend(recursive_partition(subgraph_1, min_cluster_size=min_cluster_size))
    clusters.extend(recursive_partition(subgraph_2, min_cluster_size=min_cluster_size))
    return clusters


def internal_interconnectivity(graph, cluster):
    """Calculate the internal interconnectivity of a cluster."""
    return sum(
            graph[u][v]['weight'] for u, v in cluster.edges
        )

def relative_interconnectivity(graph, cluster1, cluster2):
    """Calculate the relative interconnectivity between two clusters."""
    cut_weight = sum(
            graph[u][v]['weight']
            for u in cluster1.nodes
            for v in cluster2.nodes
            if graph.has_edge(u, v)
        )
    return cut_weight / (internal_interconnectivity(graph, cluster1) + internal_interconnectivity(graph, cluster2))

def internal_closeness(graph, cluster):
        """Calculate the internal closeness of a cluster."""
        total_weight = sum(
            graph[u][v]['weight'] for u, v in cluster.edges
        )
        return total_weight / len(cluster.edges) if len(cluster.edges) > 0 else 0

def relative_closeness(graph, cluster1, cluster2):
    """Calculate the relative closeness between two clusters."""
    cut_weight = sum(
        graph[u][v]['weight']
        for u in cluster1.nodes
        for v in cluster2.nodes
        if graph.has_edge(u, v)
    )
    return cut_weight / (
        internal_closeness(graph, cluster1) * internal_closeness(graph, cluster2)
    ) if internal_closeness(graph, cluster1) > 0 and internal_closeness(graph, cluster2) > 0 else 0

def merge_score(graph, cluster1, cluster2, alpha=0.5):
    """Calculate the merge score between two clusters."""
    ri = relative_interconnectivity(graph, cluster1, cluster2)
    rc = relative_closeness(graph, cluster1, cluster2)
    return ri * (rc ** alpha)