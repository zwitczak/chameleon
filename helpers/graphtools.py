from tqdm import tqdm
import metis
import networkx as nx
import numpy as np

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
    clusters = [graph]
    result = []

    while clusters:
        min_g = None
        min_edgecut = float('inf')

        for idx, g in enumerate(clusters):
            # Calculate possible min edgecut and decide whether to partition
            if len(g.nodes) > min_cluster_size:
                edgecuts, parts = metis.part_graph(g, nparts=2)
                if edgecuts < min_edgecut:
                    min_edgecut = edgecuts
                    min_g = idx

        if min_g is None:
            break

        # Partition the selected graph
        g = clusters.pop(min_g)
        edgecuts, parts = metis.part_graph(g, nparts=2)

        partitions = [[], []]

        for node, part in zip(g.nodes, parts):
            partitions[part].append(node)

        subgraph_1 = g.subgraph(partitions[0]).copy()
        subgraph_2 = g.subgraph(partitions[1]).copy()

        # Check if either partition is too small; if so, stop further division
        if len(subgraph_1.nodes) < min_cluster_size or len(subgraph_2.nodes) < min_cluster_size:
            result.append(g)
        else:
            clusters.append(subgraph_1)
            clusters.append(subgraph_2)

    result.extend(clusters)
    return result

def part_graph(graph):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype='cut', ufactor=250)

    # Group nodes by their partition
    partitions = [[], []]
    for node, part in zip(graph.nodes, parts):
        partitions[part].append(node)

    # Create subgraphs for each partition
    subgraph_1 = graph.subgraph(partitions[0]).copy()
    subgraph_2 = graph.subgraph(partitions[1]).copy()

    return subgraph_1, subgraph_2


def get_cluster(graph, clusters):
    nodes = [n for n in graph.node if graph.node[n]['cluster'] in clusters]
    return nodes


def connecting_edges(partitons, graph):
    """Find edges connecting two clusters."""
    return [(u, v) for u, v in graph.edges() if u in partitons[0] and v in partitons[1]]


def get_weights(graph, edges):
    """Efficiently fetch weights of given edges."""
    return [graph[u][v]['weight'] for u, v in edges]


def min_cut_bisector(graph):
    graph = graph.copy()
    partitions = part_graph(graph)
    return connecting_edges(partitions, graph)


def bisection_weights(graph, cluster):
    """Calculate the bisection weights of a graph."""
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights