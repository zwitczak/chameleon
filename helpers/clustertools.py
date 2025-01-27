import numpy as np
from helpers.graphtools import get_weights, connecting_edges, bisection_weights

def internal_interconnectivity(graph, cluster):
    """Calculate the internal interconnectivity of a cluster."""
    return np.sum(bisection_weights(graph, cluster))



def relative_interconnectivity(graph, cluster1, cluster2):
    """Calculate the relative interconnectivity between two clusters."""
    edges = connecting_edges((cluster1, cluster2), graph)

    EC = np.sum(get_weights(graph, edges))

    ECci, ECcj = internal_interconnectivity(
        graph, cluster1), internal_interconnectivity(graph, cluster2)
    
    return EC / ((ECci + ECcj) / 2.0) if ECci + ECcj > 0 else 0.0

def calculate_dynamic_min_merge_score(graph, clusters, alpha):
    merge_scores = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            score = merge_score(graph, clusters[i], clusters[j], alpha)
            merge_scores.append(score)
    return np.percentile(merge_scores, 25)  # 25th percentile

def internal_closeness(graph, cluster):
        """Calculate the internal closeness of a cluster."""
        cluster = graph.subgraph(cluster)
        edges = cluster.edges()
        weights = get_weights(cluster, edges)
        return np.sum(weights)



def relative_closeness(graph, cluster1, cluster2):
    """Calculate the relative closeness between two clusters."""
    edges = connecting_edges((cluster1, cluster2), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(
        graph, cluster1), internal_closeness(graph, cluster2)
    
    SECci, SECcj = np.mean(bisection_weights(graph, cluster1)), np.mean(bisection_weights(graph, cluster2))

    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(graph, cluster1, cluster2, alpha=0.5):
    """Calculate the merge score between two clusters."""
    if len(cluster1.nodes) == 0 or len(cluster2.nodes) == 0:
        return float('-inf')  # Return a very low score if either cluster is empty

    ri = relative_interconnectivity(graph, cluster1, cluster2)
    rc = relative_closeness(graph, cluster1, cluster2)
    return ri * np.power(rc, alpha)