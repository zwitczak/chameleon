from scipy.io import arff
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt
import matplotlib as mlp
from chameleon import Chameleon
import numpy as np
from pylab import *

def load_data(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    return df

def transform_graphs_to_dataframe(graphs, df):
    data = []
    for cluster_idx, G in enumerate(graphs):
        for node in G.nodes:
            pos_x, pos_y = df.loc[node, ['a0', 'a1']]
            data.append((node, cluster_idx, pos_x, pos_y))
    
    transformed_df = pd.DataFrame(data, columns=['node_idx', 'cluster_idx', 'pos_x', 'pos_y'])
    return transformed_df


def plot_clusters(df, filename='clusters.png'):
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['pos_x'], df['pos_y'], c=df['cluster_idx'], cmap='viridis', s=50, alpha=0.7)
    
    # Add a color bar
    plt.colorbar(scatter, label='Cluster Index')
    
    # Add labels and title
    plt.xlabel('pos_x')
    plt.ylabel('pos_y')
    plt.title('Clusters Visualization')
    
    # Save plot to file
    plt.savefig(filename)
    plt.close()
if __name__ == '__main__':

    c = 0
    for file in reversed(listdir('data/artificial')):
        file = 'spiral.arff'  # You are overwriting this line, making the loop unnecessary
        df = load_data(f'data/artificial/{file}')
        break

    chameleon = Chameleon(dataset=df, min_cluster_size=0.005, k_neighbors=2, alpha=0.5)
    clusters = chameleon.fit(verbose=True)
    n = len(clusters)

    transformed_df = transform_graphs_to_dataframe(clusters, df)
    print(transformed_df)
    plot_clusters(transformed_df)
    # print(f"Number of final clusters: {n}")

    # data = transform_graphs_to_data(chameleon.final_clusters, clusters)