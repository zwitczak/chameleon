import os
from chameleon import Chameleon
from helpers.filehandler import FileHandler
from helpers.plothandler import ClusterPlotHandler

if __name__ == '__main__':
    DATA_DIR = 'data/artificial'
    OUTPUT_DIR = 'results'

    # ensure data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directory {DATA_DIR} not found.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # initialize chameleon algorithm
    chameleon = Chameleon(min_cluster_size=0.015, k_neighbors=15, alpha=1.5)
    for alfa in [1.5, 2.0, 2.5]:
        for k in [5, 10, 15]:
            chameleon = Chameleon(min_cluster_size=0.015, k_neighbors=k, alpha=alfa)
            for c, file in enumerate(os.listdir(DATA_DIR)):
                df, class_flag = FileHandler.load_arff_data(os.path.join(DATA_DIR, file))
                if df is None:
                    continue
                
                # exclude class column from dataset
                if class_flag:
                    dataset = df.iloc[:, :-1]

                final_clusters = chameleon.fit(dataset=dataset, verbose=False)
                part_clusters = chameleon.clusters


                final_clusters_df = FileHandler.transform_graph_to_df(final_clusters, df)
                partial_clusters_df = FileHandler.transform_graph_to_df(part_clusters, df)
                
                clusters_dict = {
                    'final_clusters': final_clusters_df,
                    'partial_clusters': partial_clusters_df,
                    'correct_clusters': df
                }

                ClusterPlotHandler.draw_2d_many_clusters(output_dir=os.path.join(OUTPUT_DIR, f"{file}_comparison_k{k}_a{alfa}.png"), **clusters_dict)

                # if c == 20:
                #     break