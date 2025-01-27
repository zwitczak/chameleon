from matplotlib import pyplot as plt

COLORS = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#000000",  # Black
    "#FF4500",  # Orange Red
    "#7FFF00",  # Chartreuse
    "#1E90FF",  # Dodger Blue
    "#FFD700",  # Gold
    "#8A2BE2",  # Blue Violet
    "#40E0D0",  # Turquoise
    "#2F4F4F",  # Dark Slate Gray
    "#F5F5F5",  # White Smoke
    "#DC143C",  # Crimson
    "#32CD32",  # Lime Green
    "#4682B4",  # Steel Blue
    "#FFA500",  # Orange
    "#DA70D6",  # Orchid
    "#00CED1",  # Dark Turquoise
    "#808080",  # Gray
    "#F0E68C",  # Khaki
    "#B22222",  # Fire Brick
    "#ADFF2F",  # Green Yellow
    "#5F9EA0",  # Cadet Blue
    "#FF1493",  # Deep Pink
    "#9932CC",  # Dark Orchid
    "#20B2AA",  # Light Sea Green
    "#696969",  # Dim Gray
    "#FFFFF0"   # Ivory
]

class ClusterPlotHandler:

    @staticmethod
    def draw_2d_cluster(output_dir: str = '.', df=None, title=None):
        plt.figure(figsize=(10, 8))
        n = df['class'].nunique()
        # Plot each cluster with a different color
        for cluster_idx in df['class'].unique():
            cluster_data = df[df['class'] == cluster_idx]
            if cluster_idx == n - 1:
                plt.scatter(cluster_data['x'], cluster_data['y'], color=COLORS[cluster_idx % len(COLORS)], label=f'Cluster {cluster_idx}', alpha=0.7)
            else:
                plt.scatter(cluster_data['x'], cluster_data['y'], color=COLORS[cluster_idx % len(COLORS)], label=f'Cluster {cluster_idx}', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Clusters Visualization\nNumber of clusters: {df["class"].nunique()}')
        
        # Save plot to file
        plt.savefig(output_dir)
        plt.close()

    @staticmethod
    def draw_2d_cluster_helper(class_i: int, class_j: int, output_dir: str = '.', df=None, title=None):
        plt.figure(figsize=(10, 8))
        n = df['class'].nunique()
        # Plot each cluster with a different color
        for cluster_idx in df['class'].unique():
            cluster_data = df[df['class'] == cluster_idx]
            if cluster_idx == class_i:
                plt.scatter(cluster_data['x'], cluster_data['y'], color='green', label=f'Cluster {cluster_idx}', alpha=0.7)
            elif cluster_idx == class_j:
                plt.scatter(cluster_data['x'], cluster_data['y'], color='blue', label=f'Cluster {cluster_idx}', alpha=0.7)
            else:
                plt.scatter(cluster_data['x'], cluster_data['y'], color='red', label=f'Cluster {cluster_idx}', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Clusters Visualization\nNumber of clusters: {df["class"].nunique()}')
        
        # Save plot to file
        plt.savefig(output_dir)
        plt.close()

    


    @staticmethod
    def draw_2d_many_clusters(output_dir: str = '.', **graphs):
        
        # Create a scatter plot
        number_of_figures = len(graphs)

        fig, axes = plt.subplots(1, number_of_figures, figsize=(number_of_figures*5, 5))

        for idx, el in enumerate(graphs.items()):
            k , df = el
            print(k, df)
            n_clusters = df['class'].nunique()
            axes[idx].set_title(f"{k}: Number of clusters: {n_clusters}")
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')

            for idx2, cluster in enumerate(df['class'].unique()):
                cluster_points = df[df['class'] == cluster]
                axes[idx].scatter(
                    cluster_points['x'], 
                    cluster_points['y'], 
                    color=COLORS[idx2 % len(COLORS)], 
                    alpha=0.7
                )


            if n_clusters <= 5:
                axes[idx].legend()

        plt.savefig(output_dir)
        plt.close()
        