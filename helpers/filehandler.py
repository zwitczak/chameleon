import os
import pandas as pd
import numpy as np
from scipy.io import arff
from typing import List
import networkx as nx

GraphList = List[nx.Graph]

class FileHandler:
    @staticmethod
    def ensure_dir_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    
    @staticmethod
    def load_arff_data(path: str) -> pd.DataFrame:
        if not path.endswith('.arff'):
            raise ValueError("File must be in ARFF format.")
        if not os.path.exists(path):
            raise FileNotFoundError("File not found.")
        
        data, meta = arff.loadarff(path)
        class_column_flag = 'class' in [n.lower() for n in meta.names()]
    
        df = pd.DataFrame(data)
        
        if class_column_flag:
            result = FileHandler.format_class_dataframe(df)
        else:
            result = FileHandler.format_dataframe(df)
        
        return result, class_column_flag
    
    @staticmethod
    def format_class_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] == 3:
            df.rename(columns={df.columns[0]: 'x', df.columns[1]: 'y', df.columns[2]: 'class'}, inplace=True)
        elif df.shape[1] == 4:
            df.rename(columns={df.columns[0]: 'x', df.columns[1]: 'y', df.columns[2]: 'z', df.columns[3]: 'class'}, inplace=True)
        else:
            raise ValueError("Data frame must have 2 or 3 columns.")
        df['class'] = df['class'].apply(lambda x: int(x.decode('utf-8')) if x != b'noise' else -1)
        return df
        
    
    @staticmethod
    def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] == 2:
            df.rename(columns={df.columns[0]: 'x', df.columns[1]: 'y'}, inplace=True)
        elif df.shape[1] == 3:
            df.rename(columns={df.columns[0]: 'x', df.columns[1]: 'y', df.columns[2]: 'z'}, inplace=True)
        else:
            raise ValueError("Data frame must have 2 or 3 columns.")
        return df
    
    @staticmethod
    def transform_graph_to_df(clusters: GraphList, df: pd.DataFrame) -> pd.DataFrame:
        data = []
        for idx, G in enumerate(clusters):
            for node in G.nodes:
                pos_x, pos_y = df.loc[node, ['x', 'y']]
                data.append((node, idx, pos_x, pos_y))
        
        transformed_df = pd.DataFrame(data, columns=['idx', 'x', 'y', 'class'])
        transformed_df.set_index('idx', inplace=True)
        return transformed_df

if __name__ == "__main__":
    DATA_DIR = 'data/artificial'
    OUTPUT_DIR = 'results'

    # ensure data directory exists
    FileHandler.ensure_dir_exists(DATA_DIR)
    FileHandler.ensure_dir_exists(OUTPUT_DIR)

    for c, file in enumerate(os.listdir(DATA_DIR)):
        df = FileHandler.load_arff_data(os.path.join(DATA_DIR, file))
        print(df.head())
        if c == 2:
            break