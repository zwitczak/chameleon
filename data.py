'2d-4c.arff'
from scipy.io import arff
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt
from clustering import Chameleon

def load_data(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    return df

def plot_data(df, title=None):
    try:
        plt.scatter(df.iloc[:, 0], df.iloc[:,1], c=df.iloc[:, 2].replace({b'noise': 'white'}))
        plt.xlabel('x1')
        plt.ylabel('x2')

        if title:
            plt.title(title)
        plt.show()
    except:
        pass

c = 0
for file in reversed(listdir('data/artificial')):
    print(file)
    df = load_data(f'data/artificial/{file}')
    plot_data(df, title=file)
    c += 1
    a = Chameleon(df, min_size=0.05, k_neighbors=4)
    a.init_function_product_scheme(0.5)
    if c == 1:
        break



