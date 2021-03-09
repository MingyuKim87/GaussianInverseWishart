import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import *

def make_df(data, classname='original'):    
    if torch.is_tensor(data):
        data = data.numpy()
        
    property_list = np.array([[classname] * data.shape[0]]).T

    # concatenate
    data = np.concatenate((data, property_list), axis=1)
    df = pd.DataFrame(data, columns=list(range(data.shape[-1])))

    return df

def make_pairplot(df, classname, filepath):
    plot = sns.pairplot(df, hue=classname)
    plot.savefig(filepath)

    return plot

if __name__ == "__main__":
    # file path
    global_file_path = "./Data/GlobalInformation.csv"
    outlier_file_path = "./Data/Outliers.csv"

    # import data
    x,y = load_global_stat(global_file_path, outlier_file_path)

    # df
    y_df = make_df(y)

    # snsplot
    sns_plot = sns.pairplot(y_df, plot_kws={'alpha':0.1})
    sns_plot.savefig("scatter_plot.png")

    



    