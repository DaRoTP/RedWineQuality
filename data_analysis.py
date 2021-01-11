import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv("winequality-red.csv")


def get_column_value_count_info(column_name, x_step):
    selected_column = df[column_name]
    column_min = selected_column.min()
    column_max = selected_column.max()
    print(f"--- {column_name} ---")
    print(f"min: {column_min}")
    print(f"max - {column_max}")
    print(f"avg - {selected_column.mean()}")
    interval_list = np.arange(column_min, column_max, x_step)
    interval_list = np.append(interval_list, column_max)
    plt.hist(selected_column.to_numpy(), bins=interval_list)
    plt.xticks(interval_list)
    plt.title(f"rozkład wartości - {column_name}")
    plt.xlabel(column_name)
    plt.show()


def param_dependency_on_quality(column_name):
    plt.figure(figsize=(8, 5))
    plt.title(f"zależność jakości wina od \"{column_name}\"")
    sns.barplot(x=df['quality'], y=df[column_name], palette="GnBu_d")
    plt.show()
    plt.show()


def param_dependency_heatmap():
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, linewidth=0.5, center=0, cmap='coolwarm')
    plt.show();
