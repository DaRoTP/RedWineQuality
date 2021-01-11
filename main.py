import pandas as pd
from data_analysis import *

if __name__ == '__main__':
    df = pd.read_csv("winequality-red.csv")

    # how many null values does the column has
    # print(df.isnull().sum())

    # are all column values are numeric
    print(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))

    column_data_analysis_params = [
        {'fixed acidity': 1.03},
        {'volatile acidity': 0.16},
        {'citric acid': 0.10},
        {'residual sugar': 1.46},
        {'chlorides': 0.06},
        {'free sulfur dioxide': 9.1},
        {'total sulfur dioxide': 28.3},
        {'density': 0.0025},
        {'pH': 0.11},
        {'sulphates': 0.17},
        {'alcohol': 0.65},
        {'quality': 1},
    ]

    # param_dependency_heatmap()

    # for param in column_data_analysis_params:
    #     column_name = list(param.keys())[0]
    #     get_column_value_count_info(column_name, param.get(column_name))
    #     param_dependency_on_quality(column_name)
