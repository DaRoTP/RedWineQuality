import pandas as pd
from data_analysis import *
from classificators import *


def analyze_data(df):
    # how many null values does the column has
    print(df.info())
    print(df.isnull().sum())

    typed_quality_chart(df)

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

    param_dependency_heatmap(df)

    for param in column_data_analysis_params:
        column_name = list(param.keys())[0]
        get_column_value_count_info(df, column_name, param.get(column_name))
        param_dependency_on_quality(df, column_name)


if __name__ == '__main__':
    df = pd.read_csv("winequality-red.csv")

    df['typed_quality'] = pd.cut(df['quality'], bins=[0, 6.5, 10], labels=["bad", "good"])

    df.columns = df.columns.str.replace(' ', '_')

    test = Data_classification(df=df, feature_names=['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides'])
    test.train_test_split()

    test.decision_tree()
    test.bayes_algorithm()
    test.KNN_classification()
    test.logistic_regression()
    test.MLPClassifier()
    test.svn_algorithm()
    test.random_forest()