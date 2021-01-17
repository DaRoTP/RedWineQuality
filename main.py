import pandas as pd
from data_analysis import *
from classificators import *
import matplotlib.pyplot as plt

def analyze_data(df):
    # how many null values does the column has
    print(df.info())
    print(df.isnull().sum())

    typed_quality_chart(df)

    column_data_analysis_params = [
        {'fixed_acidity': 1.03},
        {'volatile_acidity': 0.16},
        {'citric_acid': 0.10},
        {'residual_sugar': 1.46},
        {'chlorides': 0.06},
        {'free_sulfur_dioxide': 9.1},
        {'total_sulfur_dioxide': 28.3},
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

    df['typed_quality'] = pd.cut(df['quality'], bins=[0, 6, 10], labels=["bad", "good"])

    df.columns = df.columns.str.replace(' ', '_')
    # analyze_data(df)

    test = Data_classification(df=df, feature_names=['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides'])
    test.train_test_split()

    classificator_labels = []
    classificator_accuracy = []

    classificator_labels.append("Drzewo decyzyjne")
    classificator_accuracy.append(test.decision_tree() * 100)

    classificator_labels.append("Naive Bayes")
    classificator_accuracy.append(test.bayes_algorithm() * 100)

    classificator_labels.append("k najbliższych sąsiadów dla k=2")
    classificator_accuracy.append(test.KNN_classification() * 100)

    classificator_labels.append("Logistic Regression")
    classificator_accuracy.append(test.logistic_regression() * 100)

    classificator_labels.append("Sieći neuronowe MLP")
    classificator_accuracy.append(test.MLPClassifier() * 100)

    classificator_labels.append("SVM")
    classificator_accuracy.append(test.svn_algorithm() * 100)

    classificator_labels.append("Random Forest")
    classificator_accuracy.append(test.random_forest() * 100)

    classificator_labels.append("Linear Discriminant Analysis")
    classificator_accuracy.append(test.linear_discriminant() * 100)

    plt.title("Porównanie klasyfikatorów")
    y_pos = np.arange(len(classificator_labels))
    x_pos = np.arange(0, 100, 10)

    fig, ax = plt.subplots()

    ax.barh(y_pos, classificator_accuracy, align='center', alpha=0.5)
    plt.yticks(y_pos, classificator_labels)
    plt.xticks(x_pos)
    plt.xlabel('Accuracy (%)')



    plt.show()