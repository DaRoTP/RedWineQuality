import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def decision_tree(df):
    fn = ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides']
    cn = ['low', 'mid', 'high']
    x = df.loc[:, fn]
    y = df['typed_quality']
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, y, train_size=0.7)

    dtc = DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)

    accuracy_score = dtc.score(test_inputs, test_classes) * 100
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score) + "%\n")

    confusion_matrix_res = confusion_matrix(test_classes, dtc.predict(test_inputs))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)
    # print(classification_report(test_classes, dtc.predict(test_inputs), labels=cn))


def bayes_algorithm(df):
    fn = ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides']
    cn = ['low', 'mid', 'high']
    x = df.loc[:, fn]
    y = df['typed_quality']
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, y, train_size=0.7)

    nb = GaussianNB()
    nb.fit(train_inputs, train_classes)

    accuracy_score = nb.score(test_inputs, test_classes) * 100
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score) + "%\n")

    confusion_matrix_res = confusion_matrix(test_classes, nb.predict(test_inputs))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)


def KNN_clasification(df):
    fn = ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides']
    cn = ['bad', 'good', 'high']
    # cn = ['low', 'mid', 'high']
    x = df.loc[:, fn]
    y = df['typed_quality']

    sc = StandardScaler()
    x = sc.fit_transform(x)

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, y, train_size=0.7)
    # print("====")
    # print(train_inputs)
    # print("====")
    # print(test_inputs)
    # print("====")
    # print(train_classes)
    # print("====")
    # print(test_classes)

    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors means k
    knn.fit(train_inputs, train_classes)

    accuracy_score = knn.score(test_inputs, test_classes) * 100
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score) + "%\n")

    confusion_matrix_res = confusion_matrix(test_classes, knn.predict(test_inputs))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)


def logistic(df):
    fn = ['alcohol', 'sulphates', 'citric_acid', 'volatile_acidity', 'chlorides']
    cn = ['low', 'mid', 'high']
    x = df.loc[:, fn]
    y = df['typed_quality']

    sc = StandardScaler()
    x = sc.fit_transform(x)

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x, y, train_size=0.7)

    lr = LogisticRegression()
    lr.fit(train_inputs, train_classes)

    accuracy_score = lr.score(test_inputs, test_classes) * 100
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score) + "%\n")

    confusion_matrix_res = confusion_matrix(test_classes, lr.predict(test_inputs))
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)
