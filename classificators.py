import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def get_data_accuracy(accuracy_score):
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score * 100) + "%\n")


def get_confusion_matrix(test_y, predict):
    confusion_matrix_res = confusion_matrix(test_y, predict)
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)


class Data_classification:
    def __init__(self, df, feature_names=[]):
        self.df = df
        if len(feature_names) < 1:
            self.feature_names = df.columns[:11].tolist()
        else:
            self.feature_names = feature_names

        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

    def train_test_split(self, train_size=0.7):
        x = self.df.loc[:, self.feature_names]
        y = self.df['typed_quality']
        sc = StandardScaler()
        x = sc.fit_transform(x)

        (train_x, test_x, train_y, test_y) = train_test_split(x, y, train_size=train_size)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

    def decision_tree(self):
        dtc = DecisionTreeClassifier()
        dtc.fit(self.train_x, self.train_y)

        print('_===== DECISION TREE =====_')
        get_data_accuracy(dtc.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, dtc.predict(self.test_x))

        # print(classification_report(self.test_y, dtc.predict(self.test_x), labels=['low' 'mid', 'high']))

    def bayes_algorithm(self):
        nb = GaussianNB()
        nb.fit(self.train_x, self.train_y)

        print('_===== BAYES ALGO =====_')
        get_data_accuracy(nb.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, nb.predict(self.test_x))

    def KNN_classification(self, n_neighbors=2):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.train_x, self.train_y)

        print('_===== KNN_classification =====_')
        get_data_accuracy(knn.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, knn.predict(self.test_x))

    def logistic_regression(self):
        lr = LogisticRegression()
        lr.fit(self.train_x, self.train_y)

        print('_===== logistic_regression =====_')
        get_data_accuracy(lr.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, lr.predict(self.test_x))

    def MLPClassifier(self):
        clf = MLPClassifier(random_state=1, max_iter=300)
        clf.fit(self.train_x, self.train_y)

        print('_===== MLPClassifier =====_')
        get_data_accuracy(clf.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, clf.predict(self.test_x))

