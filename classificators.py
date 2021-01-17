import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_data_accuracy(accuracy_score):
    print('\nTRAINED DATA ACCURACY {0:.2f}'.format(accuracy_score * 100) + "%\n")


def get_confusion_matrix(test_y, predict, classification_title):
    confusion_matrix_res = confusion_matrix(test_y, predict)
    print("CONFUSION MATRIX:")
    print(confusion_matrix_res)
    x_axis_labels = ['Bad', 'Good']
    y_axis_labels = ['Bad', 'Good']
    sns.heatmap(confusion_matrix_res, annot=True, fmt='', cmap='Blues', xticklabels=x_axis_labels,
                yticklabels=y_axis_labels)
    plt.title('Macież błędu ' + classification_title)
    plt.xlabel('Przewidziany')
    plt.ylabel('Prawdziwy')
    plt.show()


def get_feature_importance(feature_importance, feature_labels, classificator_name):
    print(feature_importance)
    print(feature_labels)
    plt.title("Feature importance "+classificator_name)
    y_pos = np.arange(len(feature_labels))

    plt.barh(y_pos, feature_importance, align='center', alpha=0.5)
    plt.yticks(y_pos, feature_labels)
    plt.xlabel('Importance')

    plt.show()


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

        (train_x, test_x, train_y, test_y) = train_test_split(x, y, train_size=train_size, test_size=1 - train_size,
                                                              random_state=1)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        print(self.df.describe())

    def decision_tree(self):
        dtc = DecisionTreeClassifier()
        dtc.fit(self.train_x, self.train_y)

        title = "DECISION TREE"
        print('_===== '+title+' =====_')
        get_feature_importance(feature_importance=dtc.feature_importances_, feature_labels=self.feature_names,
                               classificator_name=title)
        get_data_accuracy(dtc.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, dtc.predict(self.test_x), 'DECISION TREE')
        # print(classification_report(self.test_y, dtc.predict(self.test_x), labels=['low', 'mid', 'high']))

    def bayes_algorithm(self):
        nb = GaussianNB()
        nb.fit(self.train_x, self.train_y)

        title = "BAYES ALGO"
        print('_===== '+title+' =====_')
        get_feature_importance(feature_importance=nb., feature_labels=self.feature_names,
                               classificator_name=title)
        get_data_accuracy(nb.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, nb.predict(self.test_x), 'bayes_algorithm')

    def KNN_classification(self, n_neighbors=2):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.train_x, self.train_y)

        title = "KNN_classification"
        print('_===== '+title+' =====_')
        # get_feature_importance(feature_importance=knn.feature_importances_, feature_labels=self.feature_names,
        #                        classificator_name=title)
        get_data_accuracy(knn.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, knn.predict(self.test_x), 'KNN_classification')

    def logistic_regression(self):
        lr = LogisticRegression()
        lr.fit(self.train_x, self.train_y)

        title = "logistic_regression"
        print('_===== '+title+' =====_')
        # get_feature_importance(feature_importance=lr.feature_importances_, feature_labels=self.feature_names,
        #                        classificator_name=title)
        get_data_accuracy(lr.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, lr.predict(self.test_x), 'logistic_regression')

    def MLPClassifier(self):
        clf = MLPClassifier(random_state=1, max_iter=300)
        clf.fit(self.train_x, self.train_y)

        title = "MLPClassifier"
        print('_===== '+title+' =====_')
        # get_feature_importance(feature_importance=clf.feature_importances_, feature_labels=self.feature_names,
        #                        classificator_name=title)
        get_data_accuracy(clf.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, clf.predict(self.test_x), 'MLPClassifier')

    def svn_algorithm(self):
        svm = SVC(random_state=1)
        svm.fit(self.train_x, self.train_y)

        title = "SVC"
        print('_===== ' + title + ' =====_')
        # get_feature_importance(feature_importance=svm.feature_importances_, feature_labels=self.feature_names,
        #                        classificator_name=title)
        get_data_accuracy(svm.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, svm.predict(self.test_x), 'svn_algorithm')

    def random_forest(self):
        rf = RandomForestClassifier(n_estimators=1000, random_state=1)
        rf.fit(self.train_x, self.train_y)

        title = "random_forest"
        print('_===== ' + title + ' =====_')
        get_feature_importance(feature_importance=rf.feature_importances_, feature_labels=self.feature_names,
                               classificator_name=title)
        get_data_accuracy(rf.score(self.test_x, self.test_y))
        get_confusion_matrix(self.test_y, rf.predict(self.test_x), 'svn_algorithm')
