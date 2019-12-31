from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def logreg():
    """
    Wraps initialization of a Logistic regression Classifier
    """
    logreg = LogisticRegression(C=1e20, solver='saga', dual=False, n_jobs=-1, verbose=True, max_iter=1400)
    return logreg


def kNN(n_neighbor):
    """
    We create an instance of a Nearest Neighbours Classifier
    """
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbor)
    return clf


def SVM():
    """
    Wraps initialization of a Linear Support Vector Classifier
    """
    clf = LinearSVC(random_state=0, dual=False, tol=1e-5, verbose=True, max_iter=100)
    return clf


def NuSVM():
    """
    Wraps initialization of a Nu-Support Vector Classifier
    """
    clf = NuSVC(kernel='rbf', random_state=0, gamma='auto', verbose=True, tol=1e-5)
    return clf


def MLP():
    """
    Wraps initialization of a Multi-layer Perceptron Classifier.
    """
    clf = MLPClassifier(hidden_layer_sizes=(70, ), activation='relu', solver='adam', max_iter=1000, n_iter_no_change = 10, verbose=True, tol=1e-5)
    return clf

def RandomForest():
    """
    Wraps initialization of a Random Forest Classifier
    """
    clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, max_features='auto', n_jobs=-1, verbose=1)
    return clf
    


def pick_clf(clf_name, n):
    if clf_name == 'logreg':
        clf = logreg()
    elif clf_name == 'kNN':
        clf = kNN(n)
    elif clf_name == 'SVM':
        clf = SVM()
    elif clf_name == 'Nusvm':
        clf = NuSVM()
    elif clf_name == 'MLP':
        clf = MLP()
    elif clf_name == 'Forest':
        clf = RandomForest()
    elif clf_name == 'multiclass':
        clf = OneVsRestClassifier(SVC(gamma='auto', verbose=True))
    else:
        raise ValueError("Wrong classifier name")
    return clf
