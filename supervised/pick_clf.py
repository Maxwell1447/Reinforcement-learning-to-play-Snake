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
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbor, n_jobs=-1)
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
    clf = NuSVC(kernel='rbf', random_state=0, shrinking=True, gamma='auto', verbose=True, tol=1e-5)
    return clf


def MLP(nb_hidden_layer):
    """
    Wraps initialization of a Multi-layer Perceptron Classifier.
    """
    clf = MLPClassifier(hidden_layer_sizes=(nb_hidden_layer, ), activation='relu', solver='adam', max_iter=1000, n_iter_no_change = 10, verbose=True, tol=1e-5)
    return clf

def RandomForest(nb_estimator):
    """
    Wraps initialization of a Random Forest Classifier
    """
    clf = RandomForestClassifier(n_estimators=nb_estimator, criterion='gini', max_depth=None, max_features='auto', n_jobs=-1, verbose=1)
    return clf
    


def pick_clf(clf_name, nb_parameter):
    if clf_name == 'logreg':
        clf = logreg()
    elif clf_name == 'kNN':
        clf = kNN(nb_parameter)
    elif clf_name == 'SVM':
        clf = SVM()
    elif clf_name == 'Nusvm':
        clf = NuSVM()
    elif clf_name == 'MLP':
        clf = MLP(nb_parameter)
    elif clf_name == 'Forest':
        clf = RandomForest(nb_parameter)
    elif clf_name == 'multiclass':
        clf = OneVsRestClassifier(SVC(gamma='auto', verbose=True), n_jobs=-1)
    else:
        raise ValueError("Wrong classifier name")
    return clf
