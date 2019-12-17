from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.svm import LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier


def fit_logreg():
    """
    Wraps initialization of Logistic regression
    """
    logreg = LogisticRegression(C=1e20, solver='saga', dual=False, n_jobs=-1, verbose=True, max_iter=1400)  #
    return logreg


def fit_kNN(n_neighbor):
    """
    We create an instance of Neighbours Classifier
    """
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbor)
    return clf


def fit_SVM():
    """
    Wraps initialization of Logistic regression
    """
    clf = LinearSVC(random_state=0, dual=False, tol=1e-5, verbose=True, max_iter=100)
    return clf


def fit_NuSVM():
    """
    Wraps initialization of Logistic regression
    """
    clf = NuSVC(kernel='rbf', random_state=0, gamma='auto', verbose=True, tol=1e-5)
    return clf


def fit_MLP():
    """
    Wraps initialization of Logistic regression
    """
    clf = MLPClassifier(activation='relu', solver='adam', max_iter=1000, verbose=True, tol=1e-5)
    return clf


def pick_clf(clf_name, n):
    if clf_name == 'logreg':
        clf = fit_logreg()
    elif clf_name == 'kNN':
        clf = fit_kNN(n)
    elif clf_name == 'SVM':
        clf = fit_SVM()
    elif clf_name == 'Nusvm':
        clf = fit_NuSVM()
    elif clf_name == 'MLP':
        clf = fit_MLP()
    else:
        raise ValueError("Wrong classifier name")
    return clf
