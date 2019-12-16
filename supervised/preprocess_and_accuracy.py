import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def prepare_data(ds):
    X_cols = ds.copy()
    X = X_cols.values
    X = X.reshape(len(X_cols), -1)

    # We add the dummy x_0 and features’ high-order
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)

    return X


def predict_and_test(model, X_test, y_test):
    """
    Predicts using a model received as input and then evaluates the accuracy of the predicted data.
    As inputs it receives the model, an input dataset X_test and the corresponding targets (ground thruth) y_test
    It returs the classification accuracy.
    """
    y_hat = np.array(model.predict(X_test))
    correct = np.sum(y_hat == y_test)
    samples = y_test.size
    return correct / samples
