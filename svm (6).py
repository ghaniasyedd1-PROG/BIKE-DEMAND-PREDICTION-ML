"""Support Vector Regression (SVR) estimator."""
from sklearn.svm import SVR
def build_model():
    return SVR(kernel="rbf")
