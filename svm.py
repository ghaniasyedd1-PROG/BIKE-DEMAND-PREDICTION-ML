"""Support Vector Classification (SVC) estimator."""
from sklearn.svm import SVC
def build_model():
    return SVC(kernel="rbf", probability=True, random_state=42)
