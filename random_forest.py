"""Random Forest classification estimator."""
from sklearn.ensemble import RandomForestClassifier
def build_model():
    return RandomForestClassifier(random_state=42, n_jobs=-1)
