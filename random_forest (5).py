"""Random Forest regression estimator."""
from sklearn.ensemble import RandomForestRegressor
def build_model():
    return RandomForestRegressor(random_state=42, n_jobs=-1)
