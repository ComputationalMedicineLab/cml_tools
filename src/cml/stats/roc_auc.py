"""
A location for ROC-AUC related utilities.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def scoring(estimator, X, y):
    """Used for the `scoring` parameter of many sklearn cross-val tools"""
    preds = estimator.predict_proba(X)
    return roc_auc_score(y, preds[:, 1])
