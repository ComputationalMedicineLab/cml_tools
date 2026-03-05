"""
A location for sklearn related utilities.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate


def scoring(estimator, X, y):
    """Used for the `scoring` parameter of many sklearn cross-val tools"""
    preds = estimator.predict_proba(X)
    return roc_auc_score(y, preds[:, 1])


def cv_rf_refit(params, X, y, cv=6):
    # `cross_validate` won't give predictions, and `cross_val_predict` won't
    # give the estimators. So this is a utility function to get both.
    params.setdefault('n_estimators', 100)
    params.setdefault('class_weight', 'balanced')
    params.setdefault('n_jobs', -1)
    params.setdefault('random_state', 42)

    results = cross_validate(RandomForestClassifier(**params),
                             X, y, scoring=scoring, cv=cv,
                             return_indices=True,
                             return_estimator=True)

    # Now get the cross-val predictions
    results['predictions'] = np.empty(len(y))
    for est, idx in zip(results['estimator'], results['indices']['test']):
        results['predictions'][idx] = est.predict_proba(X[idx])[:, 1]
    return results


def auc_bootstrap(y_true, y_pred, n=100, *, max_sample_attempts=1_000):
    """Generate a distribution of AUC's using bootstrap samples from y_pred"""
    aucs = []
    m = len(y_true)
    for i in range(n):
        for _ in range(max_sample_attempts):
            idx = np.random.choice(m, size=m, replace=True)
            if np.any(y_true[idx]):
                break
        else:
            raise RuntimeError(f'{max_sample_attempts=} reached')
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.array(aucs)
