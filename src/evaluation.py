import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, brier_score_loss, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression

def pr_auc_score(y_true, p):
    precision, recall, _ = precision_recall_curve(y_true, p)
    return auc(recall, precision)

def calibration_slope(y_true, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=False).fit(logit, y_true)
    return lr.coef_[0][0]

def get_calibrated_probs(model, X_set, y_train_set, train_raw_probs):
    eps = 1e-6
    train_raw_probs = np.clip(train_raw_probs, eps, 1-eps)
    logit_train = np.log(train_raw_probs / (1 - train_raw_probs)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logit_train, y_train_set)
    raw_p = np.clip(model.predict_proba(X_set)[:, 1], eps, 1-eps)
    logit_set = np.log(raw_p / (1 - raw_p)).reshape(-1, 1)
    return calibrator.predict_proba(logit_set)[:, 1]

def evaluate_metrics(y_true, p, w, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, sample_weight=w).ravel()
    return {
        "Acc": accuracy_score(y_true, pred, sample_weight=w),
        "AUC": roc_auc_score(y_true, p, sample_weight=w),
        "PR_AUC": pr_auc_score(y_true, p),
        "Kappa": cohen_kappa_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "Brier": brier_score_loss(y_true, p, sample_weight=w),
        "Slope": calibration_slope(y_true.values, p),
        "Sens": tp / (tp + fn + 1e-9),
        "Spec": tn / (tn + fp + 1e-9),
        "Thresh": t
    }

def get_bootstrap_results(y, p, w, t, B=1000):
    rng = np.random.RandomState(42)
    n, boot_data = len(y), []
    for _ in range(B):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y.iloc[idx])) < 2: continue
        y_b, p_b, w_b = y.iloc[idx], p[idx], w.iloc[idx]
        m = evaluate_metrics(y_b, p_b, w_b, t)
        boot_data.append(list(m.values()))
    boot_data = np.array(boot_data)
    means, lower, upper = np.mean(boot_data, axis=0), np.percentile(boot_data, 2.5, axis=0), np.percentile(boot_data, 97.5, axis=0)
    return [f"{m:.3f} ({l:.3f}-{u:.3f})" if i != len(means)-1 else f"{m:.3f}" for i, (m, l, u) in enumerate(zip(means, lower, upper))]

def optimize_threshold(val_prob, y_val, w_val):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_t, best_j = 0, -1
    for t in thresholds:
        preds = (val_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, sample_weight=w_val).ravel()
        j = (tp/(tp+fn+1e-9)) + (tn/(tn+fp+1e-9)) - 1
        if j > best_j: best_j, best_t = j, t
    return best_t