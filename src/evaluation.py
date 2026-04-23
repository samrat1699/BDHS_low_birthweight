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

def calibration_intercept(y_true, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True)
    lr.fit(logit, y_true)
    return lr.intercept_[0]

def weighted_cm(y_true, p, w, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, sample_weight=w).ravel()
    return tp, fp, fn, tn

def get_calibrated_probs(model, X_set, y_train_set, train_raw_probs):
    eps = 1e-6
    train_raw_probs = np.clip(train_raw_probs, eps, 1-eps)
    logit_train = np.log(train_raw_probs / (1 - train_raw_probs)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logit_train, y_train_set)
    raw_p = np.clip(model.predict_proba(X_set)[:, 1], eps, 1-eps)
    logit_set = np.log(raw_p / (1 - raw_p)).reshape(-1, 1)
    return calibrator.predict_proba(logit_set)[:, 1]

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(n)
    T2[J] = T + 1
    return T2

def fast_delong(preds, label_1_count):
    m = int(label_1_count)
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]

    tx = np.array([compute_midrank(pos[i]) for i in range(k)])
    ty = np.array([compute_midrank(neg[i]) for i in range(k)])

    aucs = (tx.sum(axis=1) / m - (m + 1) / 2) / n

    v01 = (tx - tx.mean(axis=1, keepdims=True)) / n
    v10 = (ty - ty.mean(axis=1, keepdims=True)) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    return aucs, sx + sy

def delong_test(y_true, pred1, pred2):
    y_true = np.array(y_true)
    order = np.argsort(-pred1)
    y_true = y_true[order]
    preds = np.vstack((pred1, pred2))[:, order]
    label_1_count = np.sum(y_true)

    aucs, cov = fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = cov[0,0] + cov[1,1] - 2 * cov[0,1]

    z = diff / np.sqrt(var + 1e-12)
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return aucs[0], aucs[1], diff, z, p


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
