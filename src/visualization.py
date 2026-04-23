import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve, auc
import shap
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curves(y_test, w_test, plot_data):
    plt.figure(figsize=(8,6))
    for name, probs in plot_data.items():
        fpr, tpr, _ = roc_curve(y_test, probs, sample_weight=w_test)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.title("ROC Curves (Test Set)"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def plot_calibration(y_test, plot_data):
    plt.figure(figsize=(8,6))
    for name, probs in plot_data.items():
        p_true, p_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(p_pred, p_true, 'o-', lw=1, label=name)
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.title("Calibration Curves (Test Set)"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def plot_pr_curves(y_test, plot_data):
    plt.figure(figsize=(7,6))
    for name, probs in plot_data.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.plot(recall, precision, lw=2, label=name)
    plt.title("Precision-Recall Curves (Test Set)"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()



def plot_shap_combined(
    model,
    X,
    feature_name_map=None,
    max_display=12,
    figsize=(16, 6),
    save_path=None
):
    """
    Creates publication-ready SHAP figure:
    (A) Beeswarm summary
    (B) Feature importance (bar)

    Parameters:
    -----------
    model : trained tree-based model
    X : dataframe (test set)
    feature_name_map : dict for renaming features
    max_display : number of top features
    figsize : figure size
    save_path : optional save path
    """

    # -------- Feature Renaming --------
    if feature_name_map is not None:
        X_plot = X.rename(columns=feature_name_map)
    else:
        X_plot = X.copy()

    # -------- SHAP Values --------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_plot)

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # -------- Sort Features --------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    X_sorted = X_plot.iloc[:, sorted_idx]
    shap_sorted = shap_values[:, sorted_idx]

    # -------- Plot --------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=figsize)

    # A. Beeswarm
    plt.subplot(1, 2, 1)
    shap.summary_plot(
        shap_sorted,
        X_sorted,
        show=False,
        max_display=max_display
    )
    plt.title("A. SHAP Summary Plot", fontsize=12, fontweight='bold')

    # B. Bar
    plt.subplot(1, 2, 2)
    shap.summary_plot(
        shap_sorted,
        X_sorted,
        plot_type="bar",
        show=False,
        max_display=max_display
    )
    plt.title("B. Global Feature Importance", fontsize=12, fontweight='bold')

    # -------- Layout --------
    plt.tight_layout(pad=2.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return shap_values
