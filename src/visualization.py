import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve, auc

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