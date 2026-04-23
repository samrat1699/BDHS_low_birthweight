import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import create_inclusion_status, compare_groups, print_group_stats
from src.models import prepare_splits, initialize_models
from src.evaluation import get_calibrated_probs, evaluate_metrics, get_bootstrap_results, optimize_threshold
from src.visualization import plot_roc_curves, plot_calibration, plot_pr_curves

def main():
    DATA_PATH = "data/BDHS_2022_LBW.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Place your dataset at {DATA_PATH}")

    df = load_data(DATA_PATH)
    df = create_inclusion_status(df)

    cat_vars = ["B4", "B0", "BORD", "V106", "V714", "V701", "V190", "V130", "V151", "V102", "V024", "M45"]
    cont_vars = ["V012", "V445", "V212", "V511", "M14"]

    compare_groups(df, cat_vars, cont_vars)
    print_group_stats(df, cat_vars, cont_vars)

    target, design_cols = "low_birth", ["V005", "V021", "V023", "wt"]
    splits, feature_names = prepare_splits(df, None, target, design_cols)
    (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test) = splits["train"], splits["val"], splits["test"]

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models = initialize_models(ratio)

    master_results, plot_data = [], {}
    for name, model in models.items():
        print(f"\nTraining: {name}...")
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)

        train_raw = model.predict_proba(X_train)[:, 1]
        train_prob = get_calibrated_probs(model, X_train, y_train, train_raw)
        val_prob   = get_calibrated_probs(model, X_val, y_train, train_raw)
        test_prob  = get_calibrated_probs(model, X_test, y_train, train_raw)

        best_t = optimize_threshold(val_prob, y_val, w_val)
        master_results.append([f"{name} (Train)"] + get_bootstrap_results(y_train, train_prob, w_train, best_t))
        master_results.append([f"{name} (Val)"]   + get_bootstrap_results(y_val, val_prob, w_val, best_t))
        master_results.append([f"{name} (Test)"]  + get_bootstrap_results(y_test, test_prob, w_test, best_t))
        plot_data[name] = test_prob

    cols = ["Model/Set", "Acc", "AUC", "PR_AUC", "Kappa", "Precision", "Recall", "F1", "Brier", "Slope", "Intercept","Sens", "Spec", "Thresh"]
    print("\n" + "="*160 + "\nFINAL REPORT WITH 95% CIs\n" + "="*160)
    print(pd.DataFrame(master_results, columns=cols).to_string(index=False))
    model_names = list(plot_data.keys())

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
    
            m1, m2 = model_names[i], model_names[j]
    
            auc1, auc2, diff, p = delong_roc_test(
                y_test.values,
                plot_data[m1],
                plot_data[m2]
            )
    
            print(f"{m1} vs {m2}: "
                  f"AUC1={auc1:.3f}, AUC2={auc2:.3f}, "
                  f"Diff={diff:.3f}, p={p:.5f}")


    plot_roc_curves(y_test, w_test, plot_data)
    plot_calibration(y_test, plot_data)
    plot_pr_curves(y_test, plot_data)
    plot_shap_combined(
    model=xgb_model,
    X=X_test,
    feature_name_map=feature_map,
    save_path="Figure_SHAP_Combined.png"
)

if __name__ == "__main__":
    main()
