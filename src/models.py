import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def prepare_splits(df, features, target, design_cols):
    X = pd.get_dummies(df.drop(columns=[target] + design_cols), drop_first=True)
    y, w, clusters = df[target], df["wt"], df["V021"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss1.split(X, y, groups=clusters))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss2.split(X.iloc[train_val_idx], y.iloc[train_val_idx], groups=clusters.iloc[train_val_idx]))

    splits = {
        "train": (X.iloc[train_idx], y.iloc[train_idx], w.iloc[train_idx]),
        "val": (X.iloc[val_idx], y.iloc[val_idx], w.iloc[val_idx]),
        "test": (X.iloc[test_idx], y.iloc[test_idx], w.iloc[test_idx])
    }
    return splits, X.columns.tolist()

def initialize_models(ratio):
    return {
        "XGBoost": XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, gamma=0.2, reg_alpha=0.1,
            reg_lambda=1, scale_pos_weight=ratio, eval_metric="auc", tree_method="hist", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=10, min_samples_split=20,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=10, min_samples_split=20,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42
        ),
        "SVC": SVC(C=1, gamma="scale", kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=20, min_samples_split=40, class_weight="balanced", random_state=42
        )
    }