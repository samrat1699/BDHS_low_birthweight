import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

def create_inclusion_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Included_Status"] = df["M19A"].astype(str).str.contains("recall|card", case=False, na=False).astype(int)
    return df

def compare_groups(df: pd.DataFrame, cat_vars: list, cont_vars: list):
    print("\n--- Verified Group Counts ---")
    print(df["Included_Status"].value_counts())

    print("\n--- Table S1: Final Comparison Results ---")
    for var in cat_vars:
        ct = pd.crosstab(df[var].astype(str), df["Included_Status"])
        if ct.shape[1] == 2:
            _, p, _, _ = chi2_contingency(ct)
            print(f"Variable {var}: p-value = {p:.4f}")
        else:
            print(f"Variable {var}: Skipping (Could not find two groups to compare)")

    for var in cont_vars:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        g1 = df[df["Included_Status"] == 1][var].dropna()
        g0 = df[df["Included_Status"] == 0][var].dropna()
        if len(g1) > 0 and len(g0) > 0:
            _, p = ttest_ind(g1, g0)
            print(f"Continuous {var}: p-value = {p:.4f}")

def print_group_stats(df: pd.DataFrame, cat_vars: list, cont_vars: list):
    print("\n--- CATEGORICAL DATA FOR TABLE S1 ---")
    for var in cat_vars:
        ct = pd.crosstab(df[var], df["Included_Status"])
        pt = pd.crosstab(df[var], df["Included_Status"], normalize="columns") * 100
        print(f"\nResults for {var}:")
        for category in ct.index:
            n_excl, p_excl = ct.loc[category, 0], pt.loc[category, 0]
            n_incl, p_incl = ct.loc[category, 1], pt.loc[category, 1]
            print(f"  {category}: Included={n_incl} ({p_incl:.1f}%) | Excluded={n_excl} ({p_excl:.1f}%)")

    print("\n--- CONTINUOUS DATA (MEAN ± SD) ---")
    for var in cont_vars:
        stats = df.groupby("Included_Status")[var].agg(["mean", "std"]).round(2)
        print(f"{var}: Included={stats.loc[1, 'mean']} ± {stats.loc[1, 'std']} | Excluded={stats.loc[0, 'mean']} ± {stats.loc[0, 'std']}")