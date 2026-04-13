import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.drop(columns=["iron"], errors="ignore")
    df = df.rename(columns={"obesity": "maternal_bmi"})
    df["low_birth"] = df["low_birth"].astype(int)
    return df