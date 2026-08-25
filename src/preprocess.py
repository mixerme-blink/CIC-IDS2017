import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, MAX_ROWS
from load_data import load_cicids_csvs


def clean_cicids_data(df):
    df = df.copy()

    # Uniformise les noms de colonnes de tous les CSV CIC-IDS2017.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required_columns = ["Source IP", "Destination IP", "Label"]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        print("Colonnes réellement trouvées :", df.columns.tolist())
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Suppression des flux sans endpoints
    df = df.dropna(subset=required_columns).copy()

    # Remplacement des valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan)

    # Les colonnes numériques avec NaN deviennent 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Normal = 0, toute autre catégorie = attaque = 1
    df["Label"] = df["Label"].astype(str).str.strip()
    df["target"] = (df["Label"].str.upper() != "BENIGN").astype(int)

    # Supprimer les lignes restantes invalides
    df = df.dropna()

    return df


if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_cicids_csvs(max_rows=MAX_ROWS)
    print("\nColonnes reçues par preprocess.py :")
    print(raw_df.columns.tolist())
    clean_df = clean_cicids_data(raw_df)

    output_path = PROCESSED_DATA_DIR / "cicids_clean.parquet"
    clean_df.to_parquet(output_path, index=False)

    print(f"Données sauvegardées : {output_path}")
    print(clean_df["target"].value_counts())