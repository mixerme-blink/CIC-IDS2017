import numpy as np
import pandas as pd

from config import HARMONIZED_FEATURES, PROCESSED_DATA_DIR


def numeric_column(df, column_name):
    """Convertit une colonne en nombres ; si absente, retourne des zéros."""
    if column_name not in df.columns:
        print(f"Attention : colonne absente -> {column_name}")
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(
        df[column_name],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).fillna(0)


def main():
    input_path = PROCESSED_DATA_DIR / "cicids_sample.parquet"
    output_path = PROCESSED_DATA_DIR / "cloud_aligned.parquet"

    df = pd.read_parquet(input_path).copy()

    # Les 8 features finales harmonisées avec CICIoT2023.
    aligned_df = pd.DataFrame(index=df.index)

    aligned_df["packet_count"] = (
        numeric_column(df, "Total Fwd Packets")
        + numeric_column(df, "Total Backward Packets")
    )

    aligned_df["total_bytes"] = (
        numeric_column(df, "Total Length of Fwd Packets")
        + numeric_column(df, "Total Length of Bwd Packets")
    )

    aligned_df["packet_len_mean"] = numeric_column(
        df,
        "Average Packet Size",
    )

    aligned_df["packet_len_std"] = numeric_column(
        df,
        "Packet Length Std",
    )

    aligned_df["packet_len_var"] = numeric_column(
        df,
        "Packet Length Variance",
    )

    aligned_df["packet_len_min"] = numeric_column(
        df,
        "Min Packet Length",
    )

    aligned_df["packet_len_max"] = numeric_column(
        df,
        "Max Packet Length",
    )

    aligned_df["flow_rate"] = numeric_column(
        df,
        "Flow Packets/s",
    )

    # Colonnes nécessaires pour construire le graphe de flows.
    metadata_columns = [
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
        "target",
    ]

    result_df = df[metadata_columns].copy()

    if list(aligned_df.columns) != HARMONIZED_FEATURES:
        raise AssertionError("L'ordre des features harmonisées est invalide.")

    for feature in HARMONIZED_FEATURES:
        result_df[feature] = aligned_df[feature]

    result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)

    result_df.to_parquet(output_path, index=False)

    print(f"Fichier harmonisé créé : {output_path}")
    print(f"Nombre de flows : {len(result_df):,}")
    print("\nFeatures finales :")

    for index, feature in enumerate(HARMONIZED_FEATURES, start=1):
        print(f"{index}. {feature}")


if __name__ == "__main__":
    main()
