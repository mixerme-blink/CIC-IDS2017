import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, ALIGNED_FEATURES


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

    # Les 15 features harmonisées avec CICIoT2023.
    aligned_df = pd.DataFrame(index=df.index)

    aligned_df["feat_01_protocol"] = numeric_column(df, "Protocol")

    aligned_df["feat_02_packet_count"] = (
        numeric_column(df, "Total Fwd Packets")
        + numeric_column(df, "Total Backward Packets")
    )

    aligned_df["feat_03_total_bytes"] = (
        numeric_column(df, "Total Length of Fwd Packets")
        + numeric_column(df, "Total Length of Bwd Packets")
    )

    aligned_df["feat_04_packet_len_mean"] = numeric_column(
        df,
        "Average Packet Size",
    )

    aligned_df["feat_05_packet_len_std"] = numeric_column(
        df,
        "Packet Length Std",
    )

    aligned_df["feat_06_packet_len_var"] = numeric_column(
        df,
        "Packet Length Variance",
    )

    aligned_df["feat_07_packet_len_min"] = numeric_column(
        df,
        "Min Packet Length",
    )

    aligned_df["feat_08_packet_len_max"] = numeric_column(
        df,
        "Max Packet Length",
    )

    # Important : Rate IoT correspond à Flow Packets/s, pas Flow Bytes/s.
    aligned_df["feat_09_flow_rate"] = numeric_column(
        df,
        "Flow Packets/s",
    )

    aligned_df["feat_10_iat_mean"] = numeric_column(
        df,
        "Flow IAT Mean",
    )

    aligned_df["feat_11_syn_flag"] = numeric_column(
        df,
        "SYN Flag Count",
    )

    aligned_df["feat_12_ack_flag"] = numeric_column(
        df,
        "ACK Flag Count",
    )

    aligned_df["feat_13_fin_flag"] = numeric_column(
        df,
        "FIN Flag Count",
    )

    aligned_df["feat_14_rst_flag"] = numeric_column(
        df,
        "RST Flag Count",
    )

    aligned_df["feat_15_psh_flag"] = numeric_column(
        df,
        "PSH Flag Count",
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

    for feature in ALIGNED_FEATURES:
        result_df[feature] = aligned_df[feature]

    result_df = result_df.replace([np.inf, -np.inf], 0).fillna(0)

    result_df.to_parquet(output_path, index=False)

    print(f"Fichier harmonisé créé : {output_path}")
    print(f"Nombre de flows : {len(result_df):,}")
    print("\nFeatures finales :")

    for index, feature in enumerate(ALIGNED_FEATURES, start=1):
        print(f"{index}. {feature}")


if __name__ == "__main__":
    main()