import json

from config import PROCESSED_DATA_DIR, ALIGNED_FEATURES


CLOUD_MAPPING = {
    "feat_01_protocol": "Protocol",
    "feat_02_packet_count": "Total Fwd Packets + Total Backward Packets",
    "feat_03_total_bytes": (
        "Total Length of Fwd Packets + "
        "Total Length of Bwd Packets"
    ),
    "feat_04_packet_len_mean": "Average Packet Size",
    "feat_05_packet_len_std": "Packet Length Std",
    "feat_06_packet_len_var": "Packet Length Variance",
    "feat_07_packet_len_min": "Min Packet Length",
    "feat_08_packet_len_max": "Max Packet Length",
    "feat_09_flow_rate": "Flow Packets/s",
    "feat_10_iat_mean": "Flow IAT Mean",
    "feat_11_syn_flag": "SYN Flag Count",
    "feat_12_ack_flag": "ACK Flag Count",
    "feat_13_fin_flag": "FIN Flag Count",
    "feat_14_rst_flag": "RST Flag Count",
    "feat_15_psh_flag": "PSH Flag Count",
}


def main():
    output_path = PROCESSED_DATA_DIR / "feature_mapping.json"

    content = {
        "dataset": "CIC-IDS2017",
        "feature_order": ALIGNED_FEATURES,
        "cloud_mapping": CLOUD_MAPPING,
        "normalization": (
            "Z-score ; scaler ajusté uniquement sur les flows du train set."
        ),
        "label_binary": {
            "0": "BENIGN",
            "1": "ATTACK",
        },
        "graph_definition": {
            "node": "Un flow réseau",
            "node_features": "Les 15 features harmonisées",
            "node_label": "label_binary",
            "edges": (
                "Flows successifs partageant une même IP "
                "source ou destination."
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)

    print(f"Contrat Cloud sauvegardé : {output_path}")


if __name__ == "__main__":
    main()