from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42

# Limite raisonnable pour 12 Go de RAM.
MAX_GRAPH_FLOWS = 150_000
MAX_NEIGHBORS_PER_FLOW = 2

# Contrat Cloud CIC-IDS2017 ↔ IoT CICIoT2023.
ALIGNED_FEATURES = [
    "feat_01_protocol",
    "feat_02_packet_count",
    "feat_03_total_bytes",
    "feat_04_packet_len_mean",
    "feat_05_packet_len_std",
    "feat_06_packet_len_var",
    "feat_07_packet_len_min",
    "feat_08_packet_len_max",
    "feat_09_flow_rate",
    "feat_10_iat_mean",
    "feat_11_syn_flag",
    "feat_12_ack_flag",
    "feat_13_fin_flag",
    "feat_14_rst_flag",
    "feat_15_psh_flag",
]