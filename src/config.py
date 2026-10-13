from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
IOT_RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "ciciot2023"

CLOUD_GRAPH_PATH = PROCESSED_DATA_DIR / "cloud_flow_graph.pt"
IOT_GRAPH_PATH = PROCESSED_DATA_DIR / "iot_knn_graph.pt"
CLOUD_SCALER_PATH = PROCESSED_DATA_DIR / "cloud_train_scaler.joblib"
IOT_SCALER_PATH = PROCESSED_DATA_DIR / "iot_train_scaler.joblib"

RANDOM_STATE = 42

# Limite raisonnable pour 12 Go de RAM.
MAX_GRAPH_FLOWS = 150_000
MAX_NEIGHBORS_PER_FLOW = 2

# Contrat final Cloud CIC-IDS2017 ↔ IoT CICIoT2023.
# L'ordre est partagé avec la pipeline IoT et ne doit pas être modifié.
HARMONIZED_FEATURES = [
    "packet_count",
    "total_bytes",
    "packet_len_mean",
    "packet_len_std",
    "packet_len_var",
    "packet_len_min",
    "packet_len_max",
    "flow_rate",
]

N_FEATURES = len(HARMONIZED_FEATURES)
