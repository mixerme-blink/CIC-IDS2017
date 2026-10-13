import sys
from pathlib import Path

# ---------------------------------------------------------
# Ajouter la racine du projet et models/ au PYTHONPATH
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))


import argparse
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score

from config import N_FEATURES
from gnn_models import get_model


# =========================================================
# CONFIGURATION
# =========================================================

GRAPH_PATH = ROOT_DIR / "data" / "processed" / "cloud_flow_graph.pt"

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

PATIENCE = 8


# =========================================================
# REPRODUCTIBILITE
# =========================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# CHARGEMENT DU GRAPHE
# =========================================================

def load_graph():

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            f"Graphe introuvable : {GRAPH_PATH}\n"
            "Exécute d'abord src/build_graph.py."
        )

    # Le fichier est notre propre fichier PyG.
    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    return graph


# =========================================================
# EVALUATION VALIDATION
# =========================================================

@torch.no_grad()
def evaluate(model, data, mask):

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    predictions = logits[mask].argmax(dim=1)

    labels = data.y[mask]

    accuracy = accuracy_score(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
    )

    f1 = f1_score(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        zero_division=0,
    )

    loss = F.cross_entropy(
        logits[mask],
        labels,
    )

    return (
        loss.item(),
        accuracy,
        f1,
    )


# =========================================================
# ENTRAINEMENT
# =========================================================

def train(model_name):

    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("ENTRAINEMENT GNN - CIC-IDS2017")
    print("=" * 70)

    print(f"Modèle : {model_name}")
    print(f"Device : {device}")
    print(f"Graphe : {GRAPH_PATH}")

    data = load_graph()

    if data.x.shape[1] != N_FEATURES:
        raise ValueError(
            f"Le graphe contient {data.x.shape[1]} features ; {N_FEATURES} attendues."
        )

    print("\n--- Graphe ---")
    print(f"Nœuds : {data.num_nodes:,}")
    print(f"Arêtes : {data.edge_index.shape[1]:,}")
    print(f"Features : {data.x.shape[1]}")

    print(f"Train : {int(data.train_mask.sum()):,}")
    print(f"Validation : {int(data.val_mask.sum()):,}")
    print(f"Test : {int(data.test_mask.sum()):,}")

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    data = data.to(device)

    # -----------------------------------------------------
    # MODELE
    # -----------------------------------------------------

    model = get_model(
        name=model_name,
        in_channels=N_FEATURES,
        num_classes=2,
    ).to(device)

    print("\n--- Modèle ---")
    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nParamètres entraînables : {total_parameters:,}")

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # TRAINING HISTORY
    # -----------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    patience_counter = 0

    # -----------------------------------------------------
    # EPOCHS
    # -----------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        train_loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        train_loss.backward()

        optimizer.step()

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_loss, val_accuracy, val_f1 = evaluate(
            model,
            data,
            data.val_mask,
        )

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping à l'époque {epoch}."
            )

            break

    # -----------------------------------------------------
    # RESTORE BEST MODEL
    # -----------------------------------------------------

    model.load_state_dict(best_state)

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_best.pt"
    )

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "in_channels": N_FEATURES,
            "num_classes": 2,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "history": history,
        },
        checkpoint_path,
    )

    print("\n" + "=" * 70)
    print("ENTRAINEMENT TERMINE")
    print("=" * 70)

    print(f"Meilleure époque : {best_epoch}")
    print(f"Meilleur Val F1 : {best_val_f1:.4f}")
    print(f"Checkpoint : {checkpoint_path}")

    # -----------------------------------------------------
    # SAUVEGARDER HISTORIQUE
    # -----------------------------------------------------

    history_path = (
        CHECKPOINT_DIR
        / f"{model_name}_history.pt"
    )

    torch.save(
        history,
        history_path,
    )

    return model


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=[
            "gcn",
            "graphsage",
            "gat",
        ],
        help="Architecture GNN.",
    )

    args = parser.parse_args()

    train(args.model)
