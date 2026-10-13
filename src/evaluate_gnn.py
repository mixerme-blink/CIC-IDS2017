import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

import argparse

import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from config import N_FEATURES
from gnn_models import get_model


GRAPH_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "cloud_flow_graph.pt"
)

CHECKPOINT_DIR = (
    MODELS_DIR
    / "checkpoints"
)

FIGURES_DIR = (
    ROOT_DIR
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_graph():

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    return graph


@torch.no_grad()
def evaluate(model, data):

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    test_logits = logits[data.test_mask]

    probabilities = torch.softmax(
        test_logits,
        dim=1,
    )[:, 1]

    predictions = test_logits.argmax(
        dim=1
    )

    labels = data.y[data.test_mask]

    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_prob = probabilities.cpu().numpy()

    return y_true, y_pred, y_prob


def main(model_name):

    device = get_device()

    print("=" * 70)
    print("EVALUATION GNN - CIC-IDS2017")
    print("=" * 70)

    print(f"Modèle : {model_name}")
    print(f"Device : {device}")

    # -----------------------------------------------------
    # LOAD GRAPH
    # -----------------------------------------------------

    data = load_graph()

    if data.x.shape[1] != N_FEATURES:
        raise ValueError(
            f"Le graphe contient {data.x.shape[1]} features ; {N_FEATURES} attendues."
        )

    data = data.to(device)

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_best.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            f"Entraîne d'abord le modèle avec train_gnn.py."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = get_model(
        model_name,
        in_channels=N_FEATURES,
        num_classes=2,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    y_true, y_pred, y_prob = evaluate(
        model,
        data,
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTATS TEST")
    print("=" * 70)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\n--- Classification Report ---")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "BENIGN",
            "ATTACK",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"Matrice de confusion - {model_name.upper()}"
    )

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nMatrice sauvegardée : {figure_path}"
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics_path = (
        CHECKPOINT_DIR
        / f"{model_name}_metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Model: {model_name}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.6f}\n"
        )

        f.write(
            f"Recall: {recall:.6f}\n"
        )

        f.write(
            f"F1-score: {f1:.6f}\n"
        )

        f.write(
            f"ROC-AUC: {roc_auc:.6f}\n"
        )

    print(
        f"Métriques sauvegardées : {metrics_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gcn",
            "graphsage",
            "gat",
        ],
    )

    args = parser.parse_args()

    main(args.model)
