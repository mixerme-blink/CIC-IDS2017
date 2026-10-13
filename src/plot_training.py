from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    ROOT_DIR
    / "models"
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


MODELS = [
    "gcn",
    "graphsage",
    "gat",
]


def plot_model(model_name):

    history_path = (
        CHECKPOINT_DIR
        / f"{model_name}_history.pt"
    )

    if not history_path.exists():

        print(
            f"Historique absent : {model_name}"
        )

        return

    history = torch.load(
        history_path,
        map_location="cpu",
        weights_only=False,
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"Courbes d'apprentissage - {model_name.upper()}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / f"{model_name}_training_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Figure créée : {path}"
    )

    # -----------------------------------------------------
    # F1 VALIDATION
    # -----------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["val_f1"],
        label="Validation F1",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "F1-score"
    )

    plt.title(
        f"F1 validation - {model_name.upper()}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / f"{model_name}_validation_f1.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Figure créée : {path}"
    )


def main():

    for model in MODELS:

        plot_model(model)


if __name__ == "__main__":

    main()