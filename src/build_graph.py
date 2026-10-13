import numpy as np
import pandas as pd
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

from config import (
    PROCESSED_DATA_DIR,
    HARMONIZED_FEATURES,
    N_FEATURES,
    MAX_GRAPH_FLOWS,
    MAX_NEIGHBORS_PER_FLOW,
    RANDOM_STATE,
)


def balanced_graph_sample(df):
    """Réduit le graphe tout en gardant autant de normal que d'attaques."""
    if len(df) <= MAX_GRAPH_FLOWS:
        return df.sample(
            frac=1,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    normal_df = df[df["target"] == 0]
    attack_df = df[df["target"] == 1]

    per_class = MAX_GRAPH_FLOWS // 2

    normal_sample = normal_df.sample(
        n=min(per_class, len(normal_df)),
        random_state=RANDOM_STATE,
    )

    attack_sample = attack_df.sample(
        n=min(per_class, len(attack_df)),
        random_state=RANDOM_STATE,
    )

    return pd.concat(
        [normal_sample, attack_sample],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def connect_consecutive_flows(df):
    """
    Nœud = flow.
    Deux flows sont reliés s'ils partagent une IP et sont proches
    dans l'ordre temporel. Cette méthode évite de connecter toutes
    les paires d'un même hôte, ce qui saturerait la RAM.
    """
    order_df = df[
        ["Source IP", "Destination IP", "Timestamp"]
    ].copy()

    order_df["node_id"] = np.arange(len(order_df))

    order_df["Timestamp"] = pd.to_datetime(
        order_df["Timestamp"],
        errors="coerce",
    )

    # Les timestamps invalides sont placés en dernier.
    order_df = order_df.sort_values("Timestamp", na_position="last")

    edge_sources = []
    edge_destinations = []

    for ip_column in ["Source IP", "Destination IP"]:
        grouped = order_df.groupby(ip_column)["node_id"]

        for _, node_ids in grouped:
            node_ids = node_ids.to_numpy()

            for offset in range(1, MAX_NEIGHBORS_PER_FLOW + 1):
                if len(node_ids) <= offset:
                    continue

                source = node_ids[:-offset]
                destination = node_ids[offset:]

                # Arêtes bidirectionnelles : adaptées au GCN/GAT.
                edge_sources.extend(source.tolist())
                edge_destinations.extend(destination.tolist())

                edge_sources.extend(destination.tolist())
                edge_destinations.extend(source.tolist())

    if not edge_sources:
        raise ValueError("Aucune arête n'a été construite.")

    return torch.tensor(
        [edge_sources, edge_destinations],
        dtype=torch.long,
    )


def create_masks(labels):
    """Split stratifié : 70% train, 10% validation, 20% test."""
    indices = np.arange(len(labels))

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=2 / 3,
        random_state=RANDOM_STATE,
        stratify=labels[temporary_indices],
    )

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    validation_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[train_indices] = True
    validation_mask[validation_indices] = True
    test_mask[test_indices] = True

    return train_mask, validation_mask, test_mask


def main():
    input_path = PROCESSED_DATA_DIR / "cloud_features.parquet"
    graph_path = PROCESSED_DATA_DIR / "cloud_flow_graph.pt"
    scaler_path = PROCESSED_DATA_DIR / "cloud_train_scaler.joblib"
    metadata_path = PROCESSED_DATA_DIR / "cloud_flow_metadata.parquet"

    df = pd.read_parquet(input_path)

    required_columns = [
        "Source IP",
        "Destination IP",
        "Timestamp",
        "target",
    ] + HARMONIZED_FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes : {missing_columns}\n"
            "Exécute feature_engineering.py puis feature_selection.py."
        )

    # Nœuds = flows ; limite mémoire.
    df = balanced_graph_sample(df)

    y_np = df["target"].to_numpy()

    train_mask, validation_mask, test_mask = create_masks(y_np)

    if list(df[HARMONIZED_FEATURES].columns) != HARMONIZED_FEATURES:
        raise AssertionError("L'ordre des features harmonisées est invalide.")

    raw_x = df[HARMONIZED_FEATURES].to_numpy(dtype=np.float32)

    if not np.isfinite(raw_x).all():
        raise ValueError("Les features du graphe doivent être numériques et finies.")

    # Min-Max sans data leakage : fit uniquement sur le train set.
    scaler = MinMaxScaler()
    scaler.fit(raw_x[train_mask.numpy()])

    scaled_x = scaler.transform(raw_x)

    x = torch.tensor(scaled_x, dtype=torch.float32)

    if x.shape[1] != N_FEATURES:
        raise AssertionError("Le graphe doit contenir exactement 8 features par nœud.")
    y = torch.tensor(y_np, dtype=torch.long)

    edge_index = connect_consecutive_flows(df)

    cloud_graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(df),
    )

    cloud_graph.train_mask = train_mask
    cloud_graph.val_mask = validation_mask
    cloud_graph.test_mask = test_mask
    cloud_graph.feature_names = list(HARMONIZED_FEATURES)
    cloud_graph.graph_type = "flow_nodes_shared_ip_temporal_neighbors"

    torch.save(cloud_graph, graph_path)
    joblib.dump(scaler, scaler_path)

    # Utile pour interpréter ensuite les prédictions par flow/IP.
    df[
        [
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Label",
            "target",
        ]
    ].to_parquet(metadata_path, index=False)

    print("\n--- Graphe Cloud final créé ---")
    print(f"Fichier : {graph_path}")
    print("Type : nœud = flow ; arête = flows voisins partageant une IP")
    print(f"Nœuds / flows : {cloud_graph.num_nodes:,}")
    print(f"Arêtes : {cloud_graph.num_edges:,}")
    print(f"Features par nœud : {cloud_graph.x.shape[1]}")
    print(f"Train : {int(train_mask.sum()):,}")
    print(f"Validation : {int(validation_mask.sum()):,}")
    print(f"Test : {int(test_mask.sum()):,}")


if __name__ == "__main__":
    main()
