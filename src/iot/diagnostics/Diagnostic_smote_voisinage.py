"""
Diagnostic : les voisins Benign des noeuds TEST sont-ils des noeuds
Benign REELS (train) ou des noeuds SYNTHETIQUES (SMOTE) ?

Si la reponse est "majoritairement synthetiques", ca confirme que
l'homophilie train (94.7%) est gonflee artificiellement par SMOTE et
que le GNN apprend un pattern de voisinage qui n'existe pas au moment
du test -- d'ou sa sous-performance face au RF malgre un graphe
globalement tres homophile.

Necessite que Build_knn_graph_iot.py ait ete relance avec le flag
data.is_synthetic (voir patch_is_synthetic.md).

Utilisation :
    python diagnostic_smote_voisinage.py
"""

import numpy as np
import torch

GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"

if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False)

    if not hasattr(data, "is_synthetic"):
        raise AttributeError(
            "Le graphe n'a pas d'attribut 'is_synthetic'. "
            "Applique le patch dans Build_knn_graph_iot.py et regenere "
            "le graphe avant de relancer ce diagnostic."
        )

    y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    is_synthetic = data.is_synthetic.numpy()
    test_mask = data.test_mask.numpy()

    src, dst = edge_index[0], edge_index[1]

    test_benign_ids = np.where((y == 0) & test_mask)[0]
    print(f"\n[INFO] {len(test_benign_ids)} noeuds Benign dans le test set")

    real_neighbor_ratios = []
    synth_neighbor_ratios = []
    attack_neighbor_ratios = []

    for node in test_benign_ids:
        neighbors = dst[src == node]
        if len(neighbors) == 0:
            continue

        n_total = len(neighbors)
        n_benign_real = np.sum((y[neighbors] == 0) & (~is_synthetic[neighbors]))
        n_benign_synth = np.sum((y[neighbors] == 0) & (is_synthetic[neighbors]))
        n_attack = np.sum(y[neighbors] == 1)

        real_neighbor_ratios.append(n_benign_real / n_total)
        synth_neighbor_ratios.append(n_benign_synth / n_total)
        attack_neighbor_ratios.append(n_attack / n_total)

    real_neighbor_ratios = np.array(real_neighbor_ratios)
    synth_neighbor_ratios = np.array(synth_neighbor_ratios)
    attack_neighbor_ratios = np.array(attack_neighbor_ratios)

    print("\n=== COMPOSITION MOYENNE DU VOISINAGE DES NOEUDS BENIGN TEST ===")
    print(f"  Voisins Benign REELS (train)      : {real_neighbor_ratios.mean():.1%}")
    print(f"  Voisins Benign SYNTHETIQUES (SMOTE): {synth_neighbor_ratios.mean():.1%}")
    print(f"  Voisins Attack                     : {attack_neighbor_ratios.mean():.1%}")

    total_benign_neighbor_mass = real_neighbor_ratios.sum() + synth_neighbor_ratios.sum()
    if total_benign_neighbor_mass > 0:
        pct_synth_among_benign_neighbors = (
            synth_neighbor_ratios.sum() / total_benign_neighbor_mass
        )
        print(f"\n  Parmi les voisins Benign (reels + synthetiques) trouves,")
        print(f"  proportion qui sont SYNTHETIQUES : {pct_synth_among_benign_neighbors:.1%}")

    print("\n=== VERDICT ===")
    if synth_neighbor_ratios.mean() > real_neighbor_ratios.mean() * 1.5:
        print("Les noeuds Benign du test s'appuient tres majoritairement sur des")
        print("voisins SYNTHETIQUES (SMOTE) plutot que sur de vrais noeuds Benign.")
        print("-> Ceci confirme l'hypothese : le graphe cree un signal de voisinage")
        print("   qui depend fortement d'interpolations SMOTE, pas de vraies donnees")
        print("   Benign -- ce qui explique l'ecart d'homophilie train/test et la")
        print("   sous-performance du GNN par rapport au RF.")
    else:
        print("Les voisins Benign reels et synthetiques sont dans des proportions")
        print("comparables -- l'hypothese SMOTE seule n'explique pas tout, il faut")
        print("regarder d'autres pistes (profondeur du GNN, hyperparametres, etc.)")