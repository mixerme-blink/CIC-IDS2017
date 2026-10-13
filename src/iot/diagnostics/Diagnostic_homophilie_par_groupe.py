"""
Diagnostic : l'homophilie Benign gonflee (94.7%) vient-elle du fait que
les noeuds SYNTHETIQUES forment un cluster artificiellement dense entre
eux, alors que les vrais noeuds Benign (train et test) ont une
homophilie naturellement plus basse et plus proche l'une de l'autre ?

Si oui : le GNN s'entraine majoritairement sur un pattern de voisinage
"facile" (largement synthetique) qui ne correspond pas a la structure
locale reelle des vrais Benign -> decalage de distribution train/test
au niveau de la fonction d'agregation, meme sans fuite directe via le
graphe (le diagnostic precedent a confirme 0% de voisins synthetiques
cote test).

Utilisation :
    python diagnostic_homophilie_par_groupe.py
"""

import numpy as np
import torch

GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"


def homophily_for_nodes(node_ids, y, src, dst):
    ratios = []
    for node in node_ids:
        neighbors = dst[src == node]
        if len(neighbors) == 0:
            continue
        ratios.append((y[neighbors] == 0).mean())
    return np.array(ratios)


if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False)

    y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    is_synthetic = data.is_synthetic.numpy()
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    src, dst = edge_index[0], edge_index[1]

    real_train_benign = np.where((y == 0) & train_mask & (~is_synthetic))[0]
    synth_benign = np.where((y == 0) & is_synthetic)[0]
    test_benign = np.where((y == 0) & test_mask)[0]

    print(f"\n[INFO] Noeuds Benign REELS train      : {len(real_train_benign)}")
    print(f"[INFO] Noeuds Benign SYNTHETIQUES      : {len(synth_benign)}")
    print(f"[INFO] Noeuds Benign test               : {len(test_benign)}")

    r_real = homophily_for_nodes(real_train_benign, y, src, dst)
    r_synth = homophily_for_nodes(synth_benign, y, src, dst)
    r_test = homophily_for_nodes(test_benign, y, src, dst)

    print("\n=== HOMOPHILIE BENIGN PAR GROUPE ===")
    print(f"  Reels TRAIN       : {r_real.mean():.1%}  (n={len(r_real)})")
    print(f"  SYNTHETIQUES      : {r_synth.mean():.1%}  (n={len(r_synth)})")
    print(f"  Reels TEST        : {r_test.mean():.1%}  (n={len(r_test)})")

    print("\n=== VERDICT ===")
    gap_synth_vs_real = r_synth.mean() - r_real.mean()
    gap_real_train_vs_test = abs(r_real.mean() - r_test.mean())

    if gap_synth_vs_real > 0.15 and gap_real_train_vs_test < 0.15:
        print("Les noeuds SYNTHETIQUES ont une homophilie nettement plus haute")
        print("que les vrais noeuds Benign (train ET test, qui eux se ressemblent).")
        print("-> Confirme : SMOTE cree un pattern de voisinage artificiellement")
        print("   'facile' qui domine numeriquement le train (7682 vs 821 reels).")
        print("   Le GNN apprend une fonction d'agregation calibree sur ce pattern")
        print("   synthetique, qui generalise mal aux vrais Benign du test.")
        print("   -> Probable decalage de distribution (train majoritairement")
        print("      synthetique-facile / test 100% reel-difficile), MEME SANS")
        print("      fuite directe via le graphe (confirme par le diagnostic")
        print("      precedent : 0% de voisins synthetiques cote test).")
    else:
        print("Pas d'ecart net entre synthetiques et reels, ou ecart train/test")
        print("significatif meme sur les reels seuls -> revoir d'autres pistes")
        print("(architecture du GNN, hyperparametres, profondeur).")