"""
Script d'inspection et d'échantillonnage stratifié pour CICIoT2023.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Structure du dataset gérée ici :
D:\\CICIoT2023\\CSV\\
    Benign_Final\\*.csv
    DDoS-SYN_Flood\\*.csv
    CommandInjection\\*.csv
    ...
Le label est déduit du NOM DU DOSSIER PARENT (plus fiable que de
supposer une colonne "Label" dans chaque fichier).

Étapes :
1. Recherche récursive de tous les fichiers CSV dans les sous-dossiers
2. Inspection d'un seul fichier (colonnes, types, aperçu)
3. Comptage du nombre de lignes par catégorie (dossier)
4. Échantillonnage stratifié sur TOUS les fichiers
5. Ajout d'une colonne label + label_binary (0=Benign, 1=Attack)
6. Sauvegarde d'un seul fichier de travail réduit

Utilisation :
    python ciciot2023_inspection_sampling.py
"""

import os
import glob
import pandas as pd

# ============================================================
# CONFIGURATION — À ADAPTER SI BESOIN
# ============================================================
DATASET_DIR = r"D:\CICIoT2023\CSV"       # dossier contenant les sous-dossiers par catégorie
OUTPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"
SAMPLE_FRACTION = 0.10                    # 10% des lignes par fichier, garde les proportions
RANDOM_STATE = 42                         # pour reproductibilité (à citer dans le rapport)
BENIGN_FOLDER_NAMES = ["benign_final", "benign"]  # noms de dossier considérés comme trafic normal


# ============================================================
# ÉTAPE 1 — Lister récursivement tous les CSV, avec leur catégorie
# ============================================================
def list_csv_files_recursive(dataset_dir):
    pattern = os.path.join(dataset_dir, "**", "*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier .csv trouvé sous {dataset_dir} (recherche récursive). "
            f"Vérifie le chemin DATASET_DIR."
        )

    # Chaque fichier -> (chemin, nom_categorie déduit du dossier parent direct)
    files_with_category = []
    for f in files:
        category = os.path.basename(os.path.dirname(f))
        files_with_category.append((f, category))

    categories = sorted(set(cat for _, cat in files_with_category))
    print(f"[INFO] {len(files)} fichiers CSV trouvés dans {len(categories)} catégories.")
    print(f"[INFO] Catégories détectées : {categories}")

    total_size_mb = sum(os.path.getsize(f) for f, _ in files_with_category) / 1_000_000
    print(f"[INFO] Taille totale : {total_size_mb:.1f} Mo")

    return files_with_category


# ============================================================
# ÉTAPE 2 — Inspecter un seul fichier
# ============================================================
def inspect_first_file(files_with_category):
    first_file, first_category = files_with_category[0]
    print(f"\n=== INSPECTION DU PREMIER FICHIER ({first_category}) ===")
    df_preview = pd.read_csv(first_file, nrows=1000)
    print(f"[INFO] Colonnes ({len(df_preview.columns)}) :")
    print(list(df_preview.columns))
    print("\n[INFO] Types de données :")
    print(df_preview.dtypes)
    print("\n[INFO] Aperçu des 5 premières lignes :")
    print(df_preview.head())

    has_label_column = any(c.lower() == "label" for c in df_preview.columns)
    print(f"\n[INFO] Colonne 'Label' présente dans le fichier ? {has_label_column}")
    return df_preview, has_label_column


# ============================================================
# ÉTAPE 3 — Compter le nombre de lignes par catégorie (rapide, sans tout charger)
# ============================================================
def count_rows_per_category(files_with_category):
    print("\n=== NOMBRE DE LIGNES PAR CATÉGORIE ===")
    counts = {}
    for f, category in files_with_category:
        # compte les lignes sans charger tout le contenu en mémoire
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            n_lines = sum(1 for _ in fh) - 1  # -1 pour l'en-tête
        counts[category] = counts.get(category, 0) + max(n_lines, 0)

    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s} : {n:>12,} lignes")
    print(f"\n[INFO] Total : {sum(counts.values()):,} lignes")
    return counts


# ============================================================
# ÉTAPE 4 — Échantillonnage stratifié sur tous les fichiers
# ============================================================
def stratified_sample_all_files(files_with_category, fraction, random_state, has_label_column):
    print(f"\n=== ÉCHANTILLONNAGE STRATIFIÉ ({fraction*100:.0f}% des lignes) ===")
    sampled_chunks = []

    for i, (f, category) in enumerate(files_with_category, 1):
        print(f"[INFO] Fichier {i}/{len(files_with_category)} : {category} / {os.path.basename(f)}")
        df_chunk = pd.read_csv(f)

        sampled = df_chunk.sample(frac=fraction, random_state=random_state)

        # Ajoute/écrase la colonne label avec le nom de catégorie (dossier)
        # -> fiable même si une colonne "Label" existe déjà mais est incohérente
        sampled = sampled.copy()
        sampled["label"] = category

        sampled_chunks.append(sampled)
        del df_chunk

    result = pd.concat(sampled_chunks, ignore_index=True)
    print(f"\n[INFO] Total lignes échantillonnées : {len(result)}")
    return result


# ============================================================
# ÉTAPE 5 — Ajouter une colonne label binaire (Benign / Attack)
# ============================================================
def add_binary_label(df):
    df["label_binary"] = df["label"].apply(
        lambda x: 0 if str(x).strip().lower() in BENIGN_FOLDER_NAMES else 1
    )
    print("\n[INFO] Répartition binaire (0=Benign, 1=Attack) :")
    print(df["label_binary"].value_counts())
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    files_with_category = list_csv_files_recursive(DATASET_DIR)
    df_preview, has_label_column = inspect_first_file(files_with_category)

    count_rows_per_category(files_with_category)

    df_sample = stratified_sample_all_files(
        files_with_category, SAMPLE_FRACTION, RANDOM_STATE, has_label_column
    )
    df_sample = add_binary_label(df_sample)

    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Fichier de travail sauvegardé : {OUTPUT_FILE}")
    print(f"[OK] Taille finale : {len(df_sample)} lignes, {len(df_sample.columns)} colonnes")