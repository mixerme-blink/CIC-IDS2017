import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    HARMONIZED_FEATURES,
    RANDOM_STATE,
)


def main():
    input_path = PROCESSED_DATA_DIR / "cloud_aligned.parquet"
    output_path = PROCESSED_DATA_DIR / "cloud_features.parquet"
    importance_path = PROCESSED_DATA_DIR / "feature_importance.csv"
    model_path = MODELS_DIR / "random_forest_harmonized.joblib"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    missing_features = [
        feature for feature in HARMONIZED_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features harmonisées absentes : {missing_features}"
        )

    # Random Forest : validation de l'utilité des features finales.
    # Il ne change ni leur ordre ni leur nombre.
    training_df = df.sample(
        n=min(100_000, len(df)),
        random_state=RANDOM_STATE,
    )

    X = training_df[HARMONIZED_FEATURES]
    y = training_df["target"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Entraînement Random Forest sur les 8 features harmonisées...")
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": HARMONIZED_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(importance_path, index=False)
    joblib.dump(model, model_path)

    # Sortie finale : métadonnées + exactement 8 features du contrat.
    output_columns = [
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
        "target",
    ] + HARMONIZED_FEATURES

    df[output_columns].to_parquet(output_path, index=False)
    df[output_columns].to_csv(
        PROCESSED_DATA_DIR / "cloud_features.csv",
        index=False,
    )

    print("\nImportance des features harmonisées :")
    print(importance_df.to_string(index=False))

    print(f"\nDataset final créé : {output_path}")
    print(f"Modèle baseline créé : {model_path}")


if __name__ == "__main__":
    main()
