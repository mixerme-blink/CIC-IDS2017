import pandas as pd

from config import PROCESSED_DATA_DIR, MAX_ROWS, RANDOM_STATE


def create_balanced_sample(df, total_rows):
    benign = df[df["target"] == 0]
    attacks = df[df["target"] == 1]

    per_class = total_rows // 2
    available = min(len(benign), len(attacks), per_class)

    benign_sample = benign.sample(
        n=available,
        random_state=RANDOM_STATE
    )

    attack_sample = attacks.sample(
        n=available,
        random_state=RANDOM_STATE
    )

    result = pd.concat(
        [benign_sample, attack_sample],
        ignore_index=True
    )

    return result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "cicids_clean.parquet"
    output_path = PROCESSED_DATA_DIR / "cicids_sample.parquet"

    df = pd.read_parquet(input_path)
    sample_df = create_balanced_sample(df, MAX_ROWS)

    sample_df.to_parquet(output_path, index=False)

    print(f"Échantillon sauvegardé : {output_path}")
    print(sample_df["target"].value_counts())