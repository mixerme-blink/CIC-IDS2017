import pandas as pd
from config import RAW_DATA_DIR, CHUNK_SIZE


def load_cicids_csvs(max_rows=None):
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun CSV trouvé dans : {RAW_DATA_DIR}"
        )

    dataframes = []
    total_rows = 0

    for csv_file in csv_files:
        print(f"Lecture : {csv_file.name}")

        try:
            chunks = pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                encoding="latin1",
                low_memory=False,
            )

            for chunk in chunks:
                chunk.columns = chunk.columns.str.strip()
                dataframes.append(chunk)
                total_rows += len(chunk)

                if max_rows and total_rows >= max_rows:
                    return pd.concat(dataframes, ignore_index=True).head(max_rows)

        except Exception as error:
            print(f"Fichier ignoré : {csv_file.name} -> {error}")

    return pd.concat(dataframes, ignore_index=True)


if __name__ == "__main__":
    df = load_cicids_csvs(max_rows=100_000)
    print(df.shape)
    print(df.columns.tolist())