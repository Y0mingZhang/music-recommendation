import random
from os.path import join

import numpy as np
import pandas as pd
from pygini import gini
from scipy.sparse import csr_array


def load_dataframes(directory: str):
    album_df = pd.read_json(
        join(directory, "albums.jsonl"), lines=True, orient="records"
    )

    user_dfs = [
        pd.read_json(
            join(directory, f"users_{split}.jsonl"), lines=True, orient="records"
        )
        for split in ("train", "val", "test")
    ]
    return album_df, user_dfs


def generate_user_item_matrix(
    user_df: pd.DataFrame, album_df: pd.DataFrame
) -> np.ndarray:
    album_id_to_idx = album_df.reset_index().set_index("album_id")["index"]
    U, I = len(user_df), len(album_df)
    data = []
    row_indices = []
    col_indices = []
    for i, row in user_df.iterrows():
        for review in row["reviews"]:
            if review["album_id"] in album_id_to_idx:
                data.append(review["rating"])
                row_indices.append(i)
                col_indices.append(album_id_to_idx[review["album_id"]])

    return csr_array((data, (row_indices, col_indices)), shape=(U, I)).toarray()


def evaluate(recs, X_test: np.ndarray, k: int = 20):
    I = X_test.shape[1]
    Ps = []
    Rs = []
    F1s = []
    coverage = np.zeros(I)
    for i in range(len(recs)):
        retrieved = set(recs[i][:k])
        relevant = set(np.flatnonzero(X_test[i]))
        precision = len(retrieved & relevant) / len(retrieved)
        recall = len(retrieved & relevant) / len(relevant)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        for item_idx in retrieved:
            coverage[item_idx] += 1

        Ps.append(precision)
        Rs.append(recall)
        F1s.append(f1)

    return {
        f"Precision @ {k}": np.mean(Ps),
        f"Recall @ {k}": np.mean(Rs),
        f"F1 @ {k}": np.mean(F1s),
        f"Item Coverage @ {k}": (coverage > 0).sum() / I,
        f"Gini @ {k}": gini(coverage.astype(float)),
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
