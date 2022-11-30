import json, argparse

import numpy as np

from recommend.utils import *

data_directory = "data/filtered"


class EaseR:
    def __init__(self, X_train: np.ndarray, lambda_: float = 10.0):
        X_train = self.transform_interaction_matrix(X_train)
        G = X_train.T.dot(X_train)
        U, I = X_train.shape
        diagIndices = np.diag_indices(I)
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B

    def transform_interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        return (X >= 75).astype(float)

    def recommend(self, X_test: np.ndarray, k: int = 100) -> np.ndarray:
        X_test = self.transform_interaction_matrix(X_test)
        scores = X_test @ self.B
        topk = np.argsort(-scores, axis=1)[:, :k]
        return topk


def main():
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_", default=10.0, type=float)
    parser.add_argument("--data_dir", default="data/filtered", type=str)
    args = parser.parse_args()

    print("config")
    print(
        json.dumps(
            {
                "lambda": args.lambda_,
            }
        )
    )

    album_df, user_dfs = load_dataframes(args.data_dir)
    (user_train_df, user_val_df, user_test_df) = user_dfs

    X_train = generate_user_item_matrix(user_train_df, album_df)
    X_val = generate_user_item_matrix(user_val_df, album_df)
    X_test = generate_user_item_matrix(user_test_df, album_df)

    rec_alg = EaseR(X_train, args.lambda_)

    rec_result = rec_alg.recommend(X_val)
    summary = evaluate(rec_result, X_val)
    print("val performance")
    print(json.dumps(summary, indent=2))

    rec_result = rec_alg.recommend(X_test)
    summary = evaluate(rec_result, X_test)
    print("test performance")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
