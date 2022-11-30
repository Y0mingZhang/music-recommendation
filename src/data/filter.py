import os

import pandas as pd

album_path = "data/processed/albums.jsonl"
user_path = "data/processed/users.jsonl"


def filter_user_df(
    user_df: pd.DataFrame, album_id_to_idx: dict[int, int], k=10
) -> pd.DataFrame:
    # filter for user who has reviewed at least k of albums used
    keep_row = [
        sum(
            1 if r["album_id"] in album_id_to_idx and r["rating"] >= 75 else 0
            for r in reviews
        )
        >= k
        for reviews in user_df["reviews"]
    ]
    return user_df[keep_row]


def main():
    album_df = pd.read_json(album_path, lines=True, orient="records")
    # keep albums with total reviews >= 100
    album_df = album_df[album_df["num_reviews"] >= 100].reset_index(drop=True)
    album_id_to_idx = {album_id: i for i, album_id in enumerate(album_df["album_id"])}

    # keep users who like (rating >= 75) at least 10 albums
    user_df = pd.read_json(user_path, lines=True, orient="records")
    user_df = (
        filter_user_df(user_df, album_id_to_idx, k=10)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    I = len(album_df)
    U = len(user_df)

    user_train_df = user_df.loc[: round(U * 0.8)]
    user_val_df = user_df.loc[round(U * 0.8) : round(U * 0.9)]
    user_test_df = user_df.loc[round(U * 0.9) :]

    os.makedirs("data/filtered", exist_ok=True)
    album_df.to_json("data/filtered/albums.jsonl", lines=True, orient="records")
    user_df.to_json("data/filtered/users.jsonl", lines=True, orient="records")
    user_train_df.to_json(
        "data/filtered/users_train.jsonl", lines=True, orient="records"
    )
    user_val_df.to_json("data/filtered/users_val.jsonl", lines=True, orient="records")
    user_test_df.to_json("data/filtered/users_test.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    main()
