import os, glob, json, random
from os.path import join
from collections import defaultdict

import pandas as pd

scrape_dir = "data/albums"
processed_dir = "data/processed"


def process_reviews(reviews: list[dict]) -> list[dict]:
    processed = []
    for r in reviews:
        if r["rating"] != "NR":
            r["rating"] = int(r["rating"])
            processed.append(r)
    return processed


def user_view(album_df: pd.DataFrame) -> pd.DataFrame:
    users = defaultdict(list)
    for album_id, row in album_df.iterrows():
        for review in row["reviews"]:
            users[review["reviewer_id"], review["reviewer_name"]].append(
                {"album_id": album_id, "rating": review["rating"]}
            )

    data = []
    for user_id, username in sorted(users):
        data.append(
            {
                "user_id": user_id,
                "username": username,
                "reviews": users[(user_id, username)],
            }
        )
    return pd.DataFrame(data)


def summary(album_df: pd.DataFrame, user_df: pd.DataFrame) -> dict:

    return {
        "number of albums": len(album_df),
        "average number of reviews per album": album_df["num_reviews"].mean(),
        "number of users": len(user_df),
        "average number of reviews per album": user_df["num_reviews"].mean(),
        "sparsity of the user-album matrix": user_df["num_reviews"].sum()
        / (len(album_df) * len(user_df)),
    }


def main():
    os.makedirs(processed_dir, exist_ok=True)
    album_paths = sorted(glob.glob(join(scrape_dir, "*")))
    albums = []

    for album_path in album_paths:
        try:
            with open(album_path) as f:
                d = json.load(f)
                d["reviews"] = process_reviews(d["reviews"])
                if len(d["reviews"]) >= 100:
                    albums.append(d)
        except Exception as e:
            print(album_path)
            raise e

    album_df = pd.DataFrame(albums).set_index("album_id")
    album_df["num_reviews"] = album_df.reviews.apply(len)
    album_df["average_rating"] = (
        album_df.reviews.apply(lambda rs: sum(r["rating"] for r in rs))
        / album_df["num_reviews"]
    )
    album_df = album_df.dropna()

    user_df = user_view(album_df)
    user_df["num_reviews"] = user_df.reviews.apply(len)
    user_df["average_rating"] = (
        user_df.reviews.apply(lambda rs: sum(r["rating"] for r in rs))
        / user_df["num_reviews"]
    )
    user_df = user_df.dropna()

    album_df.reset_index().to_json(
        join(processed_dir, "albums.jsonl"), lines=True, orient="records"
    )
    user_df.to_json(join(processed_dir, "users.jsonl"), lines=True, orient="records")
    print(json.dumps(summary(album_df, user_df), indent=2))


if __name__ == "__main__":
    main()
