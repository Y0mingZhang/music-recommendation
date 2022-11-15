import os, json, functools
from typing import Any
from os.path import join, dirname, basename
from datetime import timezone
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

import bs4
from bs4 import BeautifulSoup
from dateutil import parser as datetime_parser
from tqdm.auto import tqdm
from requests_html import HTMLSession

output_dir = "data/albums"
domain = "https://www.albumoftheyear.org"
album_url = "https://www.albumoftheyear.org/album/"

session = HTMLSession()


def convert_date_to_timestamp(s: str) -> int:
    dt = datetime_parser.parse(s)

    return round(dt.replace(tzinfo=timezone.utc).timestamp())


def fix_url(func: callable) -> callable:
    @functools.wraps(func)
    def f(url: str) -> Any:
        return func(urljoin(domain, url))

    return f


def remove_non_ascii(s: str) -> str:
    return "".join(char for char in s if ord(char) < 128)


def non_span_inner(div: bs4.element.Tag) -> str:
    return " ".join(
        map(
            lambda element: element.text.strip(),
            filter(lambda elt: elt.name != "span", div.children),
        )
    ).strip()


def album_idx_to_url(idx: int) -> str:
    return urljoin(album_url, str(idx))


def album_exists(idx: int) -> bool:
    exists = session.get(album_idx_to_url(idx)).url != domain
    return exists


def album_count() -> int:
    lo = hi = 1
    while album_exists(hi):
        lo = hi
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if album_exists(mid):
            lo = mid
        else:
            hi = mid
    return lo


def parse_album_detail(soup: BeautifulSoup) -> dict[str, Any]:
    artist = soup.find("div", class_="artist").span.text
    album = soup.find("div", class_="albumTitle").span.text
    parsed = {"artist": artist, "album": album}
    keys = {
        "Release Date": "release_date",
        "Format": "format",
        "Label": "label",
        "Genre": "genre",
    }

    for div in soup.find_all("div", class_="detailRow"):
        desc = div.span.text
        for key_old, key_new in keys.items():
            if key_old in desc:
                parsed[key_new] = non_span_inner(div)
                break

    return parsed


@fix_url
def scrape_album_reviews(url: str) -> list[dict[str, Any]]:
    r = session.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    reviews = []
    for div in soup.find_all("div", class_="userRatingBlock"):
        reviewer_id = basename(dirname(div.find("div", class_="userName").a["href"]))
        reviewer_name = div.find("div", class_="userName").text
        review_time = convert_date_to_timestamp(div.find("div", class_="date")["title"])
        rating = div.find("div", class_="rating").text
        reviews.append(
            {
                "reviewer_id": reviewer_id,
                "reviewer_name": reviewer_name,
                "review_time": review_time,
                "rating": rating,
            }
        )

    if soup.find("div", class_="pageSelect next"):
        return reviews + scrape_album_reviews(
            soup.find("div", class_="pageSelect next").parent["href"]
        )
    else:
        return reviews


def scrape_album(idx: int) -> None:
    output_path = join(output_dir, f"{idx}.json")
    if os.path.exists(output_path):
        return

    url = album_idx_to_url(idx)
    r = session.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    album = {"album_id": idx}
    album.update(parse_album_detail(soup))
    reviews_div = soup.find("div", class_="albumUserScoreBox").find(
        "div", class_="text numReviews"
    )
    if reviews_div:
        album_reviews_url = reviews_div.a["href"]
        album["reviews"] = scrape_album_reviews(album_reviews_url)

    else:
        album["reviews"] = []

    with open(output_path, "w") as f:
        json.dump(album, f, indent=2)


def main():
    os.makedirs(output_dir, exist_ok=True)
    lo = 1
    hi = album_count()

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(scrape_album, range(1, hi + 1)), total=hi))

    print("Done!")


if __name__ == "__main__":
    main()
