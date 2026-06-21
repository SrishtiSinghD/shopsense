from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "All_Beauty.jsonl"
ARTIFACT_DIR = ROOT / "artifacts"

DOMAIN_STOPWORDS = {
    "product", "products", "amazon", "item", "items", "really", "very", "just",
    "use", "used", "using", "like", "good", "great", "nice", "love", "works",
    "work", "bought", "buy", "purchase", "received", "review", "stars",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS

CATEGORY_RULES = {
    "Haircare": {"hair", "shampoo", "conditioner", "scalp", "curl", "frizz", "argan", "brush"},
    "Skincare": {"skin", "face", "serum", "moisturizer", "cream", "cleanser", "acne", "spf", "toner"},
    "Makeup": {"lip", "lipstick", "mascara", "brow", "foundation", "makeup", "lashes", "eyeliner"},
    "Fragrance": {"perfume", "scent", "fragrance", "spray", "cologne", "smell"},
    "Body": {"body", "hand", "lotion", "deodorant", "scrub", "soap", "bath"},
    "Tools": {"brush", "roller", "tool", "mirror", "clipper", "comb", "dryer"},
}


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_image(images: object) -> str:
    if not isinstance(images, list) or not images:
        return ""
    first = images[0] if isinstance(images[0], dict) else {}
    return first.get("medium_image_url") or first.get("large_image_url") or first.get("small_image_url") or ""


def tokens(text: str) -> list[str]:
    out = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", text.lower()):
        if token not in STOPWORDS:
            out.append(token)
    return out


def infer_category(text: str) -> str:
    word_set = set(tokens(text))
    scores = {category: len(word_set & terms) for category, terms in CATEGORY_RULES.items()}
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score > 0 else "Beauty"


def derived_name(asin: str, text: str) -> str:
    top_terms = [term for term, _ in Counter(tokens(text)).most_common(3)]
    if top_terms:
        label = " ".join(term.title() for term in top_terms)
        return f"{label} Beauty Item"
    return f"Beauty Item {asin[-6:]}"


def read_reviews(path: Path, max_rows: int | None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows is not None and idx >= max_rows:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("parent_asin") or record.get("asin")
            user_id = record.get("user_id")
            rating = record.get("rating")
            if not asin or not user_id or rating is None:
                continue
            rows.append({
                "user_id": str(user_id),
                "product_id": str(asin),
                "rating": float(rating),
                "review_title": clean_text(record.get("title")),
                "review_text": clean_text(record.get("text")),
                "timestamp": record.get("timestamp", 0),
                "helpful_vote": int(record.get("helpful_vote") or 0),
                "verified_purchase": bool(record.get("verified_purchase")),
                "image": first_image(record.get("images")),
            })
    return pd.DataFrame(rows)


def build_artifacts(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    reviews = read_reviews(input_path, args.max_rows)
    if reviews.empty:
        raise ValueError("No valid reviews found.")

    product_counts = reviews["product_id"].value_counts()
    selected_products = product_counts[product_counts >= args.min_product_reviews].head(args.max_products).index
    reviews = reviews[reviews["product_id"].isin(selected_products)].copy()

    user_counts = reviews["user_id"].value_counts()
    selected_users = user_counts[user_counts >= args.min_user_reviews].head(args.max_users).index
    interactions = reviews[reviews["user_id"].isin(selected_users)].copy()

    if interactions.empty:
        raise ValueError("No interactions survived filtering. Lower min thresholds.")

    grouped = reviews.groupby("product_id")
    products = grouped.agg(
        avg_rating=("rating", "mean"),
        review_count=("rating", "size"),
        helpful_votes=("helpful_vote", "sum"),
        verified_rate=("verified_purchase", "mean"),
        all_reviews_text=("review_text", lambda s: " ".join(s.dropna().astype(str).head(args.reviews_per_product))),
        title_text=("review_title", lambda s: " ".join(s.dropna().astype(str).head(args.reviews_per_product))),
        image=("image", lambda s: next((x for x in s.astype(str) if x), "")),
    ).reset_index()
    products["content_text"] = (products["title_text"] + " " + products["all_reviews_text"]).str.strip()
    products["display_name"] = products.apply(lambda row: derived_name(row["product_id"], row["content_text"]), axis=1)
    products["category"] = products["content_text"].map(infer_category)
    products["search_url"] = products.apply(
        lambda row: f"https://www.amazon.in/s?k={row['product_id']}+{row['category']}",
        axis=1,
    )
    products = products.sort_values(["review_count", "avg_rating"], ascending=False).reset_index(drop=True)

    product_ids = set(products["product_id"])
    interactions = interactions[interactions["product_id"].isin(product_ids)].copy()
    cf_interactions = (
        interactions.groupby(["user_id", "product_id"], as_index=False)
        .agg(rating=("rating", "mean"), timestamp=("timestamp", "max"))
    )

    user_item_matrix = cf_interactions.pivot_table(
        index="product_id",
        columns="user_id",
        values="rating",
        fill_value=0,
    )
    user_item_sparse = csr_matrix(user_item_matrix.values)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=2,
        ngram_range=(1, 2),
    )
    product_tfidf = tfidf.fit_transform(products["content_text"].fillna(""))

    content_nn = NearestNeighbors(metric="cosine", algorithm="brute")
    content_nn.fit(product_tfidf)

    cf_model = NearestNeighbors(metric="cosine", algorithm="brute")
    cf_model.fit(user_item_sparse)

    demo_users = (
        cf_interactions.groupby("user_id")
        .agg(
            interactions=("product_id", "size"),
            avg_rating=("rating", "mean"),
            latest_timestamp=("timestamp", "max"),
        )
        .sort_values(["interactions", "avg_rating"], ascending=False)
        .head(8)
        .reset_index()
    )
    demo_users["display_name"] = [f"Real Review User {i + 1}" for i in range(len(demo_users))]

    products.to_csv(ARTIFACT_DIR / "product_df.csv", index=False)
    cf_interactions.to_csv(ARTIFACT_DIR / "cf_interactions.csv", index=False)
    user_item_matrix.to_csv(ARTIFACT_DIR / "user_item_matrix.csv")
    demo_users.to_csv(ARTIFACT_DIR / "demo_users.csv", index=False)
    save_npz(ARTIFACT_DIR / "user_item_sparse.npz", user_item_sparse)
    joblib.dump(tfidf, ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(content_nn, ARTIFACT_DIR / "content_nn_model.pkl")
    joblib.dump(cf_model, ARTIFACT_DIR / "cf_model.pkl")

    metadata = {
        "source": str(input_path),
        "review_rows_loaded": int(len(reviews)),
        "products": int(len(products)),
        "users": int(cf_interactions["user_id"].nunique()),
        "interactions": int(len(cf_interactions)),
        "max_rows": args.max_rows,
        "min_product_reviews": args.min_product_reviews,
        "min_user_reviews": args.min_user_reviews,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train real Amazon review recommender artifacts.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-products", type=int, default=1500)
    parser.add_argument("--max-users", type=int, default=1200)
    parser.add_argument("--min-product-reviews", type=int, default=8)
    parser.add_argument("--min-user-reviews", type=int, default=3)
    parser.add_argument("--reviews-per-product", type=int, default=80)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


if __name__ == "__main__":
    build_artifacts(parse_args())
