from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "best", "for", "from", "give", "i",
    "in", "is", "me", "my", "need", "of", "on", "or", "product", "products",
    "show", "that", "the", "to", "under", "want", "with",
}


def _require_artifacts() -> None:
    required = [
        "product_df.csv",
        "cf_interactions.csv",
        "user_item_matrix.csv",
        "user_item_sparse.npz",
        "tfidf_vectorizer.pkl",
        "content_nn_model.pkl",
        "cf_model.pkl",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing trained artifacts: {missing_list}. Run `python train_models.py` first."
        )


_require_artifacts()

tfidf = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.pkl")
content_nn = joblib.load(ARTIFACT_DIR / "content_nn_model.pkl")
cf_model = joblib.load(ARTIFACT_DIR / "cf_model.pkl")
product_df = pd.read_csv(ARTIFACT_DIR / "product_df.csv")
cf_df = pd.read_csv(ARTIFACT_DIR / "cf_interactions.csv")
user_item_matrix = pd.read_csv(ARTIFACT_DIR / "user_item_matrix.csv", index_col=0)
user_item_sparse = load_npz(ARTIFACT_DIR / "user_item_sparse.npz")

if (ARTIFACT_DIR / "demo_users.csv").exists():
    demo_user_df = pd.read_csv(ARTIFACT_DIR / "demo_users.csv")
else:
    demo_user_df = pd.DataFrame(columns=["user_id", "display_name", "interactions", "avg_rating"])

metadata = {}
if (ARTIFACT_DIR / "metadata.json").exists():
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))

product_df["product_id"] = product_df["product_id"].astype(str)
product_df["content_text"] = product_df["content_text"].fillna("")
product_df["display_name"] = product_df["display_name"].fillna(product_df["product_id"])
product_df["category"] = product_df["category"].fillna("Beauty")
product_df["image"] = product_df["image"].fillna("")
product_df["search_url"] = product_df["search_url"].fillna("")
for col in ["avg_rating", "review_count", "helpful_votes", "verified_rate"]:
    product_df[col] = pd.to_numeric(product_df[col], errors="coerce").fillna(0)

cf_df["user_id"] = cf_df["user_id"].astype(str)
cf_df["product_id"] = cf_df["product_id"].astype(str)
cf_df["rating"] = pd.to_numeric(cf_df["rating"], errors="coerce").fillna(0)
user_item_matrix.index = user_item_matrix.index.astype(str)
user_item_matrix.columns = user_item_matrix.columns.astype(str)

PRODUCT_TFIDF = tfidf.transform(product_df["content_text"])
PRODUCT_META = product_df[[
    "product_id",
    "display_name",
    "category",
    "avg_rating",
    "review_count",
    "helpful_votes",
    "verified_rate",
    "image",
    "search_url",
    "content_text",
]].copy()
PRODUCT_INDEX = {pid: idx for idx, pid in enumerate(product_df["product_id"].tolist())}


def artifact_summary() -> dict[str, Any]:
    return {
        "artifacts_dir": str(ARTIFACT_DIR),
        "products": int(len(product_df)),
        "users": int(cf_df["user_id"].nunique()),
        "interactions": int(len(cf_df)),
        "metadata": metadata,
    }


def catalog() -> list[dict[str, Any]]:
    return [
        _public_product(row)
        for _, row in product_df.sort_values(["review_count", "avg_rating"], ascending=False).head(300).iterrows()
    ]


def demo_users() -> list[dict[str, Any]]:
    if demo_user_df.empty:
        users = cf_df["user_id"].value_counts().head(8).reset_index()
        users.columns = ["user_id", "interactions"]
        users["display_name"] = [f"Real Review User {i + 1}" for i in range(len(users))]
        users["avg_rating"] = 0
    else:
        users = demo_user_df.copy()

    return [
        {
            "user_id": str(row["user_id"]),
            "name": str(row.get("display_name", row["user_id"])),
            "segment": "Real Amazon review-history user",
            "preferences": _user_keywords(str(row["user_id"])),
            "budget": None,
            "interactions": int(row.get("interactions", 0)),
            "avg_rating": round(float(row.get("avg_rating", 0)), 2),
        }
        for _, row in users.iterrows()
    ]


def parse_query(query: str) -> dict[str, Any]:
    text = (query or "").lower()
    CATEGORY_MAP = {
        "Makeup": [
            "makeup", "lipstick", "mascara", "eyeliner",
            "foundation", "concealer", "blush",
            "palette", "lashes", "brows"
        ],
        "Haircare": [
            "hair", "shampoo", "conditioner",
            "curl", "frizz", "scalp"
        ],
        "Skincare": [
            "skin", "serum", "moisturizer",
            "cleanser", "acne", "face"
        ],
        "Fragrance": [
            "perfume", "fragrance",
            "cologne", "scent"
        ],
        "Body": [
            "body", "lotion",
            "soap", "scrub"
        ]
    }

    product_match = re.search(r"\b[A-Z0-9]{8,12}\b", query or "")

    categories = []
    for category, words in CATEGORY_MAP.items():
        if any(word in text for word in words):
            categories.append(category)

    keywords = sorted(
        {
            t
            for t in re.findall(r"[a-z0-9]+", text)
            if t not in STOPWORDS and len(t) > 1
        }
    )

    blocked_terms = [
        term
        for term in [
            "bleach",
            "medical cure",
            "prescription",
            "guaranteed cure"
        ]
        if term in text
    ]

    if any(
        word in text
        for word in [
            "diagnose",
            "cure",
            "treat disease",
            "permanently cure"
        ]
    ):
        blocked_terms.append("medical claim")

    return {
        "intent": "recommendation",
        "keywords": keywords,
        "categories": categories,
        "product_id": product_match.group(0) if product_match else None,
        "blocked_terms": sorted(set(blocked_terms)),
        "price_data_available": False,
    }


def guardrail_check(query: str) -> dict[str, Any]:
    parsed = parse_query(query)
    risky = bool(parsed["blocked_terms"])
    return {
        "allowed": not risky,
        "message": (
            "I can recommend beauty products from review data, but I cannot provide medical, diagnostic, or guaranteed treatment advice."
            if risky else ""
        ),
        "policy": "Beauty-commerce guardrail",
    }


def hybrid_user(user_id: str, query: str = "", top_n: int = 5) -> dict[str, Any]:
    user_id = str(user_id or "guest")
    top_n = int(max(1, min(top_n, 12)))
    guardrail = guardrail_check(query)
    if not guardrail["allowed"]:
        return {"results": [], "guardrail": guardrail, "stage": "blocked"}

    parsed = parse_query(query)
    user_history = cf_df[cf_df["user_id"] == user_id].copy()
    liked = user_history[user_history["rating"] >= 4]["product_id"].tolist()
    if not liked:
        liked = user_history.sort_values("rating", ascending=False).head(8)["product_id"].tolist()

    scores = PRODUCT_META.copy()
    scores = scores.merge(_query_scores(query), on="product_id", how="left")
    scores = scores.merge(_content_profile(liked), on="product_id", how="left")
    scores = scores.merge(_collab_scores(user_id, liked), on="product_id", how="left")
    scores = scores.merge(_popular_scores(), on="product_id", how="left")
    scores[["query_score", "content_score", "cf_score", "popular_score"]] = scores[["query_score", "content_score", "cf_score", "popular_score"]].fillna(0)

    if parsed["categories"]:
        scores = scores[scores["category"].isin(parsed["categories"])].copy()
        scores["category_boost"] = 1.0
    else:
        scores["category_boost"] = 0.0

    scores["seen_penalty"] = (scores["product_id"].isin(set(user_history["product_id"])).astype(float))

    if user_history.empty:
        scores["final_score"] = 0.52 * scores["query_score"] + 0.42 * scores["popular_score"] + 0.06 * scores["category_boost"]
        stage = "cold_start"
        reason = "Cold-start ranking on real Amazon review data: query relevance plus product popularity."
    else:
        scores["final_score"] = (
                0.50 * scores["query_score"]
                + 0.20 * scores["content_score"]
                + 0.15 * scores["cf_score"]
                + 0.10 * scores["popular_score"]
                + 0.05 * scores["category_boost"]
                - 0.10 * scores["seen_penalty"]
        )
        stage = "personalized"
        reason = "Hybrid ranking from trained artifacts: review-text content, collaborative neighbors, query match, popularity, and seen-item filtering."

    if scores.empty or float(scores["final_score"].max()) <= 0:
        scores = PRODUCT_META.merge(_popular_scores(), on="product_id", how="left")
        scores["query_score"] = scores["content_score"] = scores["cf_score"] = 0.0
        scores["final_score"] = scores["popular_score"]
        stage = "fallback"
        reason = "Fallback to trusted high-rating/high-review products from the trained product table."

    response = _format(scores.sort_values(["final_score", "avg_rating", "review_count"], ascending=False), top_n, reason, stage)
    response["agent"] = {"parsed_query": parsed, "guardrail": guardrail}
    response["user_profile"] = _profile(user_id)
    return response


def hybrid_product(product_id: str, query: str = "", top_n: int = 5) -> dict[str, Any]:
    product_id = str(product_id or "").strip()
    top_n = int(max(1, min(top_n, 12)))
    if product_id not in PRODUCT_INDEX:
        return hybrid_user("guest", query or product_id, top_n)

    c_df = _content_similar(product_id, top_n=40)
    cf_part = _cf_similar_product(product_id, top_n=40)
    scores = c_df.merge(cf_part, on="product_id", how="outer")
    scores = scores.merge(PRODUCT_META, on="product_id", how="left")
    scores = scores.merge(_query_scores(query), on="product_id", how="left")
    scores = scores.merge(_popular_scores(), on="product_id", how="left")
    scores[["content_score", "cf_score", "query_score", "popular_score"]] = scores[["content_score", "cf_score", "query_score", "popular_score"]].fillna(0)
    scores["final_score"] = 0.50 * scores["content_score"] + 0.28 * scores["cf_score"] + 0.12 * scores["query_score"] + 0.10 * scores["popular_score"]
    scores = scores[scores["product_id"] != product_id].sort_values("final_score", ascending=False)
    return _format(scores, top_n, "Similar-product ranking from trained content and collaborative artifacts.", "similarity")


def _query_scores(query: str) -> pd.DataFrame:
    if not (query or "").strip():
        return product_df[["product_id"]].assign(query_score=0.0)
    expanded_query = query.lower()

    QUERY_EXPANSION = {
        "lipstick": "makeup lipstick lip color cosmetic",
        "makeup": "foundation mascara eyeliner blush cosmetic",
        "hair": "haircare shampoo conditioner scalp",
        "shampoo": "haircare shampoo conditioner",
        "skin": "skincare serum moisturizer cleanser",
        "moisturizer": "hydrating skincare cream serum",
        "perfume": "fragrance scent cologne"
    }

    for key, value in QUERY_EXPANSION.items():
        if key in expanded_query:
            expanded_query += " " + value

    q_vec = tfidf.transform([expanded_query])

    sims = cosine_similarity(
        q_vec,
        PRODUCT_TFIDF
    ).ravel()

    return pd.DataFrame(
        {
            "product_id": product_df["product_id"],
            "query_score": sims,
        }
    )


def _content_profile(product_ids: list[str]) -> pd.DataFrame:
    valid = [pid for pid in product_ids if pid in PRODUCT_INDEX]
    if not valid:
        return product_df[["product_id"]].assign(content_score=0.0)
    indices = [PRODUCT_INDEX[pid] for pid in valid]
    profile = PRODUCT_TFIDF[indices].mean(axis=0)
    sims = cosine_similarity(np.asarray(profile), PRODUCT_TFIDF).ravel()
    return pd.DataFrame({"product_id": product_df["product_id"], "content_score": sims})


def _content_similar(product_id: str, top_n: int) -> pd.DataFrame:
    idx = PRODUCT_INDEX[product_id]
    distances, indices = content_nn.kneighbors(
        PRODUCT_TFIDF[idx],
        n_neighbors=min(top_n + 1, len(product_df)),
    )
    rows = []
    for dist, i in zip(distances[0], indices[0]):
        pid = str(product_df.iloc[i]["product_id"])
        if pid != product_id:
            rows.append({"product_id": pid, "content_score": float(1 - dist)})
    return pd.DataFrame(rows)


def _cf_similar_product(product_id: str, top_n: int) -> pd.DataFrame:
    if product_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["product_id", "cf_score"])
    idx = user_item_matrix.index.get_loc(product_id)
    distances, indices = cf_model.kneighbors(
        user_item_sparse[idx],
        n_neighbors=min(top_n + 1, user_item_sparse.shape[0]),
    )
    rows = []
    for dist, i in zip(distances[0], indices[0]):
        pid = str(user_item_matrix.index[i])
        if pid != product_id:
            rows.append({"product_id": pid, "cf_score": float(1 - dist)})
    return pd.DataFrame(rows)


def _collab_scores(user_id: str, liked_ids: list[str]) -> pd.DataFrame:
    scores = pd.Series(0.0, index=product_df["product_id"].astype(str))
    for pid in liked_ids:
        sim_df = _cf_similar_product(pid, top_n=30)
        if not sim_df.empty:
            scores = scores.add(sim_df.set_index("product_id")["cf_score"], fill_value=0)

    if user_id in user_item_matrix.columns:
        user_vector = user_item_matrix[[user_id]].T
        all_users = user_item_matrix.T
        sims = cosine_similarity(user_vector, all_users).ravel()
        neighbors = pd.Series(sims, index=all_users.index).sort_values(ascending=False).head(8)
        weighted = all_users.loc[neighbors.index].mul(neighbors, axis=0).sum(axis=0)
        scores = scores.add(weighted / max(float(neighbors.sum()), 1e-9), fill_value=0)

    return scores.rename("cf_score").reset_index().rename(columns={"index": "product_id"})


def _popular_scores() -> pd.DataFrame:
    pop = PRODUCT_META[["product_id", "avg_rating", "review_count", "helpful_votes", "verified_rate"]].copy()
    pop["popular_score"] = (
        0.42 * _minmax(pop["avg_rating"])
        + 0.32 * _minmax(pop["review_count"])
        + 0.16 * _minmax(pop["helpful_votes"])
        + 0.10 * _minmax(pop["verified_rate"])
    )
    return pop[["product_id", "popular_score"]]


def _minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = float(series.min()), float(series.max())
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def _format(df: pd.DataFrame, top_n: int, reason: str, stage: str) -> dict[str, Any]:
    results = []
    for _, row in df.head(top_n).iterrows():
        item = _public_product(row)
        item.update({
            "score": round(float(row.get("final_score", 0)), 4),
            "content_score": round(float(row.get("content_score", 0)), 4),
            "cf_score": round(float(row.get("cf_score", 0)), 4),
            "query_score": round(float(row.get("query_score", 0)), 4),
            "popular_score": round(float(row.get("popular_score", 0)), 4),
            "reason": reason,
        })
        results.append(item)
    return {
        "results": results,
        "stage": stage,
        "reason": reason,
        "model": "trained TF-IDF content model + trained collaborative nearest-neighbor model",
        "artifact_summary": artifact_summary(),
    }


def _public_product(row: pd.Series) -> dict[str, Any]:
    image = str(row.get("image", "") or "").strip()
    valid_image = (
        image.lower().startswith(("http://", "https://"))
        and "..." not in image
        and len(image) > 15
    )
    if not valid_image:
        image = "https://placehold.co/600x600?text=No+Image"
    text = str(row.get("content_text", "") or "")
    return {
        "product_id": str(row.get("product_id", "")),
        "name": str(row.get("display_name", row.get("product_id", ""))),
        "category": str(row.get("category", "Beauty")),
        "avg_rating": round(float(row.get("avg_rating", 0)), 2),
        "review_count": int(float(row.get("review_count", 0))),
        "helpful_votes": int(float(row.get("helpful_votes", 0))),
        "verified_rate": round(float(row.get("verified_rate", 0)), 2),
        "image": image,
        "search_url":  f"https://www.amazon.com/dp/{row.get('product_id','')}",
        "description": text[:220],
    }


def _profile(user_id: str) -> dict[str, Any]:
    history = cf_df[cf_df["user_id"] == user_id]
    return {
        "user_id": user_id,
        "known_user": bool(not history.empty),
        "history_count": int(len(history)),
        "avg_rating": round(float(history["rating"].mean()), 2) if not history.empty else 0,
        "top_keywords": _user_keywords(user_id),
    }


def _user_keywords(user_id: str) -> list[str]:
    history = cf_df[(cf_df["user_id"] == user_id) & (cf_df["rating"] >= 4)]
    if history.empty:
        return []
    text = " ".join(PRODUCT_META[PRODUCT_META["product_id"].isin(history["product_id"])]["content_text"].head(8))
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z]{3,}", text.lower()) if w not in STOPWORDS]
    return [word for word, _ in pd.Series(words).value_counts().head(6).items()]