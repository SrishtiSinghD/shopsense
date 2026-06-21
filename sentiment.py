from __future__ import annotations


POSITIVE = {
    "amazing", "best", "bright", "calm", "comfortable", "excellent", "gentle",
    "glow", "good", "great", "hydrating", "love", "perfect", "recommend",
    "smooth", "soft", "soothing", "works",
}

NEGATIVE = {
    "bad", "breakout", "burn", "cheap", "drying", "greasy", "hate", "irritated",
    "itchy", "overpriced", "poor", "sticky", "strong", "waste", "worst",
}


def predict_sentiment(text: str):
    words = {w.strip(".,!?;:()[]{}").lower() for w in (text or "").split()}
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    total = max(pos + neg, 1)

    if pos > neg:
        label = "positive"
    elif neg > pos:
        label = "negative"
    else:
        label = "neutral"

    confidence = 0.52 + min(abs(pos - neg) / total, 1) * 0.43
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "probs": {
            "negative": round(neg / total, 3),
            "neutral": round(1 / (total + 1), 3),
            "positive": round(pos / total, 3),
        },
        "method": "lexicon fallback for offline demo",
    }
