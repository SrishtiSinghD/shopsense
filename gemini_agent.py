from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from recommender import guardrail_check, hybrid_product, hybrid_user, parse_query
from sentiment import predict_sentiment


load_dotenv(Path(__file__).resolve().parent / ".env")

PRODUCT_ID_RE = re.compile(r"\b(?=[A-Z0-9]{10}\b)(?=[A-Z0-9]*\d)[A-Z0-9]+\b", re.IGNORECASE)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

_active_model: str | None = None
_last_gemini_state: dict[str, Any] = {"live": "unknown", "detail": "No requests yet"}


def _unique_models() -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for model in FALLBACK_MODELS:
        name = (model or "").strip()
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


def gemini_status() -> dict[str, Any]:
    return {
        "enabled": bool(os.getenv("GEMINI_API_KEY")),
        "model": _active_model or GEMINI_MODEL,
        "key_env": "GEMINI_API_KEY",
        "live": _last_gemini_state.get("live", "unknown"),
        "detail": _last_gemini_state.get("detail", ""),
    }


def _set_gemini_state(live: str, detail: str, model: str | None = None) -> None:
    global _active_model, _last_gemini_state
    if model:
        _active_model = model
    _last_gemini_state = {"live": live, "detail": detail}


def _generate_with_fallback(prompt: str) -> tuple[str | None, str, str | None]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _set_gemini_state("disabled", "GEMINI_API_KEY not set")
        return None, "GEMINI_API_KEY not set", None

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    ordered = _unique_models()
    if _active_model and _active_model in ordered:
        ordered = [_active_model] + [m for m in ordered if m != _active_model]

    errors: list[str] = []
    for model_name in ordered:
        try:
            model = genai.GenerativeModel(model_name)
            text = model.generate_content(prompt).text.strip()
            if text:
                _set_gemini_state("ok", f"Using {model_name}", model_name)
                return text, f"Gemini reply via {model_name}", model_name
            errors.append(f"{model_name}: empty response")
        except Exception as exc:
            err = str(exc)
            errors.append(f"{model_name}: {err[:160]}")
            if "429" in err or "quota" in err.lower():
                continue
            if "404" in err or "not found" in err.lower():
                continue

    detail = errors[-1] if errors else "All Gemini models failed"
    if any("429" in e or "quota" in e.lower() for e in errors):
        _set_gemini_state("quota", detail)
    else:
        _set_gemini_state("error", detail)
    return None, detail, None


def _local_intent(query: str) -> str:
    text = (query or "").lower()
    if any(term in text for term in ["sentiment", "review tone", "positive", "negative review"]):
        return "sentiment"
    if any(term in text for term in ["similar", "like this", "alternatives"]) or PRODUCT_ID_RE.search(text):
        return "product_similarity"
    return "user_recommendation"


def _local_reply(query: str, response: dict[str, Any]) -> str:
    guardrail = response.get("guardrail") or {}
    if guardrail.get("allowed") is False:
        return guardrail.get("message", "I can't help with that request.")

    sentiment = response.get("sentiment")
    if sentiment:
        label = sentiment["label"]
        confidence = sentiment.get("confidence", "n/a")
        if label.lower() == "positive":
            return (
                f"The review reads as **{label}** (confidence {confidence}). "
                "Customers seem happy overall — worth considering if that matches what you want."
            )
        if label.lower() == "negative":
            return (
                f"The review reads as **{label}** (confidence {confidence}). "
                "There may be concerns in the tone — check the details before buying."
            )
        return f"I read the review tone as **{label}** with confidence {confidence}."

    results = response.get("results") or []
    if not results:
        return (
            f"I searched for \"{query}\" but didn't find strong matches in the Beauty catalog. "
            "Try a broader phrase like *hydrating face cream* or *frizz control hair serum*."
        )

    lines = [f"For **\"{query}\"**, these stood out from real Amazon Beauty review data:\n"]
    for idx, item in enumerate(results[:5], start=1):
        name = item.get("name", item.get("product_id", "Product"))
        rating = item.get("avg_rating", "?")
        reviews = item.get("review_count", 0)
        category = item.get("category", "Beauty")
        lines.append(f"{idx}. **{name}** — {rating}★ from {reviews} reviews ({category})")

    stage = response.get("stage", "hybrid")
    if stage == "cold_start":
        lines.append("\nYou're browsing as a new visitor, so I weighted query fit and popularity.")
    elif stage == "personalized":
        lines.append("\nThese also reflect this shopper's past ratings and similar purchases.")

    return "\n".join(lines)


def _gemini_reply(query: str, response: dict[str, Any]) -> tuple[str | None, str]:
    guardrail = response.get("guardrail") or {}
    sentiment = response.get("sentiment")
    results = response.get("results") or []

    context = {
        "intent": response.get("intent"),
        "stage": response.get("stage"),
        "reason": response.get("reason"),
        "guardrail_allowed": guardrail.get("allowed", True),
        "guardrail_message": guardrail.get("message"),
        "sentiment": sentiment,
        "results": [
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "avg_rating": item.get("avg_rating"),
                "review_count": item.get("review_count"),
                "score": item.get("score"),
                "description": (item.get("description") or "")[:120],
            }
            for item in results[:5]
        ],
    }

    prompt = f"""
You are ShopSense, a friendly beauty shopping assistant backed by a hybrid recommender.
Write a natural conversational reply in 2-4 sentences tailored to THIS specific user query.
Reference product names and why they fit. Do not use a generic template.
Do not mention JSON, routing, models, or internal systems.

User query: {query}
Context: {json.dumps(context)}
"""
    text, detail, _ = _generate_with_fallback(prompt)
    if text:
        return text, detail
    return None, detail


def _attach_assistant_message(response: dict[str, Any], query: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    gemini_text, gemini_detail = _gemini_reply(query, response)
    if gemini_text:
        response["assistant_message"] = gemini_text
        response["reply_router"] = "gemini"
        trace.append({"step": "assistant_reply", "status": "gemini", "detail": gemini_detail})
    else:
        response["assistant_message"] = _local_reply(query, response)
        response["reply_router"] = "local_fallback"
        trace.append({"step": "assistant_reply", "status": "local_fallback", "detail": gemini_detail})
    response["agent_trace"] = trace
    response["gemini"] = gemini_status()
    return response


def process_query(query: str, user_id: str | None = None, top_n: int = 5) -> dict[str, Any]:
    query = query or ""
    guardrail = guardrail_check(query)
    parsed = parse_query(query)
    intent = _local_intent(query)

    trace = [
        {"step": "guardrail", "status": "passed" if guardrail["allowed"] else "blocked", "detail": guardrail["policy"]},
        {"step": "intent_router", "status": intent, "detail": f"local: matched query patterns"},
        {"step": "slot_extraction", "status": "complete", "detail": parsed},
    ]

    if not guardrail["allowed"]:
        response = {
            "results": [],
            "intent": intent,
            "agent_trace": trace,
            "guardrail": guardrail,
        }
        return _attach_assistant_message(response, query, trace)

    if intent == "sentiment":
        result = predict_sentiment(query)
        trace.append({"step": "sentiment_tool", "status": result["label"], "detail": result})
        response = {
            "intent": intent,
            "sentiment": result,
            "agent_trace": trace,
            "results": [],
        }
        return _attach_assistant_message(response, query, trace)

    if intent == "product_similarity":
        product_match = PRODUCT_ID_RE.search(query)
        product_id = product_match.group(0).upper() if product_match else query.strip().upper()
        response = hybrid_product(product_id=product_id, query=query, top_n=top_n)
        trace.append({"step": "hybrid_product_model", "status": response.get("stage"), "detail": response.get("reason")})
    else:
        response = hybrid_user(user_id=user_id or "guest", query=query, top_n=top_n)
        trace.append({"step": "hybrid_user_model", "status": response.get("stage"), "detail": response.get("reason")})

    response["intent"] = intent
    return _attach_assistant_message(response, query, trace)
