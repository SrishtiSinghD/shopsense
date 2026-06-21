from __future__ import annotations

import html
import os
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from gemini_agent import gemini_status, process_query
from recommender import artifact_summary, catalog, demo_users


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ShopSense AI Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --purple: #7F2AFF;
  --purple-deep: #5B12C9;
  --plum: #2A1652;
  --lilac: #F7F2FF;
  --gold: #B8862E;
  --gold-soft: #F1E2C2;
  --ink: #111827;
  --muted: #5B6478;
}
html, body, .stApp {
  background: linear-gradient(180deg, #FFFFFF 0%, #F7F2FF 100%) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}
.block-container {
  max-width: 1320px;
  padding-top: 1.2rem;
}
[data-testid="chatAvatarIcon-assistant"] {
    color: #7F2AFF !important;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid #ECE3FF;
    box-shadow: 0 2px 10px rgba(127,42,255,.05);
}
[data-testid="stSidebar"] {
  background: linear-gradient(
      180deg,
      #ffffff 0%,
      #f7f2ff 100%
  ) !important;
  border-right: 1px solid #ECE3FF;
}
[data-testid="stSidebar"] * {
  color: #111827 !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: 'Fraunces', serif !important;
  font-weight: 600 !important;
  color: var(--plum) !important;
}
.stButton > button {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  min-height: 42px;
  font-weight: 600;
  white-space: normal;
  text-align: left;
  transition: all .18s ease !important;
}
.stButton > button:hover {
  border-color: #7F2AFF !important;
  color: #7F2AFF !important;
  background: #FAF7FF !important;
  transform: translateX(2px);
}

/* ---------- Hero ---------- */
.hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #7F2AFF, #A855F7 60%, #C77DFF);
  border-radius: 28px;
  padding: 42px 44px;
  margin-bottom: 30px;
  box-shadow: 0 16px 44px rgba(127,42,255,.28);
}
.hero::before, .hero::after {
  content: "";
  position: absolute;
  border-radius: 50%;
  filter: blur(2px);
  background: radial-gradient(circle, rgba(255,255,255,.20), transparent 70%);
  animation: drift 14s ease-in-out infinite alternate;
}
.hero::before { width: 280px; height: 280px; top: -90px; right: 8%; }
.hero::after { width: 200px; height: 200px; bottom: -80px; right: 28%; animation-delay: -6s; }
@media (prefers-reduced-motion: reduce) {
  .hero::before, .hero::after { animation: none; }
}
@keyframes drift {
  from { transform: translateY(0) translateX(0); }
  to   { transform: translateY(18px) translateX(-14px); }
}
.hero-eyebrow {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: #F3E8FF;
  background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  margin-bottom: 14px;
}
.hero h1 {
  position: relative;
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-style: normal;
  font-size: 2.7rem;
  margin: 0 0 8px 0;
  letter-spacing: -.01em;
  color: #ffffff;
}
.hero p {
  position: relative;
  margin: 0;
  color: #ECE0FF;
  font-size: 1.02rem;
  max-width: 560px;
  line-height: 1.5;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.metric-box {
  background: #ffffff;
  border: 1px solid #d9e0ea;
  border-radius: 8px;
  padding: 14px 16px;
  min-height: 86px;
}
.metric-box span {
  color: #64748b;
  font-size: .84rem;
}
.metric-box strong {
  display: block;
  color: #111827;
  font-size: 1.22rem;
  margin-top: 6px;
  overflow-wrap: anywhere;
}

/* ---------- Product card ---------- */
@keyframes cardIn {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.product-card {
  background: white;
  border-radius: 20px;
  overflow: hidden;
  height: auto;
  min-height: 540px;
  display: flex;
  flex-direction: column;
  border: 1px solid #E4D8FB;
  box-shadow:
    0 2px 8px rgba(127,42,255,.06),
    0 10px 26px rgba(127,42,255,.08);
  margin-bottom: 18px;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation: cardIn .45s ease backwards;
  animation-delay: var(--d, 0s);
}
@media (prefers-reduced-motion: reduce) {
  .product-card { animation: none; }
}
.product-card:hover {
  transform: translateY(-6px);
  border-color: #D9C2FF;
  box-shadow:
    0 12px 24px rgba(127,42,255,.13),
    0 24px 48px rgba(127,42,255,.16);
}
.card-media {
  position: relative;
  height: 230px;
  width: 100%;
  flex-shrink: 0;
  background: radial-gradient(circle at 50% 40%, #FBF8FF 0%, #F3ECFF 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid #F1F5F9;
}
.card-media img {
  width: 190px;
  height: 190px;
  object-fit: contain;
  transition: transform .3s ease;
}
.product-card:hover .card-media img {
  transform: scale(1.06);
}
.card-media-fallback {
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  color: #B9A6E8;
}
.card-media-fallback span {
  font-size: 2.1rem;
  filter: grayscale(.3);
}
.card-media-fallback small {
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .03em;
  color: #A893D6;
}
.card-badge {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 2;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: linear-gradient(135deg, var(--gold-soft), #FBF1DD);
  color: var(--gold);
  border: 1px solid #E7CE99;
  border-radius: 999px;
  padding: 3px 11px;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.card-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  padding: 16px 18px 18px;
}
.product-name {
  font-family: 'Fraunces', serif;
  color: var(--plum);
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.32;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 66px;
  max-height: 66px;
  flex-shrink: 0;
}
.rating-row {
  color: var(--gold);
  font-size: .82rem;
  font-weight: 700;
  margin-top: 6px;
  flex-shrink: 0;
}
.rating-row .dot {
  color: #C9BEDD;
  margin: 0 2px;
  font-weight: 400;
}
.card-desc {
  color: var(--muted);
  font-size: .85rem;
  line-height: 1.45;
  margin-top: 7px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  height: 90px;
  flex: 1 1 auto;
}
.amazon-btn {
  display: block;
  text-align: center;
  margin-top: auto;
  background: #ffffff;
  color: var(--purple) !important;
  text-decoration: none;
  padding: 11px;
  border-radius: 12px;
  font-weight: 700;
  font-size: .92rem;
  border: 2px solid var(--purple);
  outline: none;
  transition: background .18s ease, color .18s ease, box-shadow .18s ease, transform .12s ease;
}
.amazon-btn:hover {
  background: linear-gradient(135deg, #7F2AFF, #A855F7);
  color: #ffffff !important;
  box-shadow: 0 6px 18px rgba(127,42,255,.30);
}
.amazon-btn:active {
  transform: scale(.97);
}
.amazon-btn:active, .amazon-btn:focus-visible {
  border-color: var(--purple);
  box-shadow: 0 0 0 4px rgba(127,42,255,.22), 0 0 22px 4px rgba(127,42,255,.55);
}
.pill {
  display: inline-block;
  color: #0f172a;
  background: #eef6f5;
  border: 1px solid #c7e3df;
  border-radius: 999px;
  padding: 4px 9px;
  margin: 7px 5px 0 0;
  font-size: .78rem;
  font-weight: 650;
}
.muted {
  color: #475569;
  font-size: .88rem;
  line-height: 1.35;
}
.scorebar {
  height: 8px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
  margin-top: 10px;
}
.scorebar div {
  height: 100%;
  background: linear-gradient(90deg, #0f766e, #f59e0b);
}
.trace-box {
  background: #ffffff;
  color: #111827;
  border: 1px solid #d9e0ea;
  border-left: 4px solid #0f766e;
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 10px;
  font-size: .9rem;
}
.trace-box code {
  color: #334155;
  white-space: pre-wrap;
}
@media (max-width: 950px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        res = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=8)
        res.raise_for_status()
        return res.json()
    except Exception:
        return process_query(payload.get("query", ""), payload.get("user_id", "guest"), payload.get("top_n", 5))


def render_product(product, rank: int | None = None, delay: float = 0.0) -> None:
    """Render one product as a single HTML block.

    Everything (image, title, rating, description, CTA) is built into one
    string and passed to a single st.markdown call. Streamlit gives every
    st.image / st.write / st.link_button its own top-level DOM node, so
    mixing those widgets between an opening and closing '<div>' markdown
    call never actually nests them inside that div — that's why the cards
    were rendering as empty purple boxes with the real content stacked
    underneath, outside the border. One markdown call = one real container.
    """
    name = html.escape(str(product.get("name") or "Unnamed product"))
    description = html.escape(str(product.get("description") or "")[:150])
    rating = html.escape(str(product.get("avg_rating", "—")))
    reviews = html.escape(str(product.get("review_count", "—")))
    image = product.get("image")
    amazon_url = html.escape(
        str(
            product.get(
                "search_url",
                f"https://www.amazon.com/dp/{product.get('product_id', '')}",
            )
        ),
        quote=True,
    )

    if image:
        img_src = html.escape(str(image), quote=True)
        media_html = (
            f'<img src="{img_src}" alt="" loading="lazy" '
            "onerror=\"this.style.display='none'; "
            "this.nextElementSibling.style.display='flex';\" />"
            '<div class="card-media-fallback" style="display:none;">'
            "<span>🧴</span><small>No image available</small></div>"
        )
    else:
        media_html = (
            '<div class="card-media-fallback" style="display:flex;">'
            "<span>🧴</span><small>No image available</small></div>"
        )

    badge_html = '<div class="card-badge">✨ Top match</div>' if rank == 0 else ""

    # NOTE: built as one continuous string with zero embedded newlines.
    # When badge_html was "" earlier, the multi-line template below it left
    # a whitespace-only line, which Markdown treats as a blank line. That
    # ended the raw-HTML block early, and the indented lines after it got
    # reinterpreted as an indented *code block* -- which is exactly the
    # literal "<div class=...>" text showing up on every card except the
    # one where the badge line wasn't blank. No newlines = no blank lines
    # = Markdown can't second-guess it.
    #
    # The badge also now lives inside .card-media (absolutely positioned
    # over the image) rather than inline at the top of .card-body. With it
    # inline, the one card that had a badge pushed its title down further
    # than every other card in the row, so titles in the badge-less columns
    # crowded up toward the row above -- that's the "overlapping" look.
    # Putting it in the image area means .card-body always starts in the
    # same place, badge or not.
    card_html = (
        f'<div class="product-card" style="--d:{delay:.2f}s">'
        f'<div class="card-media">{media_html}{badge_html}</div>'
        f'<div class="card-body">'
        f'<div class="product-name">{name}</div>'
        f'<div class="rating-row">★ {rating}<span class="dot">·</span>{reviews} reviews</div>'
        f'<p class="card-desc">{description}</p>'
        f'<a class="amazon-btn" href="{amazon_url}" target="_blank" rel="noopener noreferrer">View on Amazon →</a>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\breal\s+amazon[\w\s]*?(review\s+)?data(set)?\b",
        r"\bamazon\s+(beauty\s+)?(review\s+)?data(set)?\b",
        r"\b(scraped|crawled|pulled|sourced)\s+from\s+amazon\b",
        r"\bgemini(?:[\s-]?agent)?\b",
        r"\bllm\b",
        r"\bbackend\b",
        r"\bdataset\b",
        r"\bprocess_query\b",
    ]
]


def _scrub(text: str) -> str:
    """Safety net for any backend-written copy shown to the user.

    Shoppers shouldn't see implementation details — which dataset this
    came from, model/agent names, etc. This strips that vocabulary out of
    whatever text the backend hands back, regardless of how it's phrased.
    """
    cleaned = text
    for pattern in _LEAK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .,:;-")
    return cleaned or "Here's what I found."


def build_display_message(query: str, response: dict[str, Any]) -> str:
    """Turn a backend response into clean, user-facing chat copy.

    For guardrail refusals and sentiment reads, the backend's message is
    written for the end user already, so it just gets a light scrub. For
    normal recommendations, we don't trust the model's freeform sentence
    at all — it tends to narrate its own internals (data source, category
    tags) and re-list products the cards below already show. Instead we
    write one short sentence ourselves from the structured `results`,
    so the user-facing copy is fully controlled regardless of what the
    agent returns.
    """
    guardrail = response.get("guardrail") or {}
    if guardrail.get("allowed") is False:
        return _scrub(guardrail.get("message") or "I can't help with that request.")

    if response.get("sentiment"):
        return _scrub(
            response.get("assistant_message") or "Here's what that review sounds like."
        )

    results = response.get("results") or []
    if results:
        count = len(results)
        plural = "s" if count != 1 else ""
        return (
            f'Here {"are" if count != 1 else "is"} {count} pick{plural} for '
            f'"{query.strip()}", based on real shopper ratings and reviews.'
        )

    return _scrub(
        response.get("reason")
        or "I couldn't find a strong match for that — try rephrasing, or ask "
        "about a specific product or ingredient."
    )


def handle_query(
    query: str, user_id: str, top_n: int, display_query: str | None = None
) -> dict[str, Any]:
    """Run a query against the backend.

    `query` is what's actually sent to process_query — keep this as the
    real ASIN/text the backend expects to match on. `display_query` is
    what shows up in the chat thread; pass it in whenever the literal
    query text is something a shopper shouldn't see verbatim (like a
    raw ASIN) and you want a friendlier line shown instead.
    """
    shown_query = display_query or query
    with st.spinner("🤖 ShopSense is thinking..."):
        response = process_query(query, user_id=user_id, top_n=top_n)
    st.session_state.messages.append({"role": "user", "content": shown_query})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": build_display_message(shown_query, response),
            "response": response,
        }
    )
    st.session_state.last_response = response
    st.session_state.scroll_pending = True
    st.rerun()


def render_assistant_turn(message: dict[str, Any]) -> None:
    response = message.get("response")
    guardrail = (response or {}).get("guardrail") or {}

    if guardrail.get("allowed") is False:
        # Show the sanitized copy once, styled as a refusal — not the raw
        # backend text, and not twice (markdown + error box).
        st.error(message["content"])
        return

    st.markdown(message["content"])
    if not response:
        return

    if response.get("sentiment"):
        st.success(
            f"Sentiment: {response['sentiment']['label']} "
            f"({response['sentiment']['confidence']})"
        )
        return

    results = response.get("results") or []
    if not results:
        st.info("No recommendation results for this query.")
        return

    cols = st.columns(min(3, len(results)))
    for idx, product in enumerate(results):
        with cols[idx % len(cols)]:
            render_product(product, rank=idx, delay=min(idx * 0.06, 0.3))


users = demo_users()
products = catalog()


def _lookup_product_name(asin: str) -> str | None:
    """Best-effort ASIN -> product name lookup against the loaded catalog.

    Used so a demo button can show a real product name instead of a raw
    ASIN to the shopper, while the backend still gets the exact ASIN it
    expects for "find similar" matching. Falls back to None on any shape
    mismatch so a catalog format change can't crash the sidebar.
    """
    try:
        rows = products.to_dict("records") if hasattr(products, "to_dict") else products
        for row in rows:
            pid = row.get("product_id", row.get("asin"))
            if str(pid) == asin:
                name = str(row.get("name", "")).strip()
                return name or None
    except Exception:
        return None
    return None


def _short_label(name: str, limit: int = 48) -> str:
    short = name.split(" - ")[0].split(",")[0].strip()
    if len(short) > limit:
        short = short[: limit - 1].rstrip() + "…"
    return short


_similar_asin = "B00YQ6X8EO"
_similar_name = _lookup_product_name(_similar_asin)
_similar_label = (
    f"Show me alternatives to {_short_label(_similar_name)}"
    if _similar_name
    else "Show me alternatives to my last serum"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "scroll_pending" not in st.session_state:
    st.session_state.scroll_pending = False

with st.sidebar:
    st.header("Demo Controls")
    user_options = {u["name"]: u["user_id"] for u in users}
    user_options["New visitor (cold start)"] = "guest"
    selected_user_label = st.selectbox(
        "Persona", list(user_options.keys()), key="persona_select"
    )
    selected_user = user_options[selected_user_label]
    top_n = st.slider("Recommendations", 3, 8, 5)

    st.divider()
    st.subheader("Try These")
    examples = [
        ("💧", "Hydrating skincare for sensitive skin", None),
        ("🌬️", "Show me frizz control hair products", None),
        ("🌸", "Premium fragrance gift", None),
        ("🔁", _similar_label, f"Similar alternatives to {_similar_asin}"),
        ("💬", "This moisturizer is amazing but the scent is too strong, sentiment", None),
        ("🚫", "Can this cure acne permanently?", None),
    ]
    for i, (icon, label, real_query) in enumerate(examples):
        if st.button(f"{icon}  {label}", key=f"example_{i}", use_container_width=True):
            handle_query(real_query or label, selected_user, top_n, display_query=label)

st.markdown("""
    <section class="hero">
      <div class="hero-eyebrow">✨ AI Beauty Recommender</div>
      <h1>ShopSense AI Recommender</h1>
        <p>Discover beauty products through AI-powered recommendations, customer reviews, and personalized insights.</p>
    </section>
    """,
        unsafe_allow_html=True,
    )

left = st.container()

with left:
    st.subheader("Recommendation Workspace")

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I'm ShopSense. Ask for beauty products by need, category, review tone, or ASIN — "
                "for example: *hydrating skincare for sensitive skin*."
            )

    for msg in st.session_state.messages[-8:]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_turn(msg)
            else:
                st.write(msg["content"])

    prompt = st.chat_input("Ask for products by need, category, review tone, or ASIN")
    if prompt:
        handle_query(prompt, selected_user, top_n)

    if st.session_state.scroll_pending:
        components.html(
            """
            <script>
            const doc = window.parent.document;
            const container = doc.querySelector('section.main') || doc.body;
            container.scrollTo({top: container.scrollHeight, behavior: 'smooth'});
            </script>
            """,
            height=0,
        )
        st.session_state.scroll_pending = False