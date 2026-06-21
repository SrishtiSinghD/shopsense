from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemini_agent import process_query
from recommender import catalog, demo_users, hybrid_product, hybrid_user
from sentiment import predict_sentiment


app = FastAPI(
    title="ShopSense Hybrid Recommender API",
    description="Offline demo of an agentic hybrid recommendation system.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_id: str = "guest"
    query: str = ""
    top_n: int = Field(default=5, ge=1, le=12)


class ProductRequest(BaseModel):
    product_id: str
    query: str = ""
    top_n: int = Field(default=5, ge=1, le=12)


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "ShopSense Hybrid Recommender",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/catalog")
def get_catalog():
    return {"products": catalog()}


@app.get("/demo-users")
def get_demo_users():
    return {"users": demo_users()}


@app.post("/agent/query")
def agent_query(req: RecommendRequest):
    return process_query(req.query, user_id=req.user_id, top_n=req.top_n)


@app.post("/recommend/user")
def recommend_user(req: RecommendRequest):
    return hybrid_user(user_id=req.user_id, query=req.query, top_n=req.top_n)


@app.post("/recommend/product")
def recommend_product(req: ProductRequest):
    return hybrid_product(product_id=req.product_id, query=req.query, top_n=req.top_n)


@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    return predict_sentiment(req.text)
