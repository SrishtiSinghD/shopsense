# ShopSense AI Hybrid Recommender

ShopSense is an end-to-end recommender application trained from the real Amazon Beauty review JSONL file included in this project. It no longer depends on broken live ASIN pages, missing Gemini keys, or notebook-only state.

## Current Training Run

The generated artifacts are in `artifacts/`.

```text
Source file: All_Beauty.jsonl
Products:    1500
Users:       664
Interactions:1542
```

The review dataset does not include reliable current product title, price, stock, or product-page availability. Product names are therefore derived from real review text, and outbound links use Amazon search URLs instead of brittle `/dp/{asin}` links.

## What The App Demonstrates

- Natural-language shopping queries.
- Agent routing without an external API key.
- Guardrails for unsafe medical/guaranteed-treatment claims.
- Query slot extraction.
- Hybrid ranking from trained artifacts:
  - TF-IDF review-text content model.
  - Product-product nearest neighbors.
  - Collaborative nearest-neighbor model from real user-item ratings.
  - Query relevance.
  - Popularity from rating, review count, helpful votes, and verified purchase rate.
- Cold-start fallback for unknown users.
- Explainable score breakdown for every recommendation.

## Train Or Retrain

```powershell
python train_models.py
```

Useful options:

```powershell
python train_models.py --max-products 2000 --max-users 1500 --min-product-reviews 6 --min-user-reviews 3
```

Generated files:

```text
artifacts/
├── product_df.csv
├── cf_interactions.csv
├── user_item_matrix.csv
├── user_item_sparse.npz
├── tfidf_vectorizer.pkl
├── content_nn_model.pkl
├── cf_model.pkl
├── demo_users.csv
└── metadata.json
```

## Run Locally

```powershell
pip install -r requirements.txt
uvicorn main:app --reload
```

In another terminal:

```powershell
streamlit run app.py
```

Open:

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

## Run With Docker

Stop any previous container first:

```powershell
docker compose down
```

Optional Gemini setup:

```powershell
copy .env.example .env
notepad .env
```

Put your key in `.env`:

```text
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-flash-latest
```

Then rebuild and run:

```powershell
docker compose up --build
```

Open http://localhost:8501.

If you do not set `GEMINI_API_KEY`, the app still works using the local fallback router. The UI sidebar will show whether Gemini is enabled.

## Demo Queries

- `hydrating skincare for sensitive skin`
- `show me frizz control hair products`
- `premium fragrance gift`
- `similar alternatives to B00YQ6X8EO`
- `this moisturizer is amazing but the scent is too strong, sentiment`
- `can this cure acne permanently?`

## API Endpoints

- `GET /health`
- `GET /catalog`
- `GET /demo-users`
- `POST /agent/query`
- `POST /recommend/user`
- `POST /recommend/product`
- `POST /sentiment`

Example:

```json
{
  "user_id": "guest",
  "query": "hydrating skincare for sensitive skin",
  "top_n": 5
}
```

## Project Structure

```text
.
├── app.py                 Streamlit frontend
├── main.py                FastAPI backend
├── gemini_agent.py        Local agent/router, no Gemini key needed
├── recommender.py         Loads trained artifacts and ranks products
├── train_models.py        Trains artifacts from All_Beauty.jsonl
├── sentiment.py           Offline sentiment fallback
├── artifacts/             Generated trained model/data files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── ARCHITECTURE.md
```
