# ABC Ltd — Customer Retention Predictor (Streamlit)

Two models trained on load and cached:
- **Logistic regression** — churn risk (Yes/No probability)
- **Linear regression** — predicted total billing (customer value)

## Files
- `app.py` — the Streamlit app
- `Telco-Customer-Churn.csv` — training data (must sit next to app.py)
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (GitHub + Streamlit Community Cloud) — see chat for full steps
1. Push this folder to a new GitHub repo.
2. Go to share.streamlit.io → New app → pick the repo, branch `main`, file `app.py`.
3. Deploy. Your app gets a public `*.streamlit.app` URL.
