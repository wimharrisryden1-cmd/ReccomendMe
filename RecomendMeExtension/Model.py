# Required packages (no scikit-surprise needed):
# pip install numpy pandas scikit-learn joblib openpyxl

import json
import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor

# Import our surprise-free SVD so the saved joblib can be loaded by app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Backend"))
from svd_model import SVDModel  # noqa: E402


# =========================
# CONFIG
# =========================
CSV_PATH = os.path.join(os.path.dirname(__file__), "data_recommendme.csv")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported_rating_model")


# =========================
# LOAD + CLEAN
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [str(c).strip().lower() for c in df.columns]

required_cols = ["requester_id", "recommender_id", "recomender_star_rating"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

if "requestor_notes" not in df.columns:
    df["requestor_notes"] = ""
if "requester_email" not in df.columns:
    df["requester_email"] = ""

df["requester_id"] = df["requester_id"].fillna("").astype(str).str.strip().str.upper()
df["recommender_id"] = df["recommender_id"].fillna("").astype(str).str.strip().str.upper()
df["requester_email"] = df["requester_email"].fillna("").astype(str).str.strip().str.lower()
df["requestor_notes"] = df["requestor_notes"].fillna("").astype(str).str.strip()
df["recomender_star_rating"] = pd.to_numeric(df["recomender_star_rating"], errors="coerce")

df = df[
    (df["requester_id"] != "") &
    (df["recommender_id"] != "") &
    (df["recomender_star_rating"].notna())
].copy()

if len(df) == 0:
    raise ValueError("No valid rows remain after cleaning.")


# =========================
# TRAIN SVD MODEL (no surprise dependency)
# =========================
ratings = list(zip(
    df["recommender_id"].tolist(),
    df["requester_id"].tolist(),
    df["recomender_star_rating"].tolist()
))

svd_model = SVDModel(n_factors=40, n_epochs=100, lr_all=0.005, reg_all=0.15, random_state=42)
svd_model.fit(ratings)
print("SVD training complete.")


# =========================
# FALLBACK TABLES
# =========================
global_mean = float(df["recomender_star_rating"].mean())
requester_avg = df.groupby("requester_id")["recomender_star_rating"].mean().to_dict()
recommender_avg = df.groupby("recommender_id")["recomender_star_rating"].mean().to_dict()


# =========================
# EMAIL -> ID MAPS
# =========================
requester_email_to_id = {}
email_rows = df.loc[df["requester_email"] != "", ["requester_email", "requester_id"]].drop_duplicates()
for _, row in email_rows.iterrows():
    email = row["requester_email"]
    uid = row["requester_id"]
    if email not in requester_email_to_id:
        requester_email_to_id[email] = uid


def make_dummy_email(uid: str, prefix: str) -> str:
    uid = str(uid).strip().lower() or "unknown"
    return f"{prefix}_{uid}@demo.edu"


recommender_email_to_id = {
    make_dummy_email(uid, "rec"): uid for uid in recommender_avg.keys()
}


# =========================
# COLD-START TEXT MODEL
# =========================
tfidf = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
X_cold = tfidf.fit_transform(df["requestor_notes"]).toarray()
y_cold = df["recomender_star_rating"].values

cold_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
cold_model.fit(X_cold, y_cold)
print("Cold-start model training complete.")


# =========================
# EXPORT
# =========================
os.makedirs(EXPORT_DIR, exist_ok=True)

joblib.dump(svd_model, os.path.join(EXPORT_DIR, "svd_model.joblib"))
joblib.dump(tfidf, os.path.join(EXPORT_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(cold_model, os.path.join(EXPORT_DIR, "cold_model.joblib"))

with open(os.path.join(EXPORT_DIR, "requester_avg.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): float(v) for k, v in requester_avg.items()}, f, indent=2)

with open(os.path.join(EXPORT_DIR, "recommender_avg.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): float(v) for k, v in recommender_avg.items()}, f, indent=2)

with open(os.path.join(EXPORT_DIR, "global_mean.json"), "w", encoding="utf-8") as f:
    json.dump({"global_mean": global_mean}, f, indent=2)

with open(os.path.join(EXPORT_DIR, "requester_email_to_id.json"), "w", encoding="utf-8") as f:
    json.dump(requester_email_to_id, f, indent=2)

with open(os.path.join(EXPORT_DIR, "recommender_email_to_id.json"), "w", encoding="utf-8") as f:
    json.dump(recommender_email_to_id, f, indent=2)

print(f"Export complete → {EXPORT_DIR}")


# =========================
# LOCAL TEST
# =========================
def normalize_requester(value: str) -> str:
    value = str(value).strip()
    if value.upper().startswith("U"):
        return value.upper()
    return requester_email_to_id.get(value.lower(), value)


def normalize_recommender(value: str) -> str:
    value = str(value).strip()
    if value.upper().startswith("U"):
        return value.upper()
    return recommender_email_to_id.get(value.lower(), value)


def score_pair_svd(requester_id, recommender_id, requestor_note=""):
    requester_id = normalize_requester(requester_id)
    recommender_id = normalize_recommender(recommender_id)

    requester_known = requester_id in requester_avg
    recommender_known = recommender_id in recommender_avg

    if requester_known and recommender_known:
        pred = svd_model.predict(uid=recommender_id, iid=requester_id).est
        case = "svd"
    elif recommender_known:
        pred = recommender_avg[recommender_id]
        case = "new_requester"
    elif requester_known:
        pred = requester_avg[requester_id]
        case = "new_recommender"
    else:
        x = tfidf.transform([str(requestor_note)]).toarray()
        pred = cold_model.predict(x)[0]
        case = "cold_start"

    pred = float(np.clip(pred, 1, 5))
    return {"predicted_rating": round(pred, 2), "predicted_star_rating": int(round(pred)), "case": case}


if __name__ == "__main__":
    print("Example:", score_pair_svd("wimharrisryden1@gmail.com", "U0023", "hiii"))
