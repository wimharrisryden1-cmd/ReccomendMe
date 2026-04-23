import json
import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from svd_model import SVDModel  # noqa: F401 — required for joblib deserialization
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "exported_rating_model")
DATA_PATH = os.path.join(PROJECT_ROOT, "data_recommendme.csv")


# =========================
# LOAD MODEL ARTIFACTS
# =========================
svd_model = joblib.load(os.path.join(MODEL_DIR, "svd_model.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
cold_model = joblib.load(os.path.join(MODEL_DIR, "cold_model.joblib"))

with open(os.path.join(MODEL_DIR, "requester_avg.json"), "r", encoding="utf-8") as f:
    requester_avg = json.load(f)

with open(os.path.join(MODEL_DIR, "recommender_avg.json"), "r", encoding="utf-8") as f:
    recommender_avg = json.load(f)

with open(os.path.join(MODEL_DIR, "requester_email_to_id.json"), "r", encoding="utf-8") as f:
    requester_email_to_id = json.load(f)

with open(os.path.join(MODEL_DIR, "recommender_email_to_id.json"), "r", encoding="utf-8") as f:
    recommender_email_to_id = json.load(f)

global_mean_path = os.path.join(MODEL_DIR, "global_mean.json")
if os.path.exists(global_mean_path):
    with open(global_mean_path, "r", encoding="utf-8") as f:
        global_mean = float(json.load(f).get("global_mean", 3.0))
else:
    global_mean = 3.0


# =========================
# LOAD INTERACTION DATA
# =========================
graph_df = pd.read_csv(DATA_PATH)
graph_df.columns = [str(c).strip().lower() for c in graph_df.columns]

graph_df["requester_id"] = graph_df["requester_id"].fillna("").astype(str).str.strip().str.upper()
graph_df["recommender_id"] = graph_df["recommender_id"].fillna("").astype(str).str.strip().str.upper()
graph_df["recomender_star_rating"] = pd.to_numeric(graph_df["recomender_star_rating"], errors="coerce")

graph_df = graph_df[
    (graph_df["requester_id"] != "") &
    (graph_df["recommender_id"] != "")
].copy()


# =========================
# NORMALIZATION
# =========================
def normalize_requester(value: str) -> str:
    value = str(value).strip()
    if not value:
        return value
    if value.upper().startswith("U"):
        return value.upper()
    return requester_email_to_id.get(value.lower(), value)


def normalize_recommender(value: str) -> str:
    value = str(value).strip()
    if not value:
        return value
    if value.upper().startswith("U"):
        return value.upper()
    return recommender_email_to_id.get(value.lower(), value)


# =========================
# APP
# =========================
app = FastAPI(title="RecommendMe Rating Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODELS
# =========================
class PredictRequest(BaseModel):
    requester_id: str
    recommender_id: str
    requestor_note: str = ""


class NetworkRequest(BaseModel):
    requester_id: str
    recommender_id: str


class SuggestionRequest(BaseModel):
    requester_id: str
    recommender_id: str
    requestor_note: str = ""
    max_suggestions: int = 5


# =========================
# CORE SCORING
# =========================
def score_pair_svd(requester_id, recommender_id, requestor_note=""):
    original_requester = str(requester_id).strip()
    original_recommender = str(recommender_id).strip()

    requester_id = normalize_requester(original_requester)
    recommender_id = normalize_recommender(original_recommender)
    requestor_note = str(requestor_note)

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
        x = tfidf.transform([requestor_note]).toarray()
        pred = cold_model.predict(x)[0]
        case = "cold_start"

    pred = float(np.clip(pred, 1, 5))

    return {
        "original_requester_input": original_requester,
        "original_recommender_input": original_recommender,
        "normalized_requester_id": requester_id,
        "normalized_recommender_id": recommender_id,
        "predicted_rating": round(pred, 2),
        "predicted_star_rating": int(round(pred)),
        "case": case
    }


# =========================
# NETWORK GRAPH
# =========================
def build_immediate_network(requester_input: str, recommender_input: str):
    requester_id = normalize_requester(requester_input)
    recommender_id = normalize_recommender(recommender_input)

    focus_ids = {requester_id, recommender_id}

    immediate_rows = graph_df[
        (graph_df["requester_id"].isin(focus_ids)) |
        (graph_df["recommender_id"].isin(focus_ids))
    ].copy()

    requester_neighbors = set(
        immediate_rows.loc[
            (immediate_rows["requester_id"] == requester_id) |
            (immediate_rows["recommender_id"] == requester_id),
            ["requester_id", "recommender_id"]
        ].values.ravel()
    ) - {requester_id}

    recommender_neighbors = set(
        immediate_rows.loc[
            (immediate_rows["requester_id"] == recommender_id) |
            (immediate_rows["recommender_id"] == recommender_id),
            ["requester_id", "recommender_id"]
        ].values.ravel()
    ) - {recommender_id}

    shared_neighbors = requester_neighbors & recommender_neighbors
    all_ids = {requester_id, recommender_id} | requester_neighbors | recommender_neighbors

    sub_df = graph_df[
        graph_df["requester_id"].isin(all_ids) &
        graph_df["recommender_id"].isin(all_ids)
    ].copy()

    def node_stats(uid: str):
        given = graph_df.loc[graph_df["requester_id"] == uid, "recomender_star_rating"].dropna()
        received = graph_df.loc[graph_df["recommender_id"] == uid, "recomender_star_rating"].dropna()

        return {
            "ratings_given_count": int((graph_df["requester_id"] == uid).sum()),
            "ratings_received_count": int((graph_df["recommender_id"] == uid).sum()),
            "avg_rating_given": round(float(given.mean()), 2) if len(given) else None,
            "avg_rating_received": round(float(received.mean()), 2) if len(received) else None,
        }

    nodes = []
    for uid in sorted(all_ids):
        if uid == requester_id:
            group = "focus_requester"
        elif uid == recommender_id:
            group = "focus_recommender"
        elif uid in shared_neighbors:
            group = "shared_neighbor"
        elif uid in requester_neighbors:
            group = "requester_neighbor"
        else:
            group = "recommender_neighbor"

        stats = node_stats(uid)
        node_size = 3 + min(
            stats["ratings_given_count"] + stats["ratings_received_count"], 12
        ) * 0.8

        nodes.append({
            "id": uid,
            "label": uid,
            "group": group,
            "size": node_size,
            **stats
        })

    edges = []
    for _, row in sub_df.iterrows():
        edges.append({
            "source": row["requester_id"],
            "target": row["recommender_id"],
            "rating": None if pd.isna(row["recomender_star_rating"]) else float(row["recomender_star_rating"])
        })

    return {
        "requester_id": requester_id,
        "recommender_id": recommender_id,
        "shared_neighbors_count": len(shared_neighbors),
        "nodes": nodes,
        "edges": edges
    }


# =========================
# SUGGESTIONS
# =========================
def build_recommender_suggestions(
    requester_input: str,
    recommender_input: str,
    requestor_note: str = "",
    max_suggestions: int = 5
):
    requester_id = normalize_requester(requester_input)
    recommender_id = normalize_recommender(recommender_input)
    requestor_note = str(requestor_note)
    max_suggestions = int(max(1, min(max_suggestions, 5)))

    current_score = score_pair_svd(
        requester_id=requester_id,
        recommender_id=recommender_id,
        requestor_note=requestor_note
    )
    current_pred = float(current_score["predicted_rating"])

    incoming = graph_df[
        (graph_df["recommender_id"] == recommender_id) &
        (graph_df["requester_id"] != requester_id) &
        (graph_df["requester_id"] != recommender_id) &
        (graph_df["recomender_star_rating"].notna())
    ][["requester_id", "recomender_star_rating"]].copy()

    if incoming.empty:
        return {
            "requester_id": requester_id,
            "current_recommender_id": recommender_id,
            "current_predicted_rating": round(current_pred, 2),
            "suggestions": []
        }

    # Keep the highest rating each candidate gave to the current recommender
    incoming = (
        incoming.groupby("requester_id", as_index=False)["recomender_star_rating"]
        .max()
        .rename(columns={
            "requester_id": "candidate_id",
            "recomender_star_rating": "candidate_to_current_rating"
        })
    )

    # New behavior: just take the top N highest raters, no threshold against current_pred
    incoming = incoming.sort_values(
        by=["candidate_to_current_rating", "candidate_id"],
        ascending=[False, True]
    ).head(max_suggestions)

    suggestions = []
    for _, row in incoming.iterrows():
        candidate_id = str(row["candidate_id"]).strip().upper()
        candidate_to_current_rating = float(row["candidate_to_current_rating"])

        candidate_score = score_pair_svd(
            requester_id=requester_id,
            recommender_id=candidate_id,
            requestor_note=requestor_note
        )

        predicted_for_requester = float(candidate_score["predicted_rating"])

        suggestions.append({
            "candidate_id": candidate_id,
            "candidate_to_current_rating": round(candidate_to_current_rating, 2),
            "predicted_rating_for_requester": round(predicted_for_requester, 2),
            "predicted_star_rating_for_requester": int(round(predicted_for_requester)),
            "case": candidate_score["case"]
        })

    # Rank final display by strongest predicted alternative, then strongest historical rating to current recommender
    suggestions = sorted(
        suggestions,
        key=lambda x: (
            x["predicted_rating_for_requester"],
            x["candidate_to_current_rating"],
            x["candidate_id"]
        ),
        reverse=True
    )

    return {
        "requester_id": requester_id,
        "current_recommender_id": recommender_id,
        "current_predicted_rating": round(current_pred, 2),
        "suggestions": suggestions
    }


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/mappings")
def mappings():
    return {
        "requester_email_to_id": requester_email_to_id,
        "recommender_email_to_id": recommender_email_to_id
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    return score_pair_svd(
        requester_id=payload.requester_id,
        recommender_id=payload.recommender_id,
        requestor_note=payload.requestor_note
    )


@app.post("/network")
def network(payload: NetworkRequest):
    return build_immediate_network(
        requester_input=payload.requester_id,
        recommender_input=payload.recommender_id
    )


@app.post("/suggestions")
def suggestions(payload: SuggestionRequest):
    return build_recommender_suggestions(
        requester_input=payload.requester_id,
        recommender_input=payload.recommender_id,
        requestor_note=payload.requestor_note,
        max_suggestions=payload.max_suggestions
    )