# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from supabase import create_client
import joblib
import numpy as np
from typing import Any, Dict, List, Optional

app = FastAPI(title="Fantasy Backend API")

# ---------------- Supabase Setup ----------------

SUPABASE_URL = "https://octfocmufcukbfhaepjf.supabase.co"
SUPABASE_KEY = "sb_publishable_5mYG0HDrdEZD0hFZIk-zyg_BarGoaZy"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Load Models ----------------

models = {
    "QB": {
        "rookie": joblib.load("models/qb_rookie.pkl"),
        "veteran": joblib.load("models/qb_veteran.pkl"),
    },
    "RB": {
        "rookie": joblib.load("models/rb_rookie.pkl"),
        "veteran": joblib.load("models/rb_veteran.pkl"),
    },
    "WR": {
        "rookie": joblib.load("models/wr_rookie.pkl"),
        "veteran": joblib.load("models/wr_veteran.pkl"),
    },
    "TE": {
        "rookie": joblib.load("models/te_rookie.pkl"),
        "veteran": joblib.load("models/te_veteran.pkl"),
    },
}

# ---------------- Target Distributions (Rookies + Vets Same) ----------------

POSITION_TARGETS = {
    "QB": {
        "q25": 14.875,
        "q50": 18.720,
        "q75": 23.185,
        "max": 27.0,
    },
    "RB": {
        "q25": 4.8625,
        "q50": 7.94,
        "q75": 11.965,
        "max": 25.0,
    },
    "WR": {
        "q25": 5.31,
        "q50": 8.17,
        "q75": 11.9,
        "max": 25.0,
    },
    "TE": {
        "q25": 5.49,
        "q50": 7.15,
        "q75": 9.98,
        "max": 18.0,
    },
}

# ---------------- Request/Response Models ----------------

class PredictionRequest(BaseModel):
    player_name: str
    team: Optional[str] = None

class PredictionResponse(BaseModel):
    player_name: str
    prediction: float

# ---------------- Supabase Helpers ----------------

def get_player_by_name(player_name: str, team: Optional[str] = None):
    query = (
        supabase
        .table("players")
        .select("*")
        .ilike("player_name", f"%{player_name}%")
    )

    if team:
        query = query.ilike("team", team)

    resp = query.execute()

    if not resp.data:
        raise HTTPException(404, f"Player '{player_name}' not found.")

    if len(resp.data) > 1:
        teams = [p.get("team", "Unknown") for p in resp.data]
        raise HTTPException(
            400,
            f"Multiple players named '{player_name}' found on: {', '.join(teams)}. Add ?team=TEAM."
        )

    return resp.data[0]

# ---------------- Feature Builders ----------------

def build_qb_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_qb_veteran_features(player):
    return [
        player.get("passing_yards_prev", 0.0),
        player.get("adp", 200.0),
        player.get("rushing_yards_prev", 0.0),
        player.get("qb_dropback_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
    ]

def build_rb_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_rb_veteran_features(player):
    return [
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("adp", 200.0),
        player.get("touches_prev", 0.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

def build_wr_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_wr_veteran_features(player):
    return [
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("adp", 200.0),
        player.get("receiving_yards_prev", 0.0),
        player.get("draft_ovr", 265.0),
        player.get("targets_prev", 0.0),
        player.get("age", 20.0),
        player.get("first_down_pass_prev", 0.0),
    ]

def build_te_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

def build_te_veteran_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

FEATURE_BUILDERS = {
    ("QB", True): build_qb_rookie_features,
    ("QB", False): build_qb_veteran_features,

    ("RB", True): build_rb_rookie_features,
    ("RB", False): build_rb_veteran_features,

    ("WR", True): build_wr_rookie_features,
    ("WR", False): build_wr_veteran_features,

    ("TE", True): build_te_rookie_features,
    ("TE", False): build_te_veteran_features,
}

# ---------------- Raw Prediction Logic ----------------

def raw_predict(player):
    position = player.get("position")
    if position not in models:
        raise HTTPException(400, f"Unsupported position: {position}")

    years = player.get("years_exp")
    if years is None:
        raise HTTPException(400, "Player missing years_exp field.")

    is_rookie = (years == 0)
    model_type = "rookie" if is_rookie else "veteran"

    feature_builder = FEATURE_BUILDERS[(position, is_rookie)]
    features = feature_builder(player)

    model = models[position][model_type]
    prediction = model.predict([features])[0]
    return float(prediction)

# ---------------- Quantile Mapping ----------------

def compute_percentiles(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([0.5])
    sorted_idx = np.argsort(values)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.arange(len(values), dtype=float)
    return ranks / (len(values) - 1)

def interpolate_target(position: str, p: float) -> float:
    t = POSITION_TARGETS[position]
    q25, q50, q75 = t["q25"], t["q50"], t["q75"]

    if p <= 0.25:
        return q25 * (p / 0.25)
    elif p <= 0.50:
        return q25 + (q50 - q25) * ((p - 0.25) / 0.25)
    elif p <= 0.75:
        return q50 + (q75 - q50) * ((p - 0.50) / 0.25)
    else:
        return q75 + (t["max"] - q75) * ((p - 0.75) / 0.25)

def normalize_single_prediction(player, raw_value):
    position = player["position"]

    # Get all players of same position
    players = supabase.table("players").select("*").eq("position", position).execute().data
    raw_values = [raw_predict(p) for p in players]

    arr = np.array(raw_values, dtype=float)
    percentiles = compute_percentiles(arr)

    # Find percentile of this player's raw value
    sorted_vals = np.sort(arr)
    idx = np.searchsorted(sorted_vals, raw_value, side="left")
    p = idx / (len(arr) - 1) if len(arr) > 1 else 0.5

    return interpolate_target(position, p)

# ---------------- API Endpoints ----------------

@app.get("/")
def root():
    return {"message": "Fantasy backend is running."}

@app.get("/players")
def get_players():
    return supabase.table("players").select("*").execute().data

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    player = get_player_by_name(req.player_name, req.team)
    raw_value = raw_predict(player)
    normalized = normalize_single_prediction(player, raw_value)

    return PredictionResponse(
        player_name=player["player_name"],
        prediction=normalized,
    )

@app.get("/rankings")
def get_rankings(position: Optional[str] = Query(None)):
    players = supabase.table("players").select("*").execute().data

    if position:
        players = [p for p in players if p.get("position") == position]

    rankings = []
    for p in players:
        try:
            raw_value = raw_predict(p)
            normalized = normalize_single_prediction(p, raw_value)
            rankings.append({
                "player_name": p["player_name"],
                "position": p.get("position"),
                "team": p.get("team"),
                "prediction": normalized,
            })
        except:
            continue

    rankings.sort(key=lambda x: x["prediction"], reverse=True)
    return rankings

class TradeRequest(BaseModel):
    team_a: List[str]
    team_b: List[str]

@app.post("/trade/analyze")
def analyze_trade(req: TradeRequest):
    def side_value(names: List[str]):
        total = 0.0
        details = []
        for name in names:
            player = get_player_by_name(name)
            raw_value = raw_predict(player)
            normalized = normalize_single_prediction(player, raw_value)
            details.append({
                "player_name": player["player_name"],
                "position": player.get("position"),
                "team": player.get("team"),
                "prediction": normalized,
            })
            total += normalized
        return total, details

    total_a, details_a = side_value(req.team_a)
    total_b, details_b = side_value(req.team_b)

    diff = total_a - total_b
    winner = "team_a" if diff > 0 else "team_b" if diff < 0 else "even"

    return {
        "team_a": {"total_value": total_a, "players": details_a},
        "team_b": {"total_value": total_b, "players": details_b},
        "difference": diff,
        "better_side": winner,
    }
