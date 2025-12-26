# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel
from supabase import create_client
import joblib
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


# ---------- QB ----------
def build_qb_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_qb_veteran_features(player):
    return [
        player.get("passing_yards_prev", 0.0),
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("adp", 200.0),
        player.get("rushing_yards_prev", 0.0),
        player.get("qb_dropback_prev",0.0),
        player.get("team_offense_snaps_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
    ]


# ---------- RB ----------
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


# ---------- WR ----------
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
        player.get("age",20.0),
        player.get("first_down_pass_prev", 0.0),
    ]


# ---------- TE ----------
def build_te_rookie_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age",20.0)
    ]

def build_te_veteran_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age",20.0)
    ]


# ---------- Mapping ----------
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

# ---------------- Prediction Logic ----------------

def predict_for_player(player):
    position = player.get("position")
    if position not in models:
        raise HTTPException(400, f"Unsupported position: {position}")

    years = player.get("years_exp")
    if years is None:
        raise HTTPException(400, "Player missing years_exp field.")

    is_rookie = (years == 0)
    model_type = "rookie" if is_rookie else "veteran"

    # FIXED INDENTATION BELOW
    feature_builder = FEATURE_BUILDERS[(position, is_rookie)]
    features = feature_builder(player)

    model = models[position][model_type]
    prediction = model.predict([features])[0]
    return float(prediction)
@app.get("/")
def root():
    return {"message": "Fantasy backend is running."}

@app.get("/players")
def get_players():
    return supabase.table("players").select("*").execute().data

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    player = get_player_by_name(req.player_name, req.team)
    prediction_value = predict_for_player(player)

    return PredictionResponse(
        player_name=player["player_name"],
        prediction=prediction_value,
    )
@app.get("/rankings")
def get_rankings(position: Optional[str] = Query(None)):
    # 1. Get all players
    players = supabase.table("players").select("*").execute().data

    # 2. Optionally filter by position
    if position:
        players = [p for p in players if p.get("position") == position]

    rankings = []
    for p in players:
        try:
            value = predict_for_player(p)
            rankings.append({
                "player_name": p["player_name"],
                "position": p.get("position"),
                "team": p.get("team"),
                "prediction": value,
            })
        except Exception:
            # You can log this instead of skipping silently
            continue

    # 3. Sort by prediction descending (highest value first)
    rankings.sort(key=lambda x: x["prediction"], reverse=True)
    return rankings

class TradeRequest(BaseModel):
    team_a: List[str]
    team_b: List[str]
    # optional: league format, scoring, etc. later


@app.post("/trade/analyze")
def analyze_trade(req: TradeRequest):
    def side_value(names: List[str]):
        total = 0.0
        details = []
        for name in names:
            player = get_player_by_name(name)
            value = predict_for_player(player)
            details.append({
                "player_name": player["player_name"],
                "position": player.get("position"),
                "team": player.get("team"),
                "prediction": value,
            })
            total += value
        return total, details

    total_a, details_a = side_value(req.team_a)
    total_b, details_b = side_value(req.team_b)

    diff = total_a - total_b
    if diff > 0:
        winner = "team_a"
    elif diff < 0:
        winner = "team_b"
    else:
        winner = "even"

    return {
        "team_a": {
            "total_value": total_a,
            "players": details_a,
        },
        "team_b": {
            "total_value": total_b,
            "players": details_b,
        },
        "difference": diff,
        "better_side": winner,
    }
