# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
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

# ---------------- Feature Builders (match your CSV) ----------------

def build_qb_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("passing_yards_prev", 0.0),
        player.get("rushing_yards_prev", 0.0),
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
    ]

def build_rb_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("touches_prev", 0.0),
        player.get("rushing_yards_prev", 0.0),
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
    ]

def build_wr_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("receiving_yards_prev", 0.0),
        player.get("targets_prev", 0.0),
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
    ]

def build_te_features(player):
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("receiving_yards_prev", 0.0),
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
    ]

FEATURE_BUILDERS = {
    "QB": build_qb_features,
    "RB": build_rb_features,
    "WR": build_wr_features,
    "TE": build_te_features,
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

    features = FEATURE_BUILDERS[position](player)
    model = models[position][model_type]

    prediction = model.predict([features])[0]
    return float(prediction)

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
    prediction_value = predict_for_player(player)

    return PredictionResponse(
        player_name=player["player_name"],
        prediction=prediction_value,
    )
