# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from supabase import create_client
import joblib
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

app = FastAPI(title="Fantasy Backend API")

# ---------------- Supabase Setup ----------------
# Use env vars in Render for safety. Set these in your Render dashboard.
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://octfocmufcukbfhaepjf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "sb_publishable_5mYG0HDrdEZD0hFZIk-zyg_BarGoaZy")

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
# These are per-position; rookies and veterans both map into these.
POSITION_TARGETS: Dict[str, Dict[str, float]] = {
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

class TradeRequest(BaseModel):
    team_a: List[str]
    team_b: List[str]

# ---------------- Supabase Helpers ----------------

def get_player_by_name(player_name: str, team: Optional[str] = None) -> Dict[str, Any]:
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

def get_players_by_position(position: str) -> List[Dict[str, Any]]:
    resp = (
        supabase
        .table("players")
        .select("*")
        .eq("position", position)
        .execute()
    )
    return resp.data or []

# ---------------- Feature Builders ----------------

def build_qb_rookie_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_qb_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("passing_yards_prev", 0.0),
        player.get("adp", 200.0),
        player.get("rushing_yards_prev", 0.0),
        player.get("qb_dropback_prev", 0.0),
        player.get("team_offense_snaps_prev", 0.0),
        player.get("first_down_pass_prev", 0.0),
    ]

def build_rb_rookie_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_rb_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("adp", 200.0),
        player.get("touches_prev", 0.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

def build_wr_rookie_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
    ]

def build_wr_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("fantasy_points_ppr_prev", 0.0),
        player.get("adp", 200.0),
        player.get("receiving_yards_prev", 0.0),
        player.get("draft_ovr", 265.0),
        player.get("targets_prev", 0.0),
        player.get("age", 20.0),
        player.get("first_down_pass_prev", 0.0),
    ]

def build_te_rookie_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

def build_te_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        player.get("adp", 200.0),
        player.get("draft_ovr", 265.0),
        player.get("age", 20.0),
    ]

FEATURE_BUILDERS: Dict[Tuple[str, bool], Any] = {
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

def raw_predict(player: Dict[str, Any]) -> float:
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

# ---------------- Quantile Mapping Helpers ----------------

def compute_percentiles(values: np.ndarray) -> np.ndarray:
    """
    Compute empirical percentiles for each value:
    percentile = rank / (n - 1)
    """
    n = len(values)
    if n == 1:
        return np.array([0.5])

    sorted_idx = np.argsort(values)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.arange(n, dtype=float)
    return ranks / (n - 1)

def interpolate_target(position: str, p: float) -> float:
    """
    Map percentile p in [0,1] to a target value for a given position, using
    25th, 50th, 75th percentiles and max for extrapolation.
    Min is NOT forced.
    """
    if position not in POSITION_TARGETS:
        raise ValueError(f"No target distribution for position {position}")
    t = POSITION_TARGETS[position]
    q25, q50, q75, vmax = t["q25"], t["q50"], t["q75"], t["max"]

    if p <= 0.25:
        # Linear from 0 -> q25 (soft lower tail)
        return q25 * (p / 0.25)
    elif p <= 0.50:
        # Between q25 and q50
        return q25 + (q50 - q25) * ((p - 0.25) / 0.25)
    elif p <= 0.75:
        # Between q50 and q75
        return q50 + (q75 - q50) * ((p - 0.50) / 0.25)
    else:
        # Between q75 and max (upper tail)
        return q75 + (vmax - q75) * ((p - 0.75) / 0.25)

def quantile_map_position(position: str, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Core speed optimization:
    - For a position, compute raw predictions once for all players
    - Compute percentiles once
    - Map all to normalized values
    Returns a list of {player_name, team, position, prediction}.
    """
    if not players:
        return []

    raw_vals = []
    valid_players = []
    for p in players:
        try:
            v = raw_predict(p)
            raw_vals.append(v)
            valid_players.append(p)
        except Exception:
            # Skip players that fail prediction
            continue

    if not raw_vals:
        return []

    arr = np.array(raw_vals, dtype=float)
    percentiles = compute_percentiles(arr)
    normalized_vals = [interpolate_target(position, float(p)) for p in percentiles]

    results = []
    for p, norm in zip(valid_players, normalized_vals):
        results.append({
            "player_name": p["player_name"],
            "position": p.get("position"),
            "team": p.get("team"),
            "prediction": float(norm),
        })
    return results

def get_normalized_for_position(position: str) -> List[Dict[str, Any]]:
    """
    Fetch all players at a position, compute normalized predictions once.
    """
    players = get_players_by_position(position)
    return quantile_map_position(position, players)

# ---------------- API Endpoints ----------------

@app.get("/")
def root():
    return {"message": "Fantasy backend is running."}

@app.get("/players")
def get_players():
    return supabase.table("players").select("*").execute().data

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # 1. Get the specific player
    player = get_player_by_name(req.player_name, req.team)
    position = player.get("position")
    if not position or position not in POSITION_TARGETS:
        raise HTTPException(400, f"Unsupported position: {position}")

    # 2. Get normalized predictions for this position (computed once)
    normalized_list = get_normalized_for_position(position)

    # 3. Find this player's normalized prediction
    name = player["player_name"]
    team = player.get("team")

    for entry in normalized_list:
        if entry["player_name"] == name:
            # If team specified in DB, match it; otherwise just match name
            if team is None or entry.get("team") == team:
                return PredictionResponse(
                    player_name=name,
                    prediction=entry["prediction"],
                )

    # Fallback: if somehow not found, just return raw prediction (rare)
    raw_value = raw_predict(player)
    return PredictionResponse(
        player_name=name,
        prediction=float(raw_value),
    )

@app.get("/rankings")
def get_rankings(position: Optional[str] = Query(None)):
    rankings: List[Dict[str, Any]] = []

    if position:
        pos = position.upper()
        if pos not in POSITION_TARGETS:
            raise HTTPException(400, f"Unsupported position: {pos}")
        rankings = get_normalized_for_position(pos)
    else:
        # All positions
        for pos in POSITION_TARGETS.keys():
            rankings.extend(get_normalized_for_position(pos))

    rankings.sort(key=lambda x: x["prediction"], reverse=True)
    return rankings

@app.post("/trade/analyze")
def analyze_trade(req: TradeRequest):
    """
    Trade analyzer using normalized predictions.
    - Group players by position
    - For each position, compute normalized predictions once
    - Look up values for the trade players
    """

    # 1. Fetch all players for both sides
    all_names = list(set(req.team_a + req.team_b))
    name_to_player: Dict[str, Dict[str, Any]] = {}
    for name in all_names:
        player = get_player_by_name(name)
        name_to_player[name] = player

    # 2. Group by position
    pos_to_players: Dict[str, List[Dict[str, Any]]] = {}
    for player in name_to_player.values():
        pos = player.get("position")
        if pos not in POSITION_TARGETS:
            continue
        pos_to_players.setdefault(pos, []).append(player)

    # 3. For each position, compute normalized predictions once
    pos_to_norm_list: Dict[str, List[Dict[str, Any]]] = {}
    pos_to_lookup: Dict[str, Dict[Tuple[str, Optional[str]], float]] = {}

    for pos, players in pos_to_players.items():
        norm_list = quantile_map_position(pos, players)
        pos_to_norm_list[pos] = norm_list
        lookup: Dict[Tuple[str, Optional[str]], float] = {}
        for entry in norm_list:
            key = (entry["player_name"], entry.get("team"))
            lookup[key] = entry["prediction"]
        pos_to_lookup[pos] = lookup

    def side_value(names: List[str]):
        total = 0.0
        details = []
        for name in names:
            player = name_to_player.get(name)
            if not player:
                continue
            pos = player.get("position")
            team = player.get("team")
            norm_lookup = pos_to_lookup.get(pos, {})
            value = norm_lookup.get((player["player_name"], team))

            # Fallback to raw if somehow missing
            if value is None:
                value = raw_predict(player)

            details.append({
                "player_name": player["player_name"],
                "position": pos,
                "team": team,
                "prediction": float(value),
            })
            total += float(value)
        return total, details

    total_a, details_a = side_value(req.team_a)
    total_b, details_b = side_value(req.team_b)

    diff = total_a - total_b
    if diff > 0:
        winner = "team_a"
    elif diff < 0:
        winner = "team_b"
        diff = total_a - total_b
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
