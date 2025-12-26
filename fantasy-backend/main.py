import os
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from supabase import create_client, Client
import joblib

# ============================
# App and Supabase setup
# ============================

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your-service-role-key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# Model loading
# ============================

# Adjust paths to where your models actually live
QB_VETERAN_MODEL_PATH = "models/qb_veteran.pkl"
RB_VETERAN_MODEL_PATH = "models/rb_veteran.pkl"
WR_VETERAN_MODEL_PATH = "models/wr_veteran.pkl"
TE_VETERAN_MODEL_PATH = "models/te_veteran.pkl"

qb_veteran_model = joblib.load(QB_VETERAN_MODEL_PATH)
rb_veteran_model = joblib.load(RB_VETERAN_MODEL_PATH)
wr_veteran_model = joblib.load(WR_VETERAN_MODEL_PATH)
te_veteran_model = joblib.load(TE_VETERAN_MODEL_PATH)

# ============================
# Target distributions per position
# (means, stds, quartiles, max – min is NOT forced)
# ============================

POSITION_TARGETS: Dict[str, Dict[str, float]] = {
    "QB": {
        "mean": 19.066773,
        "std": 6.501114,
        "q25": 14.875000,
        "q50": 18.720000,
        "q75": 23.185000,
        "max": 27.0,
    },
    "RB": {
        "mean": 8.950465,
        "std": 5.259036,
        "q25": 4.862500,
        "q50": 7.940000,
        "q75": 11.965000,
        "max": 25.0,
    },
    "WR": {
        "mean": 9.164468,
        "std": 4.966812,
        "q25": 5.310000,
        "q50": 8.170000,
        "q75": 11.900000,
        "max": 25.0,
    },
    "TE": {
        "mean": 8.146032,
        "std": 3.578006,
        "q25": 5.490000,
        "q50": 7.150000,
        "q75": 9.980000,
        "max": 18.0,
    },
}

# ============================
# Request / response models
# ============================

class PredictionRequest(BaseModel):
    player_name: str
    team: Optional[str] = None

class PredictionResponse(BaseModel):
    player_name: str
    prediction: float

class TradeRequest(BaseModel):
    team_a: List[str]
    team_b: List[str]

# ============================
# Helpers: fetching players
# ============================

def get_player_by_name(player_name: str, team: Optional[str] = None) -> Dict[str, Any]:
    query = supabase.table("players").select("*").eq("player_name", player_name)
    if team:
        query = query.eq("team", team)
    data = query.execute().data
    if not data:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    return data[0]

def get_all_players_for_position(position: str) -> List[Dict[str, Any]]:
    data = (
        supabase.table("players")
        .select("*")
        .eq("position", position)
        .execute()
        .data
    )
    return data or []

# ============================
# Feature builders per position
# (Make sure these match your training feature order exactly)
# ============================

def build_qb_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        float(player.get("passing_yards_prev", 0.0)),
        float(player.get("fantasy_points_ppr_prev", 0.0)),
        float(player.get("adp", 200.0)),
        float(player.get("rushing_yards_prev", 0.0)),
        float(player.get("qb_dropback_prev", 0.0)),
        float(player.get("team_offense_snaps_prev", 0.0)),
        float(player.get("first_down_pass_prev", 0.0)),
    ]

def build_rb_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        float(player.get("rushing_yards_prev", 0.0)),
        float(player.get("receiving_yards_prev", 0.0)),
        float(player.get("fantasy_points_ppr_prev", 0.0)),
        float(player.get("adp", 200.0)),
        float(player.get("rush_attempts_prev", 0.0)),
        float(player.get("targets_prev", 0.0)),
        float(player.get("touches_prev", 0.0)),
    ]

def build_wr_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        float(player.get("receiving_yards_prev", 0.0)),
        float(player.get("fantasy_points_ppr_prev", 0.0)),
        float(player.get("adp", 200.0)),
        float(player.get("targets_prev", 0.0)),
        float(player.get("receptions_prev", 0.0)),
        float(player.get("team_pass_attempts_prev", 0.0)),
        float(player.get("receiving_tds_prev", 0.0)),
    ]

def build_te_veteran_features(player: Dict[str, Any]) -> List[float]:
    return [
        float(player.get("receiving_yards_prev", 0.0)),
        float(player.get("fantasy_points_ppr_prev", 0.0)),
        float(player.get("adp", 200.0)),
        float(player.get("targets_prev", 0.0)),
        float(player.get("receptions_prev", 0.0)),
        float(player.get("redzone_targets_prev", 0.0)),
        float(player.get("receiving_tds_prev", 0.0)),
    ]

def get_position(player: Dict[str, Any]) -> str:
    pos = player.get("position")
    if not pos:
        raise HTTPException(status_code=400, detail="Player has no position field")
    pos = pos.upper()
    if pos not in {"QB", "RB", "WR", "TE"}:
        raise HTTPException(status_code=400, detail=f"Unsupported position: {pos}")
    return pos

# ============================
# Core model prediction (raw, unnormalized)
# ============================

def raw_predict_for_player(player: Dict[str, Any]) -> float:
    position = get_position(player)

    if position == "QB":
        features = build_qb_veteran_features(player)
        model = qb_veteran_model
    elif position == "RB":
        features = build_rb_veteran_features(player)
        model = rb_veteran_model
    elif position == "WR":
        features = build_wr_veteran_features(player)
        model = wr_veteran_model
    elif position == "TE":
        features = build_te_veteran_features(player)
        model = te_veteran_model
    else:
        raise HTTPException(status_code=400, detail=f"No model for position: {position}")

    arr = np.array(features, dtype=float).reshape(1, -1)
    pred = model.predict(arr)[0]
    return float(pred)

# ============================
# Quantile mapping utilities
# ============================

def compute_percentiles(values: np.ndarray) -> np.ndarray:
    """
    Compute empirical percentiles for each value in 'values'.
    Percentile = rank / (n - 1), where rank is index in sorted order.
    """
    if len(values) == 1:
        return np.array([0.5])  # Single value -> treat as median

    sorted_idx = np.argsort(values)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.arange(len(values), dtype=float)
    percentiles = ranks / (len(values) - 1)
    return percentiles

def interpolate_target_value(position: str, p: float) -> float:
    """
    Map a percentile p in [0,1] to a target value for a given position,
    using the specified target quartiles and linear extrapolation.
    We DO NOT force the min; we only anchor at 0.25, 0.5, 0.75.
    """
    if position not in POSITION_TARGETS:
        raise ValueError(f"No target distribution for position {position}")

    target = POSITION_TARGETS[position]
    q25 = target["q25"]
    q50 = target["q50"]
    q75 = target["q75"]

    # Control points: (percentile, value)
    pts_p = np.array([0.25, 0.50, 0.75], dtype=float)
    pts_v = np.array([q25, q50, q75], dtype=float)

    # Below 25%: extrapolate linearly from [0.25, 0.50]
    if p <= pts_p[0]:
        p1, v1 = pts_p[0], pts_v[0]
        p2, v2 = pts_p[1], pts_v[1]
    # Between 25% and 50%: interpolate within [0.25, 0.50]
    elif p <= pts_p[1]:
        p1, v1 = pts_p[0], pts_v[0]
        p2, v2 = pts_p[1], pts_v[1]
    # Between 50% and 75%: interpolate within [0.50, 0.75]
    elif p <= pts_p[2]:
        p1, v1 = pts_p[1], pts_v[1]
        p2, v2 = pts_p[2], pts_v[2]
    # Above 75%: extrapolate linearly from [0.50, 0.75]
    else:
        p1, v1 = pts_p[1], pts_v[1]
        p2, v2 = pts_p[2], pts_v[2]

    if p2 == p1:
        return float(v1)

    t = (p - p1) / (p2 - p1)
    v = v1 + t * (v2 - v1)
    return float(v)

def quantile_map_predictions_for_position(position: str, raw_values: List[float]) -> List[float]:
    """
    Full quantile-style mapping:
    1. Compute empirical percentiles of raw_values.
    2. Map each percentile to a target value using the position's quartiles.
    """
    if not raw_values:
        return []

    arr = np.array(raw_values, dtype=float)
    if np.all(arr == arr[0]):
        # All equal: map everything to the target mean
        target_mean = POSITION_TARGETS[position]["mean"]
        return [float(target_mean)] * len(raw_values)

    percentiles = compute_percentiles(arr)
    mapped = [interpolate_target_value(position, float(p)) for p in percentiles]
    return mapped

# ============================
# High-level helpers using quantile mapping
# ============================

def get_mapped_predictions_for_position(position: str) -> List[Dict[str, Any]]:
    """
    Fetch all players at a position, get raw predictions, then apply
    quantile mapping to produce normalized predictions.
    Returns list of dicts with player data + 'prediction'.
    """
    players = get_all_players_for_position(position)
    if not players:
        return []

    raw_vals = []
    valid_players = []

    for p in players:
        try:
            v = raw_predict_for_player(p)
            raw_vals.append(v)
            valid_players.append(p)
        except Exception:
            # Skip players that fail to predict
            continue

    if not raw_vals:
        return []

    mapped_vals = quantile_map_predictions_for_position(position, raw_vals)

    result = []
    for p, v in zip(valid_players, mapped_vals):
        result.append(
            {
                "player_name": p["player_name"],
                "position": p.get("position"),
                "team": p.get("team"),
                "prediction": float(v),
            }
        )
    return result

def get_mapped_prediction_for_single_player(player: Dict[str, Any]) -> float:
    """
    To keep predictions consistent, we:
    - Take all players at the same position,
    - Compute quantile-mapped predictions for the full group,
    - Return the mapped value for this specific player.
    """
    position = get_position(player)
    all_mapped = get_mapped_predictions_for_position(position)

    name = player["player_name"]
    team = player.get("team")

    # Try match by name + team if available, else by name only.
    for entry in all_mapped:
        if entry["player_name"] == name:
            if team is None or entry.get("team") == team:
                return float(entry["prediction"])

    # Fallback: if not found (should be rare), just use raw and map it directly
    raw = raw_predict_for_player(player)
    mapped_single = quantile_map_predictions_for_position(position, [raw])
    return float(mapped_single[0])

# ============================
# API endpoints
# ============================

@app.get("/players")
def list_players(position: Optional[str] = Query(None)):
    query = supabase.table("players").select("*")
    if position:
        query = query.eq("position", position.upper())
    data = query.execute().data
    return data or []

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    player = get_player_by_name(req.player_name, req.team)
    mapped_value = get_mapped_prediction_for_single_player(player)

    return PredictionResponse(
        player_name=player["player_name"],
        prediction=mapped_value,
    )

@app.get("/rankings")
def get_rankings(position: Optional[str] = Query(None)):
    """
    If position is provided: return rankings for that position only (mapped).
    If not: return rankings for all positions combined, each position mapped
    to its own target distribution, then sorted globally by prediction.
    """
    rankings: List[Dict[str, Any]] = []

    if position:
        pos = position.upper()
        if pos not in POSITION_TARGETS:
            raise HTTPException(status_code=400, detail=f"Unsupported position: {pos}")
        rankings = get_mapped_predictions_for_position(pos)
    else:
        # All positions
        for pos in POSITION_TARGETS.keys():
            rankings.extend(get_mapped_predictions_for_position(pos))

    rankings.sort(key=lambda x: x["prediction"], reverse=True)
    return rankings

@app.post("/trade/analyze")
def analyze_trade(req: TradeRequest):
    """
    Trade analyzer:
    - For each player name, fetch player and get mapped prediction
      based on their position's distribution.
    - Sum values for each side and compare.
    """

    def side_value(names: List[str]) -> Dict[str, Any]:
        players_info = []
        total = 0.0

        for name in names:
            player = get_player_by_name(name)
            mapped_val = get_mapped_prediction_for_single_player(player)
            info = {
                "player_name": player["player_name"],
                "position": player.get("position"),
                "team": player.get("team"),
                "prediction": mapped_val,
            }
            players_info.append(info)
            total += mapped_val

        return {
            "total_value": total,
            "players": players_info,
        }

    team_a_result = side_value(req.team_a)
    team_b_result = side_value(req.team_b)

    diff = team_a_result["total_value"] - team_b_result["total_value"]
    if diff > 0:
        winner = "team_a"
    elif diff < 0:
        winner = "team_b"
    else:
        winner = "even"

    return {
        "team_a": team_a_result,
        "team_b": team_b_result,
        "difference": diff,
        "better_side": winner,
    }

@app.get("/debug/player/{name}")
def debug_player(name: str, team: Optional[str] = None):
    """
    Debug endpoint:
    Shows raw features, raw prediction, and mapped prediction for a player.
    Useful for sanity-checking the pipeline.
    """
    player = get_player_by_name(name, team)
    position = get_position(player)

    if position == "QB":
        features = build_qb_veteran_features(player)
    elif position == "RB":
        features = build_rb_veteran_features(player)
    elif position == "WR":
        features = build_wr_veteran_features(player)
    elif position == "TE":
        features = build_te_veteran_features(player)
    else:
        features = []

    raw_pred = raw_predict_for_player(player)
    mapped_pred = get_mapped_prediction_for_single_player(player)

    return {
        "player": {
            "player_name": player["player_name"],
            "position": position,
            "team": player.get("team"),
        },
        "features": features,
        "raw_prediction": raw_pred,
        "mapped_prediction": mapped_pred,
    }
