from fastapi import FastAPI, HTTPException,APIRouter
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict
from datetime import datetime
from shared_data import get_data

app = APIRouter()

# Load the model and encoders at startup
xgb_model = joblib.load("final_model2.pkl")
label_encoders = joblib.load("final_label_encoders.pkl")
# data = pd.read_csv("predict_dataset.csv")
data = get_data()
data["date"] = pd.to_datetime(data["date"])


def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def prepare_data(input_data: Dict):
    """
    Prepare player data based on the input JSON request.
    """
    date = input_data.get("date")
    match_type = input_data.get("match_type")
    team = input_data.get("team")
    opponent = input_data.get("opponent")
    player_names = input_data.get("player_names")
    weather = input_data.get("weather")

    if not date or not match_type or not player_names or not team or not opponent or not weather:
        raise HTTPException(status_code=400, detail="Missing required input data.")

    input_date = pd.to_datetime(date)
    cutoff_date = pd.to_datetime("2024-06-30")

    if input_date > cutoff_date:
        input_date = cutoff_date

    features_list = []

    for player_name in player_names:
        player_data = data[data["player_name"] == player_name]

        if player_data.empty:
            # Assign default values if no data for player
            features = {
                # Numerical features
                "avg_runs_last_10": 0,
                "avg_wickets_last_10": 0,
                "std_runs_last_10": 0,
                "avg_runs_venue": 0,
                "avg_runs_vs_opponent": 0,
                "matches_played": 0,
                "total_wickets_cumulative": 0,
                "weighted_avg_runs_last_10": 0,
                "avg_fp_last_10": 0,
                "std_fp_last_10": 0,
                "avg_fp_venue": 0,
                "avg_fp_vs_opponent": 0,
                "weighted_avg_fp_last_10": 0,
                "runs_wickets_interaction": 0,
                "runs_per_match": 0,
                "wickets_per_match": 0,
                "prev_runs": 0,
                "prev_wickets": 0,
                "weighted_avg_runs_last_5": 0,
                "weighted_avg_fp_last_5": 0,
                # Categorical features
                "player_name": player_name,
                "match_type": match_type,
                "opponent": opponent,
                "Weather": weather
            }
            features_list.append(features)
            continue

        player_data = player_data[player_data["date"] <= input_date]

        if player_data.empty:
            # If no data is available up to the input date, assign default values
            features = {
                # Numerical features
                "avg_runs_last_10": 0,
                "avg_wickets_last_10": 0,
                "std_runs_last_10": 0,
                "avg_runs_venue": 0,
                "avg_runs_vs_opponent": 0,
                "matches_played": 0,
                "total_wickets_cumulative": 0,
                "weighted_avg_runs_last_10": 0,
                "avg_fp_last_10": 0,
                "std_fp_last_10": 0,
                "avg_fp_venue": 0,
                "avg_fp_vs_opponent": 0,
                "weighted_avg_fp_last_10": 0,
                "runs_wickets_interaction": 0,
                "runs_per_match": 0,
                "wickets_per_match": 0,
                "prev_runs": 0,
                "prev_wickets": 0,
                "weighted_avg_runs_last_5": 0,
                "weighted_avg_fp_last_5": 0,
                # Categorical features
                "player_name": player_name,
                "match_type": match_type,
                "opponent": opponent,
                "Weather": weather
            }
            features_list.append(features)
            continue

        player_data = player_data.copy()
        player_data["date_diff"] = (input_date - player_data["date"]).dt.days
        nearest_row = player_data.loc[player_data["date_diff"].idxmin()]

        features = {
            # Numerical features
            "avg_runs_last_10": nearest_row["avg_runs_last_10"],
            "avg_wickets_last_10": nearest_row["avg_wickets_last_10"],
            "std_runs_last_10": nearest_row["std_runs_last_10"],
            "avg_runs_venue": nearest_row["avg_runs_venue"],
            "avg_runs_vs_opponent": nearest_row["avg_runs_vs_opponent"],
            "matches_played": nearest_row["matches_played"],
            "total_wickets_cumulative": nearest_row["total_wickets_cumulative"],
            "weighted_avg_runs_last_10": nearest_row["weighted_avg_runs_last_10"],
            "avg_fp_last_10": nearest_row["avg_fp_last_10"],
            "std_fp_last_10": nearest_row["std_fp_last_10"],
            "avg_fp_venue": nearest_row["avg_fp_venue"],
            "avg_fp_vs_opponent": nearest_row["avg_fp_vs_opponent"],
            "weighted_avg_fp_last_10": nearest_row["weighted_avg_fp_last_10"],
            "runs_wickets_interaction": nearest_row["runs_wickets_interaction"],
            "runs_per_match": nearest_row["runs_per_match"],
            "wickets_per_match": nearest_row["wickets_per_match"],
            "prev_runs": nearest_row["prev_runs"],
            "prev_wickets": nearest_row["prev_wickets"],
            "weighted_avg_runs_last_5": nearest_row["weighted_avg_runs_last_5"],
            "weighted_avg_fp_last_5": nearest_row["weighted_avg_fp_last_5"],
            # Categorical features
            "player_name": player_name,
            "match_type": match_type,
            "opponent": opponent,
            "Weather": weather
        }

        # print(features)

        # Handle missing values
        for key, value in features.items():
            if pd.isnull(value):
                features[key] = 0 if key not in ["player_name", "match_type", "opponent", "Weather"] else "Unknown"

        features_list.append(features)
    return convert_numpy_types(features_list)


def predict(prepared_data: List[Dict]):
    """
    Predict fantasy points based on prepared data.
    """
    df = pd.DataFrame(prepared_data)

    numerical_features = [
        'avg_runs_last_10', 'avg_wickets_last_10', 'std_runs_last_10',
        'avg_runs_venue', 'avg_runs_vs_opponent', 'matches_played',
        'total_wickets_cumulative', 'weighted_avg_runs_last_10', 'avg_fp_last_10',
        'std_fp_last_10', 'avg_fp_venue', 'avg_fp_vs_opponent',
        'weighted_avg_fp_last_10',
        'runs_wickets_interaction', 'runs_per_match', 'wickets_per_match',
        'prev_runs', 'prev_wickets', 'weighted_avg_runs_last_5', 'weighted_avg_fp_last_5'
    ]

    categorical_features = ['player_name', 'match_type', 'opponent', 'Weather']

    df[numerical_features] = df[numerical_features].fillna(0)

    # Encode categorical features
    for col in categorical_features:
        if col in df:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # # Apply log transformation to numerical features
    # for feat in numerical_features:
    #     if feat in df:
    #         df[feat] = np.log1p(df[feat])

    # Make predictions
    predictions = xgb_model.predict(df[numerical_features + categorical_features])

    # Inverse transform player names if necessary
    df["player_name_original"] = label_encoders["player_name"].inverse_transform(df["player_name"])

    output = [
        {"player_name": player_name, "predicted_fantasy_points": round(float(pred), 2)}
        for player_name, pred in zip(df["player_name_original"], predictions)
    ]

    return output


@app.post("/predict")
async def input_endpoint(input_data: Dict):
    """
    Handle the input data, prepare it, and predict fantasy points.
    """
    try:
        prepared_data = prepare_data(input_data)
        predictions = predict(prepared_data)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)

