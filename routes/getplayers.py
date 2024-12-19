from fastapi import APIRouter,FastAPI
import pandas as pd
from shared_data import get_data

app = APIRouter()
# file_path = "predict_dataset.csv"  
# df = pd.read_csv(file_path)
df = get_data()

@app.get("/players")
def get_players():
    player_data = df.groupby('player_name').agg(
        team=('team', lambda x: ', '.join(x.unique())), 
        experience=('experience', 'first'), 
        avg_batting_strike_rate=('avg_batting_strike_rate', 'mean'),
        avg_runs=('avg_runs', 'mean'),
        total_wickets=('total_wickets', 'sum'),
        avg_bowling_strike_rate=('avg_bowling_strike_rate', 'mean'),
        avg_economy=('avg_economy', 'mean'),
        total_catches=('total_catches', 'sum')
    ).reset_index()

    numeric_cols = ['avg_batting_strike_rate', 'avg_runs', 'avg_bowling_strike_rate', 'avg_economy']
    player_data[numeric_cols] = player_data[numeric_cols].round(2)
    player_list = player_data.to_dict(orient="records")
    return {"players": player_list}
