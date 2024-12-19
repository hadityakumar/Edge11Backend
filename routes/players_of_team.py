from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
import pandas as pd
from shared_data import get_data

app = APIRouter()

# Load CSV data
# csv_file = "final_dataset1.csv"  
try:
    # df = pd.read_csv(csv_file)
    df = get_data()
except Exception as e:
    raise RuntimeError(f"Failed to load CSV file: {e}")

class TeamRequest(BaseModel):
    team_name: str

@app.post("/get-unique-players")
async def get_unique_players(request: TeamRequest):
    team_name = request.team_name

    # Check if the team exists in the data
    if team_name not in df['team'].unique():
        raise HTTPException(status_code=404, detail="Team not found")

    # Filter rows for the specified team
    team_players = df[df['team'] == team_name]

    # Group by player_name and calculate the average fantasy points
    player_data = (
        team_players.groupby('player_name', as_index=False)['fantasy_points']
        .mean()
        .rename(columns={'fantasy_points': 'average_fantasy_points'})
    )
    player_data['average_fantasy_points'] = player_data['average_fantasy_points'].apply(lambda x: round(x, 2))

    # Convert the result into a list of dictionaries
    players_with_points = player_data.to_dict(orient='records')

    return {"team_name": team_name, "players": players_with_points}
