from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
import pandas as pd
from shared_data import get_data

app = APIRouter()

# Load CSV data
# csv_file = "final_dataset1.csv"  # Replace with the actual path to your CSV
try:
    # df = pd.read_csv(csv_file)
    df = get_data()
except Exception as e:
    raise RuntimeError(f"Failed to load CSV file: {e}")

class MatchRequest(BaseModel):
    date: str
    team_a: str
    team_b: str


@app.post("/get-match-details")
async def get_match_details(request: MatchRequest):
    date = request.date
    team_a = request.team_a
    team_b = request.team_b


    # Filter rows by date and match involving both teams
    match = df[
        (df['date'] == date) &
        (
            ((df['team'] == team_a) & (df['opponent'] == team_b)) |
            ((df['team'] == team_b) & (df['opponent'] == team_a))
        )
    ]

    if match.empty:
        raise HTTPException(status_code=404, detail="No match found for the given date and teams")

    # Deduplicate matches by sorting the teams and dropping duplicates
    match['team_pair'] = match.apply(lambda x: tuple(sorted([x['team'], x['opponent']])), axis=1)
    deduplicated_match = match.drop_duplicates(subset=['date', 'team_pair'])

    # Fetch relevant details (assuming one unique match per date and team pair)
    matches = deduplicated_match[['date', 'team', 'opponent', 'venue']].to_dict(orient='records')

    return {"matches": matches}