from fastapi import APIRouter,FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from openai import OpenAI
from rapidfuzz import process,fuzz
import os
from dotenv import load_dotenv
from shared_data import get_data

load_dotenv()

APIKEY = os.getenv('OPENAI-APIKEY')
app = APIRouter()
client=OpenAI(
    api_key = APIKEY
)

# Load player data
# file_path = "predict_dataset.csv"  # Dataset with player stats
# df = pd.read_csv(file_path)
df = get_data()
df['player_name_lower'] = df['player_name'].str.lower()

# Define request model
class ChatRequest(BaseModel):
    question: str

@app.post("/chatbot")
def chatbot(request: ChatRequest):
    question = request.question.lower()

    # Extract the closest matching player name using fuzzy matching
    player_names_list = df['player_name_lower'].tolist()
    best_match, score, _ = process.extractOne(question, player_names_list, scorer=fuzz.partial_ratio)
    
    threshold = 80  # Adjust this threshold as needed
    if score < threshold:
        return {"response": "Sorry, I couldn't identify the player you're referring to. Please check the name."}

    # Retrieve the original player name (case-sensitive) from the dataset
    matched_name_row = df[df['player_name_lower'] == best_match]
    player_name = matched_name_row.iloc[0]['player_name']
    # Get player stats for the identified player
    player_stats = matched_name_row.iloc[0].to_dict()
    # Construct the prompt for OpenAI
    prompt = f"""
    You are an expert cricket analyst AI bot who go by the name of Edge 11. You had been created by the students of Team 57, in the context of Inter IIT 13.0.
    A user has asked: '{question}'
    Provide a detailed response using the following player stats. Give a small response for the same.:
    Name: {player_name}
    Experience: {player_stats.get('experience')}
    Average Batting Strike Rate: {player_stats.get('avg_batting_strike_rate', 0):.2f}
    Average Runs: {player_stats.get('avg_runs', 0):.2f}
    Total Wickets: {player_stats.get('total_wickets', 0):.2f}
    Average Bowling Strike Rate: {player_stats.get('avg_bowling_strike_rate', 0):.2f}
    Average Economy: {player_stats.get('avg_economy', 0):.2f}
    Total Catches: {player_stats.get('total_catches', 0):.2f}
    """

    # Use OpenAI Chat API to get a response
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o"
        )
        reply = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from OpenAI: {str(e)}")

    return {"response": reply}
