from fastapi import FastAPI,APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
from shared_data import get_data

app = APIRouter()

@app.get("/unique_teams")
async def get_unique_teams():
    try:
        # data = pd.read_csv('predict_dataset.csv')
        data = get_data()
        if 'team' not in data.columns:
            return JSONResponse(status_code=400, content={"error": "Column 'team' not found in the dataset."})
        unique_teams = data['team'].dropna().unique().tolist()
        return {"unique_teams": unique_teams}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
