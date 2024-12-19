from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chatbot, getplayers, getteams, model , predict,get_match_details,players_of_team

app = FastAPI()

origins = ['http://localhost:3000', 'https://localhost:3000',"https://nhtb6hpt-3000.inc1.devtunnels.ms","https://nhtb6hpt-8000.inc1.devtunnels.ms", "https://nhtb6hpt-8000.inc1.devtunnels.ms:8000"]

# Adding CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers from different files
app.include_router(chatbot.app)
app.include_router(getplayers.app)
app.include_router(getteams.app)
app.include_router(model.app)
app.include_router(predict.app)
app.include_router(get_match_details.app)
app.include_router(players_of_team.app)


# @app.on_event("startup")
# async def startup_event():
#     # Optional: Preload the dataset during app startup
#     from shared_data import get_data
#     get_data()  # Preload the dataset at startup

# Path: routes/predict.py
# @app.post("/predict")

#Path: routes/model.py
# @app.post("/train_and_test")
# @app.get("/")

#Path: routes/getplayers.py
# @app.get("/players")

#Path: routes/getteams.py
# @app.get("/unique_teams")

#Path: routes/chatbot.py
# @app.post("/chatbot")

#Path: routes/get_match_details.py
# @app.post("/get-match-details")  include date(in MM-DD-YYYY ), team_a, team_b

