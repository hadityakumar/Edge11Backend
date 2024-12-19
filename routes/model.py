from fastapi import FastAPI,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import shap
import sys
import json
from fastapi.responses import JSONResponse
from shared_data import get_data

app = APIRouter()

class DateRangeRequest(BaseModel):
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Edge11 App!"}



@app.post("/train_and_test")
async def train_and_test(request: DateRangeRequest):
    train_start_date = request.train_start_date
    train_end_date = request.train_end_date
    test_start_date = request.test_start_date
    test_end_date = request.test_end_date

    # Load the data
    # data = pd.read_csv('predict_dataset.csv')
    data = get_data()
    data['date'] = pd.to_datetime(data['date'])
    
    # Filter data by train and test date range
    train_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)].copy()
    test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)].copy()
    
    # Prepare features and target as per your model code
    numerical_features = ['avg_runs_last_10', 'avg_wickets_last_10', 'std_runs_last_10',
                          'avg_runs_venue', 'avg_runs_vs_opponent', 'matches_played', 
                          'total_wickets_cumulative', 'weighted_avg_runs_last_10','avg_fp_last_10',
                          'std_fp_last_10', 'avg_fp_venue', 'avg_fp_vs_opponent',
                          'weighted_avg_fp_last_10']
    categorical_features = ['player_name', 'match_type','Weather']
    target = 'fantasy_points'
    
    # Handle missing values
    train_data[numerical_features] = train_data[numerical_features].fillna(train_data[numerical_features].median(axis=0))
    train_data[categorical_features] = train_data[categorical_features].fillna(train_data[categorical_features].mode().iloc[0])

    Q1 = train_data[target].quantile(0.25)
    Q3 = train_data[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_data = train_data[(train_data[target] >= lower_bound) & (train_data[target] <= upper_bound)]

    # Feature engineering
    train_data['runs_wickets_interaction'] = train_data['avg_runs_last_10'] * train_data['avg_wickets_last_10']
    train_data['runs_per_match'] = train_data['total_runs_cumulative'] / train_data['matches_played']
    train_data['wickets_per_match'] = train_data['total_wickets_cumulative'] / train_data['matches_played']
    train_data['prev_runs'] = train_data.groupby('player_name')['total_runs_scored'].shift(1)
    train_data['prev_wickets'] = train_data.groupby('player_name')['wickets_taken'].shift(1)
    train_data['prev_runs'].fillna(train_data['prev_runs'].median(), inplace=True)
    train_data['prev_wickets'].fillna(train_data['prev_wickets'].median(), inplace=True)
    
    def weighted_average(group, field, N):
        weights = np.arange(1, N+1)
        result1 = group[field].rolling(window=N, min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        return result1

    train_data['weighted_avg_runs_last_5'] = train_data.groupby('player_name').apply(
    lambda x: weighted_average(x, 'total_runs_scored', 5)).reset_index(level=0, drop=True)
    
    train_data['weighted_avg_fp_last_5'] = train_data.groupby('player_name').apply(
    lambda x: weighted_average(x, 'fantasy_points', 5)).reset_index(level=0, drop=True)

    numerical_features.extend([ 
        'runs_wickets_interaction', 'runs_per_match', 'wickets_per_match',
        'prev_runs', 'prev_wickets', 'weighted_avg_runs_last_5'
    ])
    train_data[numerical_features] = train_data[numerical_features].fillna(train_data[numerical_features].median())
    train_data[categorical_features] = train_data[categorical_features].fillna(train_data[categorical_features].mode().iloc[0])
    
    from scipy.stats import skew

    skewed_feats = train_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index

    for feat in skewed_feats:
        train_data[feat] = np.log1p(train_data[feat])
    # Prepare the data
    
    train_data['total_fp_cumulative'] = train_data['total_fp_cumulative'].fillna(1)
    # train_data['prev_fp'] = train_data['prev_fp'].fillna(1)

    
    X = train_data[numerical_features + categorical_features]
    y = train_data[target]
    
    # Label encoding
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    split_index = int(len(train_data) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_val = X.iloc[split_index:]
    y_val = y.iloc[split_index:]
    
    infinite_values = train_data[numerical_features].applymap(np.isinf)

    
    # Train the model
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=None
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # Save the model and encoders
    joblib.dump(xgb_model, 'xgb_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    test_data_cleaned = test_data[test_data['player_name'].isin(train_data['player_name'])]
    
    # Prepare test data
    X_test = test_data_cleaned[numerical_features + categorical_features]
    y_test = test_data_cleaned[target]
    
    # Handle unseen labels in the test set: Only encode known labels
    for col in categorical_features:
        le = label_encoders.get(col)
        if le:
            X_test[col] = le.transform(X_test[col].astype(str))  # Only transform labels that exist in the encoder
    
    y_pred = xgb_model.predict(X_test)

    # Add predicted points to the test data
    test_data_cleaned['predicted_points'] = y_pred
    
    # Group by date and venue (or match identifier) for each match
    result = []
    grouped_matches = test_data_cleaned.groupby(['date', 'venue'])
    
    for (match_date, venue), match_data in grouped_matches:
        teams = match_data['team'].unique()
        team_1 = teams[0] if len(teams) > 0 else "Unknown"
        team_2 = teams[1] if len(teams) > 1 else "Unknown"
        # Get top 11 predicted players for the match based on predicted points
        top_predicted_players = match_data.sort_values(by='predicted_points', ascending=False).head(11)
        
        # Get top 11 actual players for the match based on fantasy points
        top_dream_team_players = match_data.sort_values(by='fantasy_points', ascending=False).head(11)
        
        # Calculate MAE for the match (predicted vs actual)
        mae = mean_absolute_error(top_dream_team_players['fantasy_points'], top_predicted_players['predicted_points'])
        # team_1 = test_data[test_data['team'] == row['team']]['team'].iloc[0] if not test_data[test_data['team'] == row['team']].empty else "Unknown"
        # team_2 = test_data[test_data['team'] != row['team']]['team'].iloc[0] if not test_data[test_data['team'] != row['team']].empty else "Unknown"
        # Prepare the match result entry
        match_result = {
            "date": match_date.strftime('%Y-%m-%d'),
            "venue": venue,
            "team_1": team_1,  
            "team_2": team_2,  
            "predicted_players": top_predicted_players[['player_name', 'predicted_points']].to_dict(orient='records'),
            "dream_team_players": top_dream_team_players[['player_name', 'fantasy_points']].to_dict(orient='records'),
            "mae": mae
        }
        
        # Append the match result to the final result list
        result.append(match_result)

    # Return the final structured result
    return {"matches": result}
