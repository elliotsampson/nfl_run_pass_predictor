import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load("best_run_pass_model.pkl")
print("Model loaded successfully")
#format of the model 
#columns_needed = [
#    "away_team", "home_team", "posteam", "defteam", "game_seconds_remaining", "game_half", "yardline_100", "down", 
#    "ydstogo", "qtr", "score_differential", "play_type"
#]
#now remove the kick punt type of plays from the list
#df_cleaned = df_filtered[df_filtered["play_type"].isin(["pass", "run"])].copy() 
#cleaned half types and play times 
#df_cleaned["game_half"] = df_cleaned["game_half"].map({"Half1": 1, "Half2": 2}) 
#df_cleaned["play_type"] = df_cleaned["play_type"].map({"run": 0, "pass": 1}) 
# renaming teams to numbers
#team_map = {
#    "ARI": 0, "ATL": 1, "BAL": 2, "BUF": 3, "CAR": 4, "CHI": 5, "CIN": 6, "CLE": 7,
#    "DAL": 8, "DEN": 9, "DET": 10, "GB": 11, "HOU": 12, "IND": 13, "JAX": 14, "KC": 15,
#    "LV": 16, "LAC": 17, "LA": 18, "MIA": 19, "MIN": 20, "NE": 21, "NO": 22, "NYG": 23,
#    "NYJ": 24, "PHI": 25, "PIT": 26, "SF": 27, "SEA": 28, "TB": 29, "TEN": 30, "WAS": 31
#}
# in essence the model takes the following inputs 
# "away_team", "home_team", "posteam", "defteam", "game_seconds_remaining", "game_half", "yardline_100", "down", 
#    "ydstogo", "qtr", "score_differential",
# to predict the following output 
# "play_type" with playtype being {"run": 0, "pass": 1}
