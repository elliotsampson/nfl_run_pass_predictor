import joblib
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

# Load the saved model
best_loaded_model = joblib.load("best_run_pass_model.pkl")
print("Best model loaded successfully")
rival_loaded_model = joblib.load("rival_run_pass_model.pkl")
print("Rival model loaded successfully")

# as before load the data
df = pd.read_csv("nfl_data_2026.csv")
# clean the data to be in the correct format 
columns_needed = [
    "away_team", "home_team", "posteam", "defteam", "game_seconds_remaining", "game_half", "yardline_100", "down", 
    "ydstogo", "qtr", "score_differential", "play_type"
]
df_filtered = df[columns_needed].copy()

#remove the kick / punt type of plays from the list
df_cleaned = df_filtered[df_filtered["play_type"].isin(["pass", "run"])].copy() 

#change the variables to be numbers for the modelling  
#question we ask, was it a pass? 0 = run, 1 = pass 
df_cleaned["game_half"] = df_cleaned["game_half"].map({"Half1": 1, "Half2": 2}) 
df_cleaned["play_type"] = df_cleaned["play_type"].map({"run": 0, "pass": 1}) 

# use numbers to map to team names 
team_map = {
    "ARI": 0, "ATL": 1, "BAL": 2, "BUF": 3, "CAR": 4, "CHI": 5, "CIN": 6, "CLE": 7,
    "DAL": 8, "DEN": 9, "DET": 10, "GB": 11, "HOU": 12, "IND": 13, "JAX": 14, "KC": 15,
    "LV": 16, "LAC": 17, "LA": 18, "MIA": 19, "MIN": 20, "NE": 21, "NO": 22, "NYG": 23,
    "NYJ": 24, "PHI": 25, "PIT": 26, "SF": 27, "SEA": 28, "TB": 29, "TEN": 30, "WAS": 31
}

df_cleaned["away_team"] = df_cleaned["away_team"].replace(team_map)
df_cleaned["home_team"] = df_cleaned["home_team"].replace(team_map)
df_cleaned["posteam"] = df_cleaned["posteam"].replace(team_map)
df_cleaned["defteam"] = df_cleaned["defteam"].replace(team_map)

#now we begin the testing 
#isolate the variables for testing 
y_2026 = df_cleaned["play_type"]
x_2026 = df_cleaned.drop(columns=["play_type"]) 

#test the 2025 data on 2026 
best_accuracy = best_loaded_model.score(x_2026, y_2026)
rival_accuracy = rival_loaded_model.score(x_2026, y_2026)

#start your testing
print("Beginning testing 2025 models on 2026 data...")
start_time = datetime.now() #keep an eye on run times
print("your start time is: ... ", start_time.strftime("%H:%M:%S"))


#%H %M %S  
end_time = datetime.now()
duration = end_time - start_time
print("your end time is: ... ", end_time.strftime("%H:%M:%S"))
print("your testing duration was: ... ", duration)
print("")
print("Test complete")
print("best_loaded_model accuracy = ", best_accuracy)
print("rival_loaded_model accuracy = ", rival_accuracy)

#As of week 4 Thursday October 8th at 08:20 the modles are 
# "best" model on 2025 = 0.6925464717560008 (69.25) 
# "rival" model on 2025 = 0.6927630391626061 (69.28) 
# "best" model on 2026 = 0.690467050543826 (69.05) 
# "rival model" on 2026 = 0.6913627639155471 (69.14) 

#available_weeks = sorted(df['week'].dropna().unique())
#print("Weeks included in 2026 data:", available_weeks)

# maybe to do: would be fun to go through each week and plot which model scores higher 