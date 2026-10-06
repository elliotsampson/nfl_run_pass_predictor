import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from datetime import datetime

#import the data frame called "nfl_data_2025.csv", 
df = pd.read_csv("nfl_data_2025.csv", low_memory=False)
#list the column headings you need from the csv
columns_needed = [
    "away_team", "home_team", "posteam", "defteam", "game_seconds_remaining", "game_half", "yardline_100", "down", 
    "ydstogo", "qtr", "score_differential", "play_type"
]

#use pandas to copy only the columns needed 
df_filtered = df[columns_needed].copy()

#now remove the kick punt type of plays from the list
df_cleaned = df_filtered[df_filtered["play_type"].isin(["pass", "run"])].copy() 

#now we want to use .map to make the game half and play type to be numerical, so later when modelling its easier 
#question we ask, was it a pass? 0 = run, 1 = pass 
df_cleaned["game_half"] = df_cleaned["game_half"].map({"Half1": 1, "Half2": 2}) 
df_cleaned["play_type"] = df_cleaned["play_type"].map({"run": 0, "pass": 1}) 

#now we want to use .map to get rid of all the team names in the other columns 
team_map = {
    "ARI": 0, "ATL": 1, "BAL": 2, "BUF": 3, "CAR": 4, "CHI": 5, "CIN": 6, "CLE": 7,
    "DAL": 8, "DEN": 9, "DET": 10, "GB": 11, "HOU": 12, "IND": 13, "JAX": 14, "KC": 15,
    "LV": 16, "LAC": 17, "LA": 18, "MIA": 19, "MIN": 20, "NE": 21, "NO": 22, "NYG": 23,
    "NYJ": 24, "PHI": 25, "PIT": 26, "SF": 27, "SEA": 28, "TB": 29, "TEN": 30, "WAS": 31
}
#cleaning the data 
df_cleaned["away_team"] = df_cleaned["away_team"].replace(team_map)
df_cleaned["home_team"] = df_cleaned["home_team"].replace(team_map)
df_cleaned["posteam"] = df_cleaned["posteam"].replace(team_map)
df_cleaned["defteam"] = df_cleaned["defteam"].replace(team_map)

#now we begin the learning 
#first isolate what we want to predict, pass or run
y = df_cleaned["play_type"]
x = df_cleaned.drop(columns=["play_type"]) 

#split the training data for the model 
# x is our paramters we are training on, y is the output run or pass 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=6)

#Grid search
#make a base model that will be optimised.
base_run_pass_model = RandomForestClassifier(random_state=6)

#specify the parameters you want to optimse over  
md_range = [9, 10, 11]
n_est_range = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
parameters = {
    "max_depth" : md_range, 
    "n_estimators" : n_est_range} 
#name your search
search = GridSearchCV(base_run_pass_model, parameters, cv=5) 

#start your search
print("Beginning automated search across your regions...")
start_time = datetime.now() #keep an eye on run times
print("your start time is: ... ", start_time.strftime("%H:%M:%S"))
search.fit(x_train, y_train)
#%H %M %S  
end_time = datetime.now()
duration = end_time - start_time
print("your end time is: ... ", end_time.strftime("%H:%M:%S"))
print("your search duration was: ... ", duration)
print("")
print("Search complete")
print("your optimised parameters are: ",  search.best_params_)
print("your optimised parameters give an accuracy of: ", search.best_score_)

#lets save this model so we can play predictions with it 
best_run_pass_model = search.best_estimator_
model_filename = "best_run_pass_model.pkl"
joblib.dump(best_run_pass_model, model_filename)
print("your best_run_pass_model has been saved as... ", model_filename)

# in the next file 10_nfl_2025_scenarios.py we manually give the model scenarios, and it should predict the pass or run outcome 
# we will test this with 2024 data that it has never seen 