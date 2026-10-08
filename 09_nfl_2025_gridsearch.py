import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#gird search introduce 
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import numpy as np

#import the data frame called "nfl_data_2025.csv", 
#low_memory=False, basically lets it read the file before asssuming types
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
#begin modelling
#RandomForrest is a simple-ish decision tree network 
#6 is a number of your choosing for the randomness (keep this the same)
#this time we are going to do look at the n_estimators on accuracy 
#we will add mdnone_ to the start to name different models
#we reintroduce the ne20 factor into naming
ne50_md10_run_pass_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=6)

#begin training the model 
print("Beginning new training, please wait... ")
ne50_md10_run_pass_model.fit(x_train, y_train)
print("new_run_pass_model has completed training successfully!")

#now we have two types of accuracy, accuracy of the new_run_pass_model on the training data or the test data
#training accuracy
ne50_md10_train_accuracy = ne50_md10_run_pass_model.score(x_train, y_train)
print("your ne50_md10_run_pass_model training accuracy is...", ne50_md10_train_accuracy)

#testing accuracy
ne50_md10_test_accuracy = ne50_md10_run_pass_model.score(x_test, y_test)
print("your actual ne20_md12_run_pass_model test accuracy is...", ne50_md10_test_accuracy)

# ne40_md10 = 0.6923187987294254 or 69.23% 
# ne50_md10 = 0.6928963326595438 or 69.29% (new best)
# ne60_md10 = 0.6927519491770141 or 69.28% (worse)

#Grid search
#make a base model that will be optimised.
base_run_pass_model = RandomForestClassifier(random_state=6)
 
md_range = [9, 10, 11]
n_est_range = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
parameters = {
    "max_depth" : md_range, 
    "n_estimators" : n_est_range} 
search = GridSearchCV(base_run_pass_model, parameters, cv=5) 


print("Beginning automated search across your regions...")
start_time = datetime.now()
print("your start time is: ... ", start_time.strftime("%H:%M:%S"))
search.fit(x_train, y_train)
#%H %M %S  
end_time = datetime.now()
duration = end_time - start_time
print("")
print("Search complete")
print("your end time is: ... ", end_time.strftime("%H:%M:%S"))
print("your search duration was: ... ", duration)
print("")
print("your optimised parameters are: ",  search.best_params_)
print("your optimised parameters give an accuracy of: ", search.best_score_)

#now we have conclusively found that 
# {'max_depth': 10, 'n_estimators': 55} is our best model, with 0.694462728449621 or 69.45% 

#lets save this model so we can play predictions with it 
import joblib 
best_run_pass_model = search.best_estimator_
model_filename = "best_run_pass_model.pkl"
joblib.dump(best_run_pass_model, model_filename)
print("your best_run_pass_model has been saved as... ", model_filename)