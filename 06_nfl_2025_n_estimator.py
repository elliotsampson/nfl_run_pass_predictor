import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
#we will add n20_ to the start to name different models
n10_run_pass_model = RandomForestClassifier(n_estimators=10, random_state=6)
n20_run_pass_model = RandomForestClassifier(n_estimators=20, random_state=6)
n50_run_pass_model = RandomForestClassifier(n_estimators=50, random_state=6)
n100_run_pass_model = RandomForestClassifier(n_estimators=100, random_state=6)
#begin training the model 
print("Beginning new training, please wait... ")
n10_run_pass_model.fit(x_train, y_train)
n20_run_pass_model.fit(x_train, y_train)
n50_run_pass_model.fit(x_train, y_train)
n100_run_pass_model.fit(x_train, y_train)
print("new_run_pass_model has completed training successfully!")

#now we have two types of accuracy, accuracy of the new_run_pass_model on the training data or the test data
#training accuracy
n10_train_accuracy = n10_run_pass_model.score(x_train, y_train)
n20_train_accuracy = n20_run_pass_model.score(x_train, y_train)
n50_train_accuracy = n50_run_pass_model.score(x_train, y_train)
n100_train_accuracy = n100_run_pass_model.score(x_train, y_train)
print("your n10_run_pass_model training accuracy is...", n10_train_accuracy)
print("your n20_run_pass_model training accuracy is...", n20_train_accuracy)
print("your n50_run_pass_model training accuracy is...", n50_train_accuracy)
print("your n100_run_pass_model training accuracy is...", n100_train_accuracy)
#testing accuracy
n10_test_accuracy = n10_run_pass_model.score(x_test, y_test)
n20_test_accuracy = n20_run_pass_model.score(x_test, y_test)
n50_test_accuracy = n50_run_pass_model.score(x_test, y_test)
n100_test_accuracy = n100_run_pass_model.score(x_test, y_test)
print("your actual n10_run_pass_model test accuracy is...", n10_test_accuracy)
print("your actual n20_run_pass_model test accuracy is...", n20_test_accuracy)
print("your actual n50_run_pass_model test accuracy is...", n50_test_accuracy)
print("your actual n100_run_pass_model test accuracy is...", n100_test_accuracy)

#from this we conclude that there is little difference between more estimators, 0.23% for longer compute isnt ideal 
# n10_run_pass_model test accuracy is... 0.646404851285013 or 64.64% 
# n20_run_pass_model test accuracy is... 0.662575801328328 or 66.26% 
# n50_run_pass_model test accuracy is... 0.6648859370488016 or 66.49%
# n100_run_pass_model test accuracy is... 0.6677736066993936 or 66.78%  
#next we look at the max_depth for the trees in the 07_nfl_2025_max_depth.py 