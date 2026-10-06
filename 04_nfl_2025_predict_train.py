import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#import the data frame called "nfl_data_2025.csv", and take the first 5 nrows from it 
#df = pd.read_csv("nfl_data_2025.csv", nrows=5)
df = pd.read_csv("nfl_data_2025.csv")

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

#.head() gives you the first 10 rows from the cleaned data frame including the headings of those columns 
#print(df_cleaned.head(10))

#now we begin the learning 
#first isolate what we want to predict, pass or run
y = df_cleaned["play_type"]
x = df_cleaned.drop(columns=["play_type"]) 

#now we introduce scikit learn to build the model 
#RandomForrest is a simple-ish decision tree network 
#6 is a number of your choosing for the randomness 
#n_estimator is 20, think of it as the number of expert opinions you have 
run_pass_model = RandomForestClassifier(n_estimators=20, random_state=6)

#Begin training the model 
print("Beginning training, please wait... ")
run_pass_model.fit(x, y)
print("Model has completed training successfully!")

#Now we need to assess the model, how accurate is it? 
accuracy = run_pass_model.score(x, y)
print("your models accuracy is...", accuracy)
# now we have a very accurate model just for our data set. 
# lets move onto splitting the training data so we arent overfitting and we can feed new data to the model 
#this is done in the 05_nfl_2025_predict_test.py file 