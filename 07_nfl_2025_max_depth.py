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
#we will add mdnone_ to the start to name different models
mdnone_run_pass_model = RandomForestClassifier(n_estimators=20, random_state=6)
md2_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=6)
md4_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=6)
md8_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=6)
md16_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=16, random_state=6)
md6_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=6)
md10_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=6)
md12_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=12, random_state=6)
md14_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=14, random_state=6)

#begin training the model 
print("Beginning new training, please wait... ")
mdnone_run_pass_model.fit(x_train, y_train)
md2_run_pass_model.fit(x_train, y_train)
md4_run_pass_model.fit(x_train, y_train)
md8_run_pass_model.fit(x_train, y_train)
md16_run_pass_model.fit(x_train, y_train)
md6_run_pass_model.fit(x_train, y_train)
md10_run_pass_model.fit(x_train, y_train)
md12_run_pass_model.fit(x_train, y_train)
md14_run_pass_model.fit(x_train, y_train)
print("new_run_pass_model has completed training successfully!")

#now we have two types of accuracy, accuracy of the new_run_pass_model on the training data or the test data
#training accuracy
mdnone_train_accuracy = mdnone_run_pass_model.score(x_train, y_train)
md2_train_accuracy = md2_run_pass_model.score(x_train, y_train)
md4_train_accuracy = md4_run_pass_model.score(x_train, y_train)
md8_train_accuracy = md8_run_pass_model.score(x_train, y_train)
md16_train_accuracy = md16_run_pass_model.score(x_train, y_train)
md6_train_accuracy = md6_run_pass_model.score(x_train, y_train)
md10_train_accuracy = md10_run_pass_model.score(x_train, y_train)
md12_train_accuracy = md12_run_pass_model.score(x_train, y_train)
md14_train_accuracy = md14_run_pass_model.score(x_train, y_train)
print("your mdnone_run_pass_model training accuracy is...", mdnone_train_accuracy)
#print("your md2_run_pass_model training accuracy is...", md2_train_accuracy)
#print("your md4_run_pass_model training accuracy is...", md4_train_accuracy)
print("your md8_run_pass_model training accuracy is...", md8_train_accuracy)
#print("your md16_run_pass_model training accuracy is...", md16_train_accuracy)
#print("your md6_run_pass_model training accuracy is...", md6_train_accuracy)
#print("your md10_run_pass_model training accuracy is...", md10_train_accuracy)
print("your md12_run_pass_model training accuracy is...", md12_train_accuracy)
print("your md14_run_pass_model training accuracy is...", md14_train_accuracy)

#testing accuracy
mdnone_test_accuracy = mdnone_run_pass_model.score(x_test, y_test)
md2_test_accuracy = md2_run_pass_model.score(x_test, y_test)
md4_test_accuracy = md4_run_pass_model.score(x_test, y_test)
md8_test_accuracy = md8_run_pass_model.score(x_test, y_test)
md16_test_accuracy = md16_run_pass_model.score(x_test, y_test)
md6_test_accuracy = md6_run_pass_model.score(x_test, y_test)
md10_test_accuracy = md10_run_pass_model.score(x_test, y_test)
md12_test_accuracy = md12_run_pass_model.score(x_test, y_test)
md14_test_accuracy = md14_run_pass_model.score(x_test, y_test)
print("your actual mdnone_run_pass_model test accuracy is...", mdnone_test_accuracy)
print("your actual md2_run_pass_model test accuracy is...", md2_test_accuracy)
print("your actual md4_run_pass_model test accuracy is...", md4_test_accuracy)
print("your actual md8_run_pass_model test accuracy is...", md8_test_accuracy)
print("your actual md16_run_pass_model test accuracy is...", md16_test_accuracy)
print("your actual md6_run_pass_model test accuracy is...", md6_test_accuracy)
print("your actual md10_run_pass_model test accuracy is...", md10_test_accuracy)
print("your actual md12_run_pass_model test accuracy is...", md12_test_accuracy)
print("your actual md14_run_pass_model test accuracy is...", md14_test_accuracy)

#interesting, now we see that the max_depth none has been outperformed on test accuracy 
#order of testing 
#mdnone vs md2 and md4, mdnone won at 0.662575801328328 or 66.26% 
#mdnone vs md8, md8 won at 0.6862546924631822 or 68.63% 
#mdnone vs md16, mdnone won 
#mdnone vs m6 and m10 and m12, m12 won at 0.6924631822119549 or 69.25% 
#md12 vs md14, md12 won 
#md_12 seems to be the best performing, now we can investigate the depth vs the n_estimators 