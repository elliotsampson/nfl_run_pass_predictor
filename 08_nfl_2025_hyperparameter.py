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
#we reintroduce the ne20 factor into naming
ne20_md12_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=12, random_state=6)
ne50_md12_run_pass_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=6)
ne100_md12_run_pass_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=6)
ne10_md12_run_pass_model = RandomForestClassifier(n_estimators=10, max_depth=12, random_state=6)
ne30_md12_run_pass_model = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=6)
ne10_md10_run_pass_model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=6)
ne20_md10_run_pass_model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=6)
ne30_md10_run_pass_model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=6)
ne40_md10_run_pass_model = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=6)
ne50_md10_run_pass_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=6)
ne60_md10_run_pass_model = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=6)
ne80_md10_run_pass_model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=6)


#begin training the model 
print("Beginning new training, please wait... ")
ne20_md12_run_pass_model.fit(x_train, y_train)
ne50_md12_run_pass_model.fit(x_train, y_train)
ne100_md12_run_pass_model.fit(x_train, y_train)
ne10_md12_run_pass_model.fit(x_train, y_train)
ne30_md12_run_pass_model.fit(x_train, y_train)
ne10_md10_run_pass_model.fit(x_train, y_train)
ne20_md10_run_pass_model.fit(x_train, y_train)
ne30_md10_run_pass_model.fit(x_train, y_train)
ne40_md10_run_pass_model.fit(x_train, y_train)
ne50_md10_run_pass_model.fit(x_train, y_train)
ne60_md10_run_pass_model.fit(x_train, y_train)
ne80_md10_run_pass_model.fit(x_train, y_train)
print("new_run_pass_model has completed training successfully!")

#now we have two types of accuracy, accuracy of the new_run_pass_model on the training data or the test data
#training accuracy
ne20_md12_train_accuracy = ne20_md12_run_pass_model.score(x_train, y_train)
ne50_md12_train_accuracy = ne50_md12_run_pass_model.score(x_train, y_train)
ne100_md12_train_accuracy = ne100_md12_run_pass_model.score(x_train, y_train)
ne10_md12_train_accuracy = ne10_md12_run_pass_model.score(x_train, y_train)
ne30_md12_train_accuracy = ne30_md12_run_pass_model.score(x_train, y_train)
ne10_md10_train_accuracy = ne10_md10_run_pass_model.score(x_train, y_train)
ne20_md10_train_accuracy = ne20_md10_run_pass_model.score(x_train, y_train)
ne30_md10_train_accuracy = ne30_md10_run_pass_model.score(x_train, y_train)
ne40_md10_train_accuracy = ne40_md10_run_pass_model.score(x_train, y_train)
ne50_md10_train_accuracy = ne50_md10_run_pass_model.score(x_train, y_train)
ne60_md10_train_accuracy = ne50_md10_run_pass_model.score(x_train, y_train)
ne80_md10_train_accuracy = ne50_md10_run_pass_model.score(x_train, y_train)
#print("your ne20_md12_run_pass_model training accuracy is...", ne20_md12_train_accuracy)
#print("your ne50_md12_run_pass_model training accuracy is...", ne50_md12_train_accuracy)
#print("your ne100_md12_run_pass_model training accuracy is...", ne100_md12_train_accuracy)
#print("your ne10_md12_run_pass_model training accuracy is...", ne10_md12_train_accuracy)
#print("your ne30_md12_run_pass_model training accuracy is...", ne30_md12_train_accuracy)
#print("your ne10_md10_run_pass_model training accuracy is...", ne10_md10_train_accuracy)
#print("your ne20_md10_run_pass_model training accuracy is...", ne20_md10_train_accuracy)
#print("your ne30_md10_run_pass_model training accuracy is...", ne30_md10_train_accuracy)
#print("your ne40_md10_run_pass_model training accuracy is...", ne40_md10_train_accuracy)
#print("your ne50_md10_run_pass_model training accuracy is...", ne50_md10_train_accuracy)
#print("your ne60_md10_run_pass_model training accuracy is...", ne60_md10_train_accuracy)
#print("your ne80_md10_run_pass_model training accuracy is...", ne80_md10_train_accuracy)

#testing accuracy
ne20_md12_test_accuracy = ne20_md12_run_pass_model.score(x_test, y_test)
ne50_md12_test_accuracy = ne50_md12_run_pass_model.score(x_test, y_test)
ne100_md12_test_accuracy = ne100_md12_run_pass_model.score(x_test, y_test)
ne10_md12_test_accuracy = ne10_md12_run_pass_model.score(x_test, y_test)
ne30_md12_test_accuracy = ne30_md12_run_pass_model.score(x_test, y_test)
ne10_md10_test_accuracy = ne10_md10_run_pass_model.score(x_test, y_test)
ne30_md10_test_accuracy = ne30_md10_run_pass_model.score(x_test, y_test)
ne20_md10_test_accuracy = ne20_md10_run_pass_model.score(x_test, y_test)
ne40_md10_test_accuracy = ne40_md10_run_pass_model.score(x_test, y_test)
ne50_md10_test_accuracy = ne50_md10_run_pass_model.score(x_test, y_test)
ne60_md10_test_accuracy = ne60_md10_run_pass_model.score(x_test, y_test)
ne80_md10_test_accuracy = ne80_md10_run_pass_model.score(x_test, y_test)
print("your actual ne20_md12_run_pass_model test accuracy is...", ne20_md12_test_accuracy)
print("your actual ne50_md12_run_pass_model test accuracy is...", ne50_md12_test_accuracy)
print("your actual ne100_md12_run_pass_model test accuracy is...", ne100_md12_test_accuracy)
print("your actual ne10_md12_run_pass_model test accuracy is...", ne10_md12_test_accuracy)
print("your actual ne30_md12_run_pass_model test accuracy is...", ne30_md12_test_accuracy)
print("your actual ne10_md10_run_pass_model test accuracy is...", ne10_md10_test_accuracy)
print("your actual ne20_md10_run_pass_model test accuracy is...", ne20_md10_test_accuracy)
print("your actual ne30_md10_run_pass_model test accuracy is...", ne30_md10_test_accuracy)
print("your actual ne40_md10_run_pass_model test accuracy is...", ne40_md10_test_accuracy)
print("your actual ne50_md10_run_pass_model test accuracy is...", ne50_md10_test_accuracy)
print("your actual ne60_md10_run_pass_model test accuracy is...", ne60_md10_test_accuracy)
print("your actual ne80_md10_run_pass_model test accuracy is...", ne80_md10_test_accuracy)

#interesting, now we see that the max_depth none has been outperformed on test accuracy 
#md_12 seems to be the best performing, now we can investigate the depth vs the n_estimators 
#order of testing 
#current best model is 69.25% which is ne20_md12 
# ne50_md12_run_pass_model test accuracy is... 0.6910193473866589 or 69.10% (worse)
# ne100_md12_run_pass_model test accuracy is... 0.6898642795264222 or 68.97% (worse)
# now we try 10 20 and 30 
# your actual ne10_md12_run_pass_model test accuracy is... 0.684522090672827 or 68.45% (worse)
# your actual ne30_md12_run_pass_model test accuracy is... 0.6904418134565405 or 69.04% (worse)
# now we try ne10_md10, ne20_md10, ne30_md10, ne40_md10, ne50_md10
# ne10_md10 = 0.6862546924631822 or 68.63% 
# ne20_md10 = 0.6911637308691886 or 69.12% 
# ne30_md10 = 0.6913081143517181 or 69.13% 
# ne40_md10 = 0.6923187987294254 or 69.23% 
# ne50_md10 = 0.6928963326595438 or 69.29% (new best)
# now ne60_md10, ne80_md10
# ne60_md10 = 0.6927519491770141 or 69.28% (worse)
# ne80_md10 = 0.6923187987294254 or 69.23% (worse)

# now we compare how well we have done manually with an automated search in 09_nfl_2025_gridsearch
