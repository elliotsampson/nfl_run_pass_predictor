import pandas as pd

#in this file, I open a csv file based on the nfl_data from 2025 and check the data is there

#import the data frame called "nfl_data_2025.csv", and take the first 5 nrows from it 
df = pd.read_csv("nfl_data_2025.csv", nrows=5)
 

#list all the possible columns needed for a basic prediction model
columns_needed = df[[
    "away_team", "home_team", "posteam", "defteam", "game_seconds_remaining", "game_half", "yardline_100", "down", 
    "ydstogo", "qtr", "score_differential", "play_type"
]]

#print all columns needed
print(columns_needed) 

