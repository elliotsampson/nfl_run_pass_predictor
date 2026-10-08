import joblib
import pandas as pd
import nfl_data_py as nfl

#load the new years, ongoing seasons data to test on a new dataset
df = nfl.import_pbp_data(years=[2026])
df.to_csv("nfl_data_2026.csv", index=False)

