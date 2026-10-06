import nfl_data_py as nfl

#in this file, I download a specific year of data from the nfl_data_py library and turn it into a csv 

print("Downloading 2025 play-by-play data...")
# 1. Grab the 2025 data from the internet
df = nfl.import_pbp_data(years=[2025])

print("Saving to your project folder...")
# 2. Save it as a 2025 CSV file physically on your machine
df.to_csv("nfl_data_2025.csv", index=False)

print("Done!")