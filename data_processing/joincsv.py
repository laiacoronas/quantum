
import pandas as pd
import os


data_dir = "C:\\Users\\lclai\\Desktop\\qm7\\qm7e"

csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]

all_data = []

for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv("traingse.csv", index=False)
