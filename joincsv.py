
import pandas as pd
import os


data_dir = "\\Users\\lclai\\Desktop\\qm7\\qm7formulas"

csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".csv")]

all_data = []

for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv("combined_data.csv", index=False)
