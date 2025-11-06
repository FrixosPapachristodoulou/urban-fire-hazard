import pandas as pd

# Paths to your two CSV files
file1 = "data/raw/lfb_fire_data/lfb_incident_data_2024_1.csv"
file2 = "data/raw/lfb_fire_data/lfb_incident_data_2024_2.csv"

# Read both CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine them (stack rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new CSV file
combined_df.to_csv("data/raw/lfb_fire_data/lfb_incident_data_2024.csv", index=False)

print("✅ Files combined successfully into 'lfb_incident_data_2024.csv'")
