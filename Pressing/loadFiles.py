import pandas as pd
import os

# Path where Parquet files are stored
DATA_FOLDER = 'raw'

def load_dataframes():
    """
    Load all tables from Parquet files into Pandas DataFrames.
    """
    tables = ["eventtypes_df", "matches_df", "matchevents_df", "player_tracking_df", "players_df",
              "qualifiers_df", "qualifiertypes_df", "spadl_action_df", "teams_df"]
    
    dfs = {}
    for table in tables:
        file_path = os.path.join(DATA_FOLDER, f"{table}.parquet")
        if os.path.exists(file_path):
            dfs[table] = pd.read_parquet(file_path)
        else:
            print(f"Warning: {file_path} not found.")
    
    return dfs

# Load all data
dfs = load_dataframes()

