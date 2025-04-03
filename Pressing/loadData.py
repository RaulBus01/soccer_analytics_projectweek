import pandas as pd
from loadFiles import dfs



def simulate_list_of_all_matches_query():
    """Simulates the LIST_OF_ALL_MATCHES SQL query using pandas."""
    df_matches = dfs['matches_df'].copy()
    df_teams = dfs['teams_df'].copy()
    
    # Merge matches with home teams
    matches_home = pd.merge(df_matches, df_teams, left_on='home_team_id', right_on='team_id', how='left')
    
    # Merge the result with away teams
    matches_all = pd.merge(matches_home, df_teams, left_on='away_team_id', right_on='team_id', suffixes=('_home', '_away'), how='left')
    
    # Concatenate team names
    matches_all['matchup'] = matches_all['team_name_home'] + ' vs ' + matches_all['team_name_away']
    
    # Select and order the columns
    result_df = matches_all[['match_id', 'home_score', 'away_score', 'home_team_id', 'away_team_id', 'matchup']].sort_values(by='match_id')
    
    return result_df
def simulate_team_queries(df_player_tracking, df_players, df_teams, match_id, team_id):
    """Simulates the TEAM_QUERIES SQL query using pandas."""
    merged_df = pd.merge(df_player_tracking, df_players, on='player_id')
    merged_df = pd.merge(merged_df, df_teams, on='team_id')
    filtered_df = merged_df[
        (merged_df['game_id'] == match_id) & (merged_df['player_id'] != 'ball') & (merged_df['team_id'] == team_id)
    ]
    result_df = filtered_df[['frame_id', 'timestamp', 'player_id', 'x', 'y', 'team_id']].sort_values(by='timestamp')
    return result_df

def simulate_ball_query(df_player_tracking, df_players, df_teams, match_id):
    """Simulates the BALL_QUERY SQL query using pandas."""
    merged_df = pd.merge(df_player_tracking, df_players, on='player_id')
    merged_df = pd.merge(merged_df, df_teams, on='team_id')
    filtered_df = merged_df[
        (merged_df['game_id'] == match_id) & (merged_df['player_id'] == 'ball') & (merged_df['period_id'] == 1)
        ]
    result_df = filtered_df[['period_id', 'frame_id', 'timestamp', 'x', 'y', 'player_id', 'team_id']].sort_values(by='timestamp')
    return result_df

def simulate_team_query(df_player_tracking, df_players, df_teams, match_id):
    """Simulates the TEAM_QUERY SQL query using pandas."""
    merged_df = pd.merge(df_player_tracking, df_players, on='player_id')
    merged_df = pd.merge(merged_df, df_teams, on='team_id')
    filtered_df = merged_df[(merged_df['game_id'] == match_id) & (merged_df['player_id'] != 'ball')]
    team_ids = filtered_df['team_id'].unique().tolist()
    return team_ids

def simulate_possession_query(df_spadl_actions, df_teams, game_id):
    """Simulates the POSSESION_QUERY using pandas."""

    # action_changes CTE
    df_action_changes = df_spadl_actions[df_spadl_actions['game_id'] == game_id].copy()
    df_action_changes.sort_values(by=['period_id', 'seconds', 'id'], inplace=True)
    df_action_changes['prev_team_id'] = df_action_changes['team_id'].shift(1)
    df_action_changes['next_team_id'] = df_action_changes['team_id'].shift(-1)

    # possession_markers CTE
    df_possession_markers = df_action_changes.copy()
    df_possession_markers['is_new_possession'] = ((df_possession_markers['prev_team_id'].isnull()) | (df_possession_markers['team_id'] != df_possession_markers['prev_team_id'])).astype(int)

    # possession_sequences CTE
    df_possession_sequences = df_possession_markers.copy()
    df_possession_sequences['possession_group'] = df_possession_sequences['is_new_possession'].cumsum()

    # possession_stats CTE
    df_possession_stats = df_possession_sequences.groupby(['possession_group', 'team_id']).agg(
        action_count=('id', 'count'),
        last_action_id=('id', 'max')
    ).reset_index()

    # Final SELECT statement
    df_s = df_possession_sequences.copy()
    df_ps = df_possession_stats.copy()
    
    # Merge DataFrames
    df_merged = pd.merge(df_s, df_ps, left_on=['possession_group', 'team_id'], right_on=['possession_group', 'team_id'])
    df_merged = df_merged[df_merged['id'] == df_merged['last_action_id']]
    df_teams_1 = df_teams.rename(columns={'team_id': 'team_id_losing', 'team_name': 'losing_team'})
    df_teams_2 = df_teams.rename(columns={'team_id': 'team_id_gaining', 'team_name': 'gaining_team'})
    df_merged = pd.merge(df_merged, df_teams_1, left_on='team_id', right_on='team_id_losing')
    df_merged = pd.merge(df_merged, df_teams_2, left_on='next_team_id', right_on='team_id_gaining')

    # Apply WHERE conditions
    df_result = df_merged[
        (df_merged['action_count'] >= 3) &
        (df_merged['team_id'] != df_merged['next_team_id']) &
        (df_merged['next_team_id'].notnull())
    ]

    # Select and order columns
    df_result = df_result[[
        'id', 'period_id', 'seconds', 'action_type', 'team_id', 'next_team_id',
        'losing_team', 'gaining_team', 'start_x', 'start_y', 'end_x', 'end_y', 'action_count'
    ]].sort_values(by=['period_id', 'seconds'])

    return df_result# Example Usage (assuming you have df_player_tracking, df_players, df_teams loaded)

def load_data_local(match_id):
    team_ids = simulate_team_query(dfs['player_tracking_df'], dfs['players_df'], dfs['teams_df'], match_id)
    if len(team_ids) < 2:
        return None, None, None, None, None
    else:
        df_home_simulated = simulate_team_queries(dfs['player_tracking_df'], dfs['players_df'], dfs['teams_df'], match_id, team_ids[0])
        df_away_simulated = simulate_team_queries(dfs['player_tracking_df'], dfs['players_df'], dfs['teams_df'], match_id, team_ids[1])
        df_ball_simulated = simulate_ball_query(dfs['player_tracking_df'], dfs['players_df'], dfs['teams_df'], match_id)
        df_possession_simulated = simulate_possession_query(dfs['spadl_action_df'], dfs['teams_df'], match_id)

        df_possession_first_period = df_possession_simulated[df_possession_simulated['period_id'] == 1]
        df_possession_second_period = df_possession_simulated[df_possession_simulated['period_id'] == 2]
        df_home_simulated['timestamp'] = pd.to_timedelta(df_home_simulated['timestamp']).dt.total_seconds().astype(float)
        df_away_simulated['timestamp'] = pd.to_timedelta(df_away_simulated['timestamp']).dt.total_seconds().astype(float)
        df_ball_simulated['timestamp'] = pd.to_timedelta(df_ball_simulated['timestamp']).dt.total_seconds().astype(float)
        print("Loading data from local files")
        return df_ball_simulated,df_home_simulated, df_away_simulated, df_possession_first_period, df_possession_second_period
