import psycopg2
import dotenv
import os
import pandas as pd

# Load environment variables
dotenv.load_dotenv()

# Database connection parameters
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_USER = os.getenv("PG_USER")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DATABASE = os.getenv("PG_DB")

# Connect to the database
def get_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
        sslmode="require",
    )

# SQL queries
# PLEASE MAKE IT CUSTOMIZABLE SOON HAL THIS IS BAD
BALL_QUERY = """
SELECT pt.period_id, pt.frame_id, pt.timestamp, pt.x, pt.y, pt.player_id, p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id
WHERE pt.game_id = %s AND p.player_id = 'ball' AND pt.period_id = 1
ORDER BY timestamp;
"""

TEAM_QUERY = """
SELECT DISTINCT p.team_id
FROM player_tracking pt
JOIN players p ON pt.player_id = p.player_id
JOIN teams t ON p.team_id = t.team_id AND p.player_id != 'ball'
WHERE pt.game_id = %s
"""

TEAM_QUERIES = """
SELECT 
        pt.frame_id,
        pt.timestamp,
        pt.period_id,
        pt.player_id,
        p.player_name,
        p.jersey_number,
        p.team_id,
        pt.x,
        pt.y
    FROM 
        player_tracking pt
    JOIN 
        players p ON pt.player_id = p.player_id
    WHERE 
        pt.game_id = %s
        AND p.player_name != 'Ball'
    ORDER BY 
        pt.frame_id, p.team_id
    """

LIST_OF_ALL_MATCHES = """
SELECT 
    m.match_id,
    m.home_score, 
    m.away_score, 
    m.home_team_id,
    m.away_team_id,
    CONCAT(t1.team_name, ' vs ', t2.team_name) AS matchup
FROM matches m
JOIN teams t1 ON m.home_team_id = t1.team_id
JOIN teams t2 ON m.away_team_id = t2.team_id
ORDER BY m.match_id;
"""
GET_SPECIFIC_MATCH = """
SELECT 
    m.match_id,
    m.home_score, 
    m.away_score, 
    m.home_team_id,
    m.away_team_id,
    CONCAT(t1.team_name, ' vs ', t2.team_name) AS matchup
FROM matches m
JOIN teams t1 ON m.home_team_id = t1.team_id
JOIN teams t2 ON m.away_team_id = t2.team_id
WHERE m.match_id = '%s'
ORDER BY m.match_id;
"""
# Function to get all available matches
def get_all_matches():
    conn = get_connection()
    
    matches_df = pd.read_sql_query(LIST_OF_ALL_MATCHES, conn)
    conn.close()
    return matches_df

POSSESION_QUERY = """
    WITH action_changes AS (
        SELECT
            a.*,
            LAG(a.team_id) OVER (ORDER BY a.period_id, a.seconds, a.id) AS prev_team_id,
            LEAD(a.team_id) OVER (ORDER BY a.period_id, a.seconds, a.id) AS next_team_id
        FROM
            spadl_actions a
        WHERE
            a.game_id = %s
    ),
    possession_markers AS (
        SELECT
            *,
            CASE 
                WHEN prev_team_id IS NULL OR team_id != prev_team_id 
                THEN 1 
                ELSE 0 
            END AS is_new_possession
        FROM
            action_changes
    ),
    possession_sequences AS (
        SELECT
            *,
            SUM(is_new_possession) OVER (ORDER BY period_id, seconds, id) AS possession_group
        FROM
            possession_markers
    ),
    possession_stats AS (
        SELECT
            possession_group,
            team_id,
            COUNT(*) AS action_count,
            MAX(id) AS last_action_id
        FROM
            possession_sequences
        GROUP BY
            possession_group, team_id
    )
    SELECT 
        s.id,
        s.period_id,
        s.seconds,
        s.action_type,
        s.team_id AS losing_team_id,
        s.next_team_id AS gaining_team_id,
        t1.team_name AS losing_team,
        t2.team_name AS gaining_team,
        s.start_x,
        s.start_y,
        s.end_x,
        s.end_y,
        ps.action_count
    FROM
        possession_sequences s
    JOIN
        possession_stats ps ON s.possession_group = ps.possession_group 
                             AND s.team_id = ps.team_id
                             AND s.id = ps.last_action_id
    JOIN teams t1 ON s.team_id = t1.team_id
    JOIN teams t2 ON s.next_team_id = t2.team_id
    WHERE
        ps.action_count >= 3
        AND s.team_id != s.next_team_id
        AND s.next_team_id IS NOT NULL
    ORDER BY
        s.period_id, s.seconds
    """
SPADL_QUERY = """
SELECT  
    p.player_id,
    p.player_name, 
    pt.frame_id, 
    pt.x, 
    pt.y, 
    t.team_name, 
    sa.action_type, 
    sa.bodypart,
    pt.timestamp,
    sa.seconds,
    EXTRACT(EPOCH FROM pt.timestamp::interval) AS timestamp_seconds
FROM 
    spadl_actions sa
INNER JOIN 
    players p ON p.player_id = sa.player_id
INNER JOIN 
    player_tracking pt ON p.player_id = pt.player_id
INNER JOIN 
    teams t ON t.team_id = p.team_id
WHERE 
    pt.game_id = %s 
    AND sa.game_id = %s
    AND EXTRACT(EPOCH FROM pt.timestamp::interval) = sa.seconds
GROUP BY
    p.player_id,
    p.player_name, 
    pt.frame_id, 
    pt.x, 
    pt.y, 
    t.team_name, 
    sa.action_type, 
    sa.bodypart,
    pt.timestamp,
    sa.seconds;
"""

    

# Load data from the database
def load_data(match_id):
    try:
     conn = get_connection()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None, None, None
    # Add the tuple parentheses around match_id - this is what's missing
    team_ids_df = pd.read_sql_query(TEAM_QUERY, conn, params=(match_id,))
    team_ids = team_ids_df['team_id'].tolist()
    print(team_ids)
    if len(team_ids) < 2:
        conn.close()
        return None, None, None  # No data available for this match
    

    df_ball = pd.read_sql_query(BALL_QUERY, conn, params=(match_id,))
    df_teams = pd.read_sql_query(TEAM_QUERIES, conn, params=(match_id,))
    df_home = df_teams[df_teams['team_id'] == team_ids[0]]
    df_away = df_teams[df_teams['team_id'] == team_ids[1]]
    df_actions_label = pd.read_sql_query(SPADL_QUERY, conn, params=(match_id, match_id))
    df_possesion = pd.read_sql_query(POSSESION_QUERY, conn, params=(match_id,))
    print(df_actions_label.head())
    print(df_home.head())
    print(df_away.head())

    df_possesion_first_period = df_possesion[df_possesion['period_id'] == 1]
    df_possesion_second_period = df_possesion[df_possesion['period_id'] == 2]
    df_home['timestamp'] = pd.to_timedelta(df_home['timestamp']).dt.total_seconds().astype(float)
    df_away['timestamp'] = pd.to_timedelta(df_away['timestamp']).dt.total_seconds().astype(float)
    df_ball['timestamp'] = pd.to_timedelta(df_ball['timestamp']).dt.total_seconds().astype(float)
    df_actions_label['timestamp'] = pd.to_timedelta(df_actions_label['timestamp']).dt.total_seconds().astype(float)
    print(df_actions_label.head())

    conn.close()
    return df_ball, df_teams, df_home, df_away, df_actions_label, df_possesion_first_period, df_possesion_second_period



