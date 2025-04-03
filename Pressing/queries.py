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
SELECT DISTINCT
    p.team_id,
    t.team_name
FROM 
    player_tracking pt
JOIN 
    players p ON pt.player_id = p.player_id
JOIN 
    teams t ON p.team_id = t.team_id
WHERE 
    pt.game_id = %s 
    AND p.player_name != 'Ball'
ORDER BY 
    p.team_id;
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
POSSESION_CHANGE_SUMMARY = """
    WITH action_changes AS (
    SELECT
        a.*,
        LAG(a.team_id) OVER (PARTITION BY a.game_id ORDER BY a.period_id, a.seconds, a.id) AS prev_team_id,
        LEAD(a.team_id) OVER (PARTITION BY a.game_id ORDER BY a.period_id, a.seconds, a.id) AS next_team_id
    FROM spadl_actions a
),
possession_markers AS (
    SELECT
        *,
        CASE 
            WHEN prev_team_id IS NULL OR team_id != prev_team_id 
            THEN 1 
            ELSE 0 
        END AS is_new_possession
    FROM action_changes
),
possession_sequences AS (
    SELECT
        *,
        SUM(is_new_possession) OVER (PARTITION BY game_id ORDER BY period_id, seconds, id) AS possession_group
    FROM possession_markers
),
possession_stats AS (
    SELECT
        game_id,
        possession_group,
        team_id,
        COUNT(*) AS action_count,
        MAX(id) AS last_action_id
    FROM possession_sequences
    GROUP BY game_id, possession_group, team_id
),
possession_events AS (
    SELECT 
        s.game_id,
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
    FROM possession_sequences s
    JOIN possession_stats ps 
        ON s.game_id = ps.game_id 
       AND s.possession_group = ps.possession_group 
       AND s.team_id = ps.team_id
       AND s.id = ps.last_action_id
    JOIN teams t1 ON s.team_id = t1.team_id
    JOIN teams t2 ON s.next_team_id = t2.team_id
    WHERE ps.action_count >= 3
      AND s.team_id != s.next_team_id
      AND s.next_team_id IS NOT NULL
),
match_level AS (
    SELECT 
        game_id,
        team_id,
        team_name,
        AVG(start_x) AS avg_start_x,
        AVG(start_y) AS avg_start_y,
        SUM(lost) AS lost_count,
        SUM(gained) AS gained_count,
        SUM(action_count) AS total_actions,
        COUNT(*) AS total_events
    FROM (
        SELECT 
            game_id,
            losing_team_id AS team_id,
            losing_team AS team_name,
            start_x,
            start_y,
            action_count,
            1 AS lost,
            0 AS gained
        FROM possession_events
        UNION ALL
        SELECT 
            game_id,
            gaining_team_id AS team_id,
            gaining_team AS team_name,
            start_x,
            start_y,
            action_count,
            0 AS lost,
            1 AS gained
        FROM possession_events
    ) sub
    GROUP BY game_id, team_id, team_name
),
lost_zones AS (
    SELECT
        game_id,
        losing_team_id AS team_id,
        CASE 
            WHEN start_x < 33 THEN 'defending'
            WHEN start_x < 66 THEN 'midfield'
            ELSE 'attack'
        END AS lost_zone,
        AVG(start_x) AS avg_start_x,
        AVG(start_y) AS avg_start_y,
        COUNT(*) AS lost_count
    FROM possession_events
    GROUP BY game_id, losing_team_id,
        CASE 
            WHEN start_x < 33 THEN 'defending'
            WHEN start_x < 66 THEN 'midfield'
            ELSE 'attack'
        END
),
match_lost_zones AS (
    SELECT
        team_id,
        lost_zone,
        AVG(lost_count) AS avg_lost,
        AVG(avg_start_x) AS avg_start_x_zone,
        AVG(avg_start_y) AS avg_start_y_zone
    FROM lost_zones
    GROUP BY team_id, lost_zone
),
pivot_lost_zones AS (
    SELECT 
        team_id,
        MAX(CASE WHEN lost_zone = 'defending' THEN avg_lost ELSE 0 END) AS avg_lost_defending,
        MAX(CASE WHEN lost_zone = 'midfield' THEN avg_lost ELSE 0 END) AS avg_lost_midfield,
        MAX(CASE WHEN lost_zone = 'attack' THEN avg_lost ELSE 0 END) AS avg_lost_attack,
        MAX(CASE WHEN lost_zone = 'defending' THEN avg_start_x_zone ELSE 0 END) AS avg_start_x_defending,
        MAX(CASE WHEN lost_zone = 'midfield' THEN avg_start_x_zone ELSE 0 END) AS avg_start_x_midfield,
        MAX(CASE WHEN lost_zone = 'attack' THEN avg_start_x_zone ELSE 0 END) AS avg_start_x_attack,
        MAX(CASE WHEN lost_zone = 'defending' THEN avg_start_y_zone ELSE 0 END) AS avg_start_y_defending,
        MAX(CASE WHEN lost_zone = 'midfield' THEN avg_start_y_zone ELSE 0 END) AS avg_start_y_midfield,
        MAX(CASE WHEN lost_zone = 'attack' THEN avg_start_y_zone ELSE 0 END) AS avg_start_y_attack
    FROM match_lost_zones
    GROUP BY team_id
)
SELECT 
    ml.team_id, 
    ml.team_name,
    ROUND(AVG(ml.avg_start_x)::numeric, 2) AS avg_start_x,
    ROUND(AVG(ml.avg_start_y)::numeric, 2) AS avg_start_y,
    ROUND(AVG(ml.lost_count)::numeric, 2) AS avg_lost_possessions_per_game,
    ROUND(AVG(ml.gained_count)::numeric, 2) AS avg_gained_possessions_per_game,
    ROUND(AVG(ml.total_actions)::numeric, 2) AS avg_total_actions,
    ROUND(AVG(ml.total_events)::numeric, 2) AS avg_total_events,
    ROUND(plz.avg_lost_defending::numeric, 2) AS avg_lost_defending,
    ROUND(plz.avg_lost_midfield::numeric, 2) AS avg_lost_midfield,
    ROUND(plz.avg_lost_attack::numeric, 2) AS avg_lost_attack,
    ROUND(plz.avg_start_x_defending::numeric, 2) AS avg_start_x_defending,
    ROUND(plz.avg_start_y_defending::numeric, 2) AS avg_start_y_defending,
    ROUND(plz.avg_start_x_midfield::numeric, 2) AS avg_start_x_midfield,
    ROUND(plz.avg_start_y_midfield::numeric, 2) AS avg_start_y_midfield,
    ROUND(plz.avg_start_x_attack::numeric, 2) AS avg_start_x_attack,
    ROUND(plz.avg_start_y_attack::numeric, 2) AS avg_start_y_attack
FROM match_level ml
LEFT JOIN pivot_lost_zones plz
     ON ml.team_id = plz.team_id
GROUP BY ml.team_id, ml.team_name, plz.avg_lost_defending, plz.avg_lost_midfield, plz.avg_lost_attack, 
         plz.avg_start_x_defending, plz.avg_start_y_defending, plz.avg_start_x_midfield, plz.avg_start_y_midfield, 
         plz.avg_start_x_attack, plz.avg_start_y_attack
ORDER BY ml.team_name;

    """


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
# Function to get all available matches
def get_all_matches():
    conn = get_connection()
    
    matches_df = pd.read_sql_query(LIST_OF_ALL_MATCHES, conn)
    conn.close()
    return matches_df
    

# Load data from the database
def load_basic_data(match_id):

    conn = get_connection()
    df_ball = pd.read_sql_query(BALL_QUERY, conn, params=(match_id,))
    df_teams = pd.read_sql_query(TEAM_QUERIES, conn, params=(match_id,))
    df_teams['timestamp'] = pd.to_timedelta(df_teams['timestamp']).dt.total_seconds().astype(float)
    df_ball['timestamp'] = pd.to_timedelta(df_ball['timestamp']).dt.total_seconds().astype(float)
    conn.close()
    return df_ball,df_teams
def load_possession_data(match_id):
    conn = get_connection()
    df_possesion = pd.read_sql_query(POSSESION_QUERY, conn, params=(match_id,))
    conn.close()
    return df_possesion

def load_possession_summary():
    conn = get_connection()
    df_possesion_summary = pd.read_sql_query(POSSESION_CHANGE_SUMMARY, conn,)
    conn.close()
    return df_possesion_summary

def load_spadl(match_id):
    conn = get_connection()
    df_spadl = pd.read_sql_query(SPADL_QUERY, conn, params=(match_id, match_id))
    df_spadl['timestamp'] = pd.to_timedelta(df_spadl['timestamp']).dt.total_seconds().astype(float)
    conn.close()
    return df_spadl



