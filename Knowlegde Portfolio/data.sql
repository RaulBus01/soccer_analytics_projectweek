CREATE TABLE eventtypes (
    eventtype_id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY,
    match_date TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER
);

CREATE TABLE matchevents (
    match_id INTEGER,
    event_id INTEGER,
    eventtype_id INTEGER,
    result TEXT,
    success INTEGER,
    period_id INTEGER,
    timestamp TEXT,
    end_timestamp TEXT,
    ball_state TEXT,
    ball_owning_team INTEGER,
    team_id INTEGER,
    player_id INTEGER,
    x INTEGER,
    y INTEGER,
    end_coordinates_x INTEGER,
    end_coordinates_y INTEGER,
    receiver_player_id INTEGER
);

CREATE TABLE player_tracking (
    id INTEGER PRIMARY KEY,
    game_id INTEGER,
    frame_id INTEGER,
    timestamp TEXT,
    period_id INTEGER,
    player_id INTEGER,
    x INTEGER,
    y INTEGER
);

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT,
    team_id INTEGER,
    jersey_number INTEGER
);

CREATE TABLE qualifiers (
    match_id INTEGER,
    event_id INTEGER,
    qualifier_type_id INTEGER,
    qualifier_value TEXT
);

CREATE TABLE qualifiertypes (
    qualifier_id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE spadl_actions (
    id INTEGER PRIMARY KEY,
    game_id INTEGER,
    period_id INTEGER,
    seconds INTEGER,
    player_id INTEGER,
    team_id INTEGER,
    start_x INTEGER,
    start_y INTEGER,
    end_x INTEGER,
    end_y INTEGER,
    action_type TEXT,
    result TEXT,
    bodypart TEXT
);

CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT
);
