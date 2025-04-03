import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from mplsoccer import Pitch
import time
from IPython.display import HTML
from queries import get_all_matches, load_basic_data, load_possession_data, load_possession_summary, load_spadl
from functions import interpolate_ball_data, prepare_player_data, get_interpolated_positions
from loadData import load_data_local, simulate_list_of_all_matches_query
from plots import create_voronoi_animation,create_voronoi_animation_with_labels,create_animation

st.set_page_config(layout="wide", page_title="Soccer Match Viewer")

# Set up matplotlib for animations
rc('animation', html='jshtml')

# Initialize session state variables if they don't exist
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'viewing_possession' not in st.session_state:
    st.session_state.viewing_possession = False
if 'possession_info' not in st.session_state:
    st.session_state.possession_info = ""
if 'plot_container' not in st.session_state:
    st.session_state.plot_container = st.empty()
if 'skip_frames' not in st.session_state:
    st.session_state.skip_frames = False
if 'playback_speed' not in st.session_state:
    st.session_state.playback_speed = 1



def main():
    with st.sidebar:
        st.header("Match Selection")
        # matches_df = get_all_matches()
        matches_df = simulate_list_of_all_matches_query()
        
        if matches_df.empty:
            st.error("No matches found in the database.")
            return
        
        match_options = matches_df['matchup'].tolist()
        selected_match = st.selectbox("Select a match to view", match_options)
        match_id = matches_df[matches_df['matchup'] == selected_match]['match_id'].values[0]
        
        # Load match data button
        if st.button("Load Match Data"):
            with st.spinner("Loading match data..."):
              
                df_ball,df_teams= load_basic_data(match_id)
                df_actions_label = load_spadl(match_id)
                df_possesion = load_possession_data(match_id)
                df_possesion_first_period = df_possesion[df_possesion['period_id'] == 1].copy()
                df_possesion_second_period = df_possesion[df_possesion['period_id'] == 2].copy()
                unique_team_ids = df_teams['team_id'].unique()
                if df_ball is None or df_ball.empty:
                    st.error(f"No data available for match {match_id}")
                    return
                
          
                df_home = df_teams[df_teams['team_id'] == unique_team_ids[0]].copy()
                df_away = df_teams[df_teams['team_id'] == unique_team_ids[1]].copy()
        
                st.session_state.df_teams = df_teams
                st.session_state.df_ball_original = df_ball
                st.session_state.df_home_original = df_home
                st.session_state.df_away_original = df_away
                st.session_state.df_actions_label = df_actions_label
                
                frames_between = 12  
                st.session_state.df_ball_interp = interpolate_ball_data(df_ball, frames_between)
                st.session_state.home_frames, st.session_state.home_positions = prepare_player_data(df_home, "home")
                st.session_state.away_frames, st.session_state.away_positions = prepare_player_data(df_away, "away")
                st.session_state.df_possesion_first_period = df_possesion_first_period
                st.session_state.df_possesion_second_period = df_possesion_second_period
                st.session_state.total_frames = len(st.session_state.df_ball_interp)
                st.session_state.current_frame = 0
                st.session_state.match_id = match_id
                st.session_state.data_loaded = True
                st.session_state.frames_between = frames_between
                st.session_state.viewing_possession = False
                
              
             
        
        # Possession Analysis section
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:

            st.header("Possession Analysis")
            period = st.radio("Select Period", ["1st Period", "2nd Period"])
            period_df = st.session_state.df_possesion_first_period if period == "1st Period" else st.session_state.df_possesion_second_period
            
            possession_options = []
            for idx, possession in period_df.iterrows():
                timestamp = possession['seconds']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                losing_team = possession.get('losing_team', 'Unknown')
                gaining_team = possession.get('gaining_team', 'Unknown')
                possession_options.append(f"{time_str} - {losing_team} → {gaining_team}")
               
            
            if possession_options:
                selected_possession = st.selectbox("Select a possession change", possession_options)
                possession_idx = possession_options.index(selected_possession)
                st.session_state.possession_info = selected_possession
                # Ensure data is available for animation generation
                possession = period_df.iloc[possession_idx]
                timestamp = possession['seconds']
                
                window_start = timestamp - 5  
                window_end = timestamp + 5    
                
                df_home_filtered = st.session_state.df_home_original[
                    (st.session_state.df_home_original['timestamp'] >= window_start) & 
                    (st.session_state.df_home_original['timestamp'] <= window_end)
                ]
                df_away_filtered = st.session_state.df_away_original[
                    (st.session_state.df_away_original['timestamp'] >= window_start) & 
                    (st.session_state.df_away_original['timestamp'] <= window_end)
                ]
                df_ball_filtered = st.session_state.df_ball_original[
                    (st.session_state.df_ball_original['timestamp'] >= window_start) & 
                    (st.session_state.df_ball_original['timestamp'] <= window_end)
                ]
                df_actions_label_filtered = st.session_state.df_actions_label[
                    (st.session_state.df_actions_label['timestamp'] >= window_start) &
                    (st.session_state.df_actions_label['timestamp'] <= window_end)
                ]
                


                if not df_home_filtered.empty and not df_away_filtered.empty and not df_ball_filtered.empty:
                    df_ball_interp = interpolate_ball_data(df_ball_filtered, st.session_state.frames_between)
                    home_frames, home_positions = prepare_player_data(df_home_filtered, "home")
                    away_frames, away_positions = prepare_player_data(df_away_filtered, "away")
                    df_teams = st.session_state.df_teams.copy()

                    st.session_state.possession_df_ball_interp = df_ball_interp
                    st.session_state.possession_home_frames = home_frames
                    st.session_state.possession_home_positions = home_positions
                    st.session_state.possession_away_frames = away_frames
                    st.session_state.possession_away_positions = away_positions

                    # Display buttons only when possession change is selected
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Generate Normal Animation"):
                            with st.spinner("Generating Normal animation..."):
                                anim = create_animation(
                                    df_ball_interp,
                                    home_frames,
                                    home_positions,
                                    away_frames,
                                    away_positions
                                )
                                st.session_state.possession_animation = anim.to_jshtml()
                                st.session_state.viewing_possession = True
                                st.success("Normal animation ready!")
                                st.rerun()

                    with col2:
                        if st.button("Generate Voronoi Animation"):
                            with st.spinner("Generating Voronoi animation..."):
                                voronoi_anim = create_voronoi_animation_with_labels(
                                    df_teams,
                                    df_ball_interp,
                                    home_frames,
                                    home_positions,
                                    away_frames,
                                    away_positions,
                                    actions_df=df_actions_label_filtered,
                                )
                                st.session_state.possession_voronoi_animation = voronoi_anim.to_jshtml()
                                st.session_state.viewing_possession = True
                                st.success("Voronoi animation ready!")
                                st.rerun()
                    
                else:
                    st.error("Insufficient data for the selected possession change.")

            
           
    # Main content area: display the animation
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    header_container = st.empty()
    animation_type = st.radio("Select Animation Type", ["Normal Animation", "Voronoi Animation"], horizontal=True)
    
    if st.session_state.viewing_possession:
        header_container.subheader(f"Possession Change: {st.session_state.possession_info}")
       
        # Use the selected animation type to determine which animation to show
        if animation_type == "Normal Animation":
            if 'possession_animation' in st.session_state:
                st.components.v1.html(st.session_state.possession_animation, height=1200,scrolling=True)
            else:
                st.warning("No animation available. Please load match data.")
        if animation_type == "Voronoi Animation":
            if 'possession_voronoi_animation' in st.session_state:
                st.components.v1.html(st.session_state.possession_voronoi_animation, height=1200, scrolling=True)
            else:
                st.warning("No Voronoi animation available. Please load match data.")
        
if __name__ == "__main__":
    main()
