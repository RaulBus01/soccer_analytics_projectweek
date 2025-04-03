from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.collections import PatchCollection
from matplotlib import animation, pyplot as plt, gridspec
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from shapely.geometry import Polygon as ShapelyPolygon, box
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from IPython.display import HTML
from functions import get_interpolated_positions

def create_animation(df_ball_interp, home_frames, home_positions, away_frames, away_positions, fps=24):
    """Create a complete animation of the soccer match"""
    pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
    fig, ax = pitch.draw(figsize=(12, 6))
    
    marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
 
    ball, = ax.plot([], [], ms=10, markerfacecolor='yellow', zorder=3, **marker_kwargs)
    away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
    home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)
    
    def animate_func(i):
        try:
            ball_x = df_ball_interp.iloc[i]['x'] / 100
            ball_y = df_ball_interp.iloc[i]['y'] / 100
            ball.set_data([ball_x], [ball_y])
            
            frame = df_ball_interp.iloc[i]['frame_id']
            home_pos = get_interpolated_positions(frame, home_frames, home_positions)
            away_pos = get_interpolated_positions(frame, away_frames, away_positions)
            
            home_x = [pos[0] / 100 for pos in home_pos.values()]
            home_y = [pos[1] / 100 for pos in home_pos.values()]
            away_x = [pos[0] / 100 for pos in away_pos.values()]
            away_y = [pos[1] / 100 for pos in away_pos.values()]
            
        
            home.set_data(home_x, home_y)
            away.set_data(away_x, away_y)
            return (ball, away, home)
        except Exception as e:
            print(f"Animation error: {e}")
            ball.set_data([], [])
            home.set_data([], [])
            away.set_data([], [])
            return (ball, away, home)
    
    frames = len(df_ball_interp)  
    anim = animation.FuncAnimation(fig, animate_func, frames=frames, interval=1000/fps, blit=False)
    plt.close(fig)
    return anim
def create_voronoi_animation(ball_df, home_frames, home_positions, away_frames, away_positions, fps=24):
    # Initialize patches list for Voronoi regions
    voronoi_patches = []
    
    # Create figure and axes
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax_pitch = fig.add_subplot(gs[0])
    ax_possession = fig.add_subplot(gs[1])
    ax_possession.axis('off')

    # Create pitch
    pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
    pitch.draw(ax=ax_pitch)

    # Create plot elements
    marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
    ball_dot, = ax_pitch.plot([], [], ms=6, markerfacecolor='white', zorder=3, **marker_kwargs)
    home_dot, = ax_pitch.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)
    away_dot, = ax_pitch.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
    ball_holder_marker, = ax_pitch.plot([], [], marker='o', color='red', markersize=10, zorder=4)

    # Text displays
    possession_display = ax_possession.text(0.1, 0.8, "", fontsize=18, va='top')
    home_half_count_display = ax_possession.text(0.1, 0.4, "", fontsize=16, va='top')
    away_half_count_display = ax_possession.text(0.1, 0.2, "", fontsize=16, va='top')
 
    def bounded_voronoi_regions(vor, pitch_width=68, pitch_length=105):
        bbox = box(0, 0, pitch_length, pitch_width)
        regions = []
        for region_idx in vor.point_region:
            vertices = vor.regions[region_idx]
            if -1 in vertices or len(vertices) == 0:
                regions.append(None)
                continue
            poly = ShapelyPolygon(vor.vertices[vertices])
            poly = poly.intersection(bbox)
            if not poly.is_empty and isinstance(poly, ShapelyPolygon):
                regions.append(poly)
            else:
                regions.append(None)
        return regions

    def animate(i):
        # Clear previous patches
        for patch in voronoi_patches:
            patch.remove()
        voronoi_patches.clear()
        
        try:
            # Get ball position
            ball_frame = ball_df.iloc[i]
            ball_x = ball_frame['x'] / 100
            ball_y = ball_frame['y'] / 100
            frame_id = ball_frame['frame_id']
            
            ball_dot.set_data([ball_x], [ball_y])
            
            # Get player positions for this frame using the get_interpolated_positions function
            home_pos = get_interpolated_positions(frame_id, home_frames, home_positions)
            away_pos = get_interpolated_positions(frame_id, away_frames, away_positions)
            
            # Extract coordinates
            home_x = [pos[0] / 100 for pos in home_pos.values()]
            home_y = [pos[1] / 100 for pos in home_pos.values()]
            away_x = [pos[0] / 100 for pos in away_pos.values()]
            away_y = [pos[1] / 100 for pos in away_pos.values()]
            
            # Update player dots
            home_dot.set_data(home_x, home_y)
            away_dot.set_data(away_x, away_y)
            
            # Combine coordinates and create team labels
            all_x = home_x + away_x
            all_y = home_y + away_y
            team = ['Home'] * len(home_x) + ['Away'] * len(away_x)
            
            # Create points array for Voronoi
            points = np.vstack((all_x, all_y)).T
            
            # Calculate Voronoi regions
            home_area = 0
            away_area = 0
            
            if len(points) >= 4:  # Need at least 4 points for a meaningful Voronoi diagram
                ghost_margin = 30 
                ghost_spacing = 20  
                
                # Add ghost points around the pitch to bound the Voronoi regions
                x_ghosts = np.arange(-ghost_margin, 105 + ghost_margin + 1, ghost_spacing)
                y_ghosts = np.arange(-ghost_margin, 68 + ghost_margin + 1, ghost_spacing)
                
                ghost_points = []
                for gx in x_ghosts:
                    ghost_points.append([gx, -ghost_margin])       
                    ghost_points.append([gx, 68 + ghost_margin])   
                for gy in y_ghosts:
                    ghost_points.append([-ghost_margin, gy])       
                    ghost_points.append([105 + ghost_margin, gy]) 
                
                ghost_points = np.array(ghost_points)
                
                all_points = np.vstack((points, ghost_points))
                vor = Voronoi(all_points)
                
                regions = bounded_voronoi_regions(vor)
                
                # Only use regions corresponding to the players
                num_players = len(points)
                player_regions = regions[:num_players]
                
                # Create team colors for players
                team_colors = ['#7f63b8' if t == 'Home' else '#b94b75' for t in team]
                
                for region, color, tlabel in zip(player_regions, team_colors, team):
                    if region:
                        area = region.area
                        patch = Polygon(list(region.exterior.coords), alpha=0.2, edgecolor='black', facecolor=color, zorder=0)
                        ax_pitch.add_patch(patch)
                        voronoi_patches.append(patch)
                        if tlabel == 'Home':
                            home_area += area
                        else:
                            away_area += area
                
                # Calculate space control percentages
                total_area = home_area + away_area
                if total_area > 0:
                    home_pct = 100 * home_area / total_area
                    away_pct = 100 * away_area / total_area
                    possession_display.set_text(f"Home control:\n{home_pct:.1f}%\n\nAway control:\n{away_pct:.1f}%")
                else:
                    possession_display.set_text("Pitch Control\n-")
                    
                # Calculate player positions by half
                home_left_half = sum(1 for x in home_x if x * 100 < 52.5)
                home_right_half = sum(1 for x in home_x if x * 100 >= 52.5)
                
                away_left_half = sum(1 for x in away_x if x * 100 < 52.5)
                away_right_half = sum(1 for x in away_x if x * 100 >= 52.5)
                
                home_half_count_display.set_text(
                    f"Home players:\nLeft: {home_left_half}\nRight: {home_right_half}"
                )
                away_half_count_display.set_text(
                    f"Away players:\nLeft: {away_left_half}\nRight: {away_right_half}"
                )
                
                # Find the nearest player to the ball
                if len(points) > 0:
                    ball_pos = np.array([ball_x, ball_y])
                    dists = np.linalg.norm(points - ball_pos, axis=1)
                    closest = np.argmin(dists)
                    closest_x, closest_y = points[closest]
                    ball_holder_marker.set_data([closest_x], [closest_y])
                else:
                    ball_holder_marker.set_data([], [])
                
        except Exception as e:
            print(f"Animation error at frame {i}: {e}")
            ball_dot.set_data([], [])
            home_dot.set_data([], [])
            away_dot.set_data([], [])
        
        return [ball_dot, home_dot, away_dot, ball_holder_marker, possession_display, home_half_count_display, away_half_count_display] + voronoi_patches

    # Create animation with fewer frames for performance
    frames = min(150, len(ball_df))  # Limit to 150 frames for better performance
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps, blit=False)
    
    plt.close(fig)  # Close the figure to prevent displaying it twice
    return anim

def create_voronoi_animation_with_labels(df_teams,ball_df, home_frames, home_positions, away_frames, away_positions, actions_df=None, fps=24, ):
   

    # Initialize collections
    voronoi_patches = []
    player_labels = []
    last_actions = {}  # To persist actions for a few frames
    ACTION_PERSIST_FRAMES = 10
    
    # Create figure and axes
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax_pitch = fig.add_subplot(gs[0])
    ax_possession = fig.add_subplot(gs[1])
    ax_possession.axis('off')

    # Create pitch
    pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
    pitch.draw(ax=ax_pitch)

    # Create plot elements
    marker_kwargs = {'marker': 'o', 'markeredgecolor': 'black', 'linestyle': 'None'}
    ball_dot, = ax_pitch.plot([], [], ms=6, markerfacecolor='white', zorder=3, **marker_kwargs)
    home_dot, = ax_pitch.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)
    away_dot, = ax_pitch.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)
    ball_holder_marker, = ax_pitch.plot([], [], marker='*', color='gold', markersize=20, zorder=4)

    # Text displays
    possession_display = ax_possession.text(0.1, 0.8, "", fontsize=18, va='top')
    home_half_count_display = ax_possession.text(0.1, 0.4, "", fontsize=16, va='top')
    away_half_count_display = ax_possession.text(0.1, 0.2, "", fontsize=16, va='top')
 
    def bounded_voronoi_regions(vor, pitch_width=68, pitch_length=105):
        bbox = box(0, 0, pitch_length, pitch_width)
        regions = []
        for region_idx in vor.point_region:
            vertices = vor.regions[region_idx]
            if -1 in vertices or len(vertices) == 0:
                regions.append(None)
                continue
            poly = ShapelyPolygon(vor.vertices[vertices])
            poly = poly.intersection(bbox)
            if not poly.is_empty and isinstance(poly, ShapelyPolygon):
                regions.append(poly)
            else:
                regions.append(None)
        return regions

    # Mapping of action codes to readable names
    spadl_actions_mapping = {
        "0": "pass", "1": "cross", "2": "throw-in", "3": "crossed FK",
        "4": "short FK", "5": "crossed corner", "6": "short corner",
        "7": "take-on", "8": "foul", "9": "tackle", "10": "interception",
        "11": "shot", "12": "penalty", "13": "FK shot", "14": "save",
        "15": "claim", "16": "punch", "17": "pick-up", "18": "clearance",
        "19": "bad touch", "21": "dribble", "22": "goal kick"
    }
    
    # Process actions data
    if actions_df is not None:
        if 'action_type' in actions_df.columns:
            actions_df = actions_df.copy()
            actions_df['action_type'] = actions_df['action_type'].astype(str).map(
                lambda x: spadl_actions_mapping.get(x, x)
            )
    
    # Create a player name lookup dictionary
    player_name_map = {}
    
    # Extract names from actions data if available
    if df_teams is not None:
        for _, row in df_teams.iterrows():
            player_id = row['player_id']
            player_name = row['player_name']
            player_name_map[player_id] = player_name
    
    
    def animate(i):
        nonlocal voronoi_patches, player_labels
        
        # Clear previous elements
        for patch in voronoi_patches:
            patch.remove()
        voronoi_patches.clear()
        
        for label in player_labels:
            label.remove()
        player_labels.clear()
        
        try:
            # Get ball position
            ball_frame = ball_df.iloc[i]
            ball_x = ball_frame['x'] / 100
            ball_y = ball_frame['y'] / 100
            frame_id = ball_frame['frame_id']
            
            ball_dot.set_data([ball_x], [ball_y])
            
            # Get player positions
            home_pos = get_interpolated_positions(frame_id, home_frames, home_positions)
            away_pos = get_interpolated_positions(frame_id, away_frames, away_positions)
            
            # Extract coordinates and player IDs
            home_ids = list(home_pos.keys())
            away_ids = list(away_pos.keys())
            
            home_x = [home_pos[pid][0] / 100 for pid in home_ids]
            home_y = [home_pos[pid][1] / 100 for pid in home_ids]
            away_x = [away_pos[pid][0] / 100 for pid in away_ids]
            away_y = [away_pos[pid][1] / 100 for pid in away_ids]
            
            # Update player dots
            home_dot.set_data(home_x, home_y)
            away_dot.set_data(away_x, away_y)
            
            # Combine coordinates and team info
            all_x = home_x + away_x
            all_y = home_y + away_y
            all_ids = home_ids + away_ids
            all_teams = ['Home'] * len(home_x) + ['Away'] * len(away_x)
            
            # Create points array for Voronoi
            points = np.vstack((all_x, all_y)).T
            
            # Calculate Voronoi regions
            home_area = 0
            away_area = 0
            
            if len(points) >= 4:  # Need at least 4 points for a meaningful Voronoi diagram
                # Create ghost points to bound the Voronoi regions
                ghost_margin = 30 
                ghost_spacing = 20  
                
                x_ghosts = np.arange(-ghost_margin, 105 + ghost_margin + 1, ghost_spacing)
                y_ghosts = np.arange(-ghost_margin, 68 + ghost_margin + 1, ghost_spacing)
                
                ghost_points = []
                for gx in x_ghosts:
                    ghost_points.append([gx, -ghost_margin])
                    ghost_points.append([gx, 68 + ghost_margin])
                for gy in y_ghosts:
                    ghost_points.append([-ghost_margin, gy])
                    ghost_points.append([105 + ghost_margin, gy])
                
                ghost_points = np.array(ghost_points)
                all_points = np.vstack((points, ghost_points))
                
                vor = Voronoi(all_points)
                regions = bounded_voronoi_regions(vor)
                
                # Only use regions corresponding to players
                num_players = len(points)
                player_regions = regions[:num_players]
                
                # Set team colors for visualization
                team_colors = ['#111f34' if t == 'Home' else '#b94523' for t in all_teams]
                
                # Create Voronoi patches
                for region, color, team_name in zip(player_regions, team_colors, all_teams):
                    if region:
                        area = region.area
                        patch = Polygon(list(region.exterior.coords), 
                                       alpha=0.2, edgecolor='black', facecolor=color, zorder=0)
                        ax_pitch.add_patch(patch)
                        voronoi_patches.append(patch)
                        
                        if team_name == 'Home':
                            home_area += area
                        else:
                            away_area += area
            
            # Calculate space control percentages
            total_area = home_area + away_area
            if total_area > 0:
                home_pct = 100 * home_area / total_area
                away_pct = 100 * away_area / total_area
                possession_display.set_text(f"Home control:\n{home_pct:.1f}%\n\nAway control:\n{away_pct:.1f}%")
            else:
                possession_display.set_text("Pitch Control\n-")
            
            # Calculate player positions by half
            home_left_half = sum(1 for x in home_x if x * 100 < 52.5)
            home_right_half = sum(1 for x in home_x if x * 100 >= 52.5)
            away_left_half = sum(1 for x in away_x if x * 100 < 52.5)
            away_right_half = sum(1 for x in away_x if x * 100 >= 52.5)
            
            home_half_count_display.set_text(
                f"Home players:\nLeft: {home_left_half}\nRight: {home_right_half}"
            )
            away_half_count_display.set_text(
                f"Away players:\nLeft: {away_left_half}\nRight: {away_right_half}"
            )
            ball_label = ax_pitch.text(ball_x, ball_y, "Ball", fontsize=12, ha='center', va='center', 
                                   color='cyan', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle="round,pad=0.3"))

            # Append to player labels to keep track
            player_labels.append(ball_label)
            # Add player labels with actions
            for idx, (px, py, player_id, team_name) in enumerate(zip(all_x, all_y, all_ids, all_teams)):
                # Get player name using the mapping
                player_name = player_name_map.get(player_id, f"#{player_id}")
                
                # Default label is just player name (or short version)
                short_name = player_name.split()[-1] if ' ' in player_name else player_name
                label_text = f"{short_name}"
                
                # Look for actions by this player at current frame
                if actions_df is not None:
                    player_actions = actions_df[(actions_df['frame_id'] == frame_id) & 
                                               (actions_df['player_id'] == player_id)]
                    
                    if not player_actions.empty:
                        action = player_actions.iloc[0]['action_type']
                        last_actions[player_id] = {"action": action, "frames_left": ACTION_PERSIST_FRAMES}
                    elif player_id in last_actions:
                        last_actions[player_id]["frames_left"] -= 1
                        if last_actions[player_id]["frames_left"] <= 0:
                            del last_actions[player_id]
                    
                   
                    if player_id in last_actions:
                        label_text = f"{short_name}\n{last_actions.get(player_id, {}).get('action', 'None').upper()}"
                     
                if team_name == 'Home':
                    if player_name == 'Ball':
                        continue
                    
                    label = ax_pitch.text(px, py, label_text, 
                                        fontsize=10, color='white', ha='center', 
                                        bbox=dict(facecolor='#7f63b8', alpha=0.7, edgecolor='none'),
                                        zorder=5)
                else:
                    
                    label = ax_pitch.text(px, py, label_text, 
                                        fontsize=10, color='white', ha='center', 
                                        bbox=dict(facecolor='#b94b75', alpha=0.7, edgecolor='none'),
                                        zorder=5)
                
                player_labels.append(label)
            
            # Mark player closest to the ball
            if len(points) > 0:
                ball_pos = np.array([ball_x, ball_y])
                dists = np.linalg.norm(points - ball_pos, axis=1)
                closest = np.argmin(dists)
                closest_x, closest_y = points[closest]
                ball_holder_marker.set_data([closest_x], [closest_y])
            else:
                ball_holder_marker.set_data([], [])
            
        except Exception as e:
            print(f"Animation error at frame {i}: {e}")
            ball_dot.set_data([], [])
            home_dot.set_data([], [])
            away_dot.set_data([], [])
        
        return [ball_dot, home_dot, away_dot, ball_holder_marker, 
               possession_display, home_half_count_display, 
               away_half_count_display] + voronoi_patches + player_labels
    
    # Create animation with fewer frames for performance
    frames = min(150, len(ball_df))
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps, blit=False)
    
    plt.close(fig)  # Close the figure to prevent displaying it twice
    return anim