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
    ball_holder_marker, = ax_pitch.plot([], [], marker='*', color='red', markersize=20, zorder=4)

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

def create_voronoi_animation(ball_df, home_frames, home_positions, away_frames, away_positions, fps=24):
    