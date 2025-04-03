import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Interpolate ball data
def interpolate_ball_data(ball_df, frames_between=3):
    x_values = ball_df['x'].values
    y_values = ball_df['y'].values
    frame_ids = ball_df['frame_id'].values

    original_len = len(ball_df)
    new_points = original_len * frames_between
    new_indices = np.linspace(0, original_len - 1, new_points)

    x_interp = interp1d(np.arange(original_len), x_values, kind='cubic')
    y_interp = interp1d(np.arange(original_len), y_values, kind='cubic')
    frame_interp = interp1d(np.arange(original_len), frame_ids, kind='linear')

    new_x = x_interp(new_indices)
    new_y = y_interp(new_indices)
    new_frames = frame_interp(new_indices)

    result_df = pd.DataFrame({
        'x': new_x,
        'y': new_y,
        'frame_id': new_frames
    })

    nearest_indices = np.round(new_indices).astype(int).clip(0, original_len - 1)
    for col in ball_df.columns:
        if col not in ['x', 'y', 'frame_id']:
            if ball_df[col].dtype == np.dtype('O'):
                values = [ball_df[col].iloc[i] for i in nearest_indices]
                result_df[col] = values
            else:
                try:
                    num_interp = interp1d(np.arange(original_len), ball_df[col].values, kind='linear')
                    result_df[col] = num_interp(new_indices)
                except:
                    values = [ball_df[col].iloc[i] for i in nearest_indices]
                    result_df[col] = values

    return result_df

# Prepare player data
def prepare_player_data(df, team):
    frames = sorted(df['frame_id'].unique())
    player_positions = {}

    for frame in frames:
        frame_data = df[df['frame_id'] == frame]
        positions = {}
        for _, player in frame_data.iterrows():
            player_id = player['player_id']
            positions[player_id] = [player['x'], player['y']]
        player_positions[frame] = positions

    return frames, player_positions

# Get interpolated player positions
def get_interpolated_positions(frame_id, frames, positions):
    if frame_id <= frames[0]:
        return positions[frames[0]]
    elif frame_id >= frames[-1]:
        return positions[frames[-1]]

    frame_before = frames[0]
    frame_after = frames[-1]

    for i in range(len(frames) - 1):
        if frames[i] <= frame_id <= frames[i + 1]:
            frame_before = frames[i]
            frame_after = frames[i + 1]
            break

    factor = (frame_id - frame_before) / (frame_after - frame_before) if frame_after != frame_before else 0

    result = {}
    for player_id in set(positions[frame_before].keys()) & set(positions[frame_after].keys()):
        x1, y1 = positions[frame_before][player_id]
        x2, y2 = positions[frame_after][player_id]
        x = x1 + factor * (x2 - x1)
        y = y1 + factor * (y2 - y1)
        result[player_id] = [x, y]

    return result