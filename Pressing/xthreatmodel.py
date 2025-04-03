from socceraction.xthreat import ExpectedThreat
import pandas as pd
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from mplsoccer import Pitch

# Step 1: Database Connection Function
def connect_to_database(dbname, user, password, host, port, sslmode):
    """Connect to PostgreSQL database and return connection"""
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            sslmode=sslmode
        )
        print("Database connection established successfully!")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Step 2: Load Data Function with EventType Join
def load_spadl_actions(conn):
    """Load pass and carry actions from SpadlAction table with proper EventType mapping"""
    query = """
    SELECT 
        s.id, s.game_id, s.period_id, s.seconds, 
        s.player_id, s.team_id, 
        s.start_x, s.start_y, s.end_x, s.end_y, 
        e.name as action_type_name, s.action_type as action_type_id,
        s.result, s.bodypart
    FROM 
        SpadlAction s
    JOIN 
        EventType e ON s.action_type = e.eventtype_id
    WHERE 
        e.name IN ('pass', 'dribble', 'cross', 'carry')
        AND s.result != 'offside'  -- Exclude offside passes
        AND s.start_x IS NOT NULL  -- Ensure coordinates are present
        AND s.end_x IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, conn)
        print(f"Successfully loaded {len(df)} actions from database")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative query without EventType join...")
        
        # Fallback query if EventType join doesn't work
        fallback_query = """
        SELECT 
            id, game_id, period_id, seconds, 
            player_id, team_id, 
            start_x, start_y, end_x, end_y, 
            action_type as action_type_id, result, bodypart
        FROM 
            SpadlAction
        WHERE 
            start_x IS NOT NULL  -- Ensure coordinates are present
            AND end_x IS NOT NULL
        """
        
        try:
            df = pd.read_sql(fallback_query, conn)
            print(f"Successfully loaded {len(df)} actions with fallback query")
            
            # Get list of unique action type IDs from the data
            action_type_ids = df['action_type_id'].unique()
            print(f"Found action type IDs: {action_type_ids}")
            
            # Try to map action type IDs to names if possible
            try:
                type_query = "SELECT eventtype_id, name FROM EventType"
                type_mapping = pd.read_sql(type_query, conn)
                print("Available event types:")
                print(type_mapping)
                
                # Create mapping dictionary
                type_dict = dict(zip(type_mapping['eventtype_id'], type_mapping['name']))
                
                # Apply mapping
                df['action_type_name'] = df['action_type_id'].map(type_dict)
                
                # Filter for relevant actions if mapping successful
                relevant_actions = ['pass', 'dribble', 'cross', 'carry']
                df = df[df['action_type_name'].isin(relevant_actions)]
                
            except Exception as type_err:
                print(f"Could not map action types: {type_err}")
            
            return df
        except Exception as fb_err:
            print(f"Error with fallback query: {fb_err}")
            return pd.DataFrame()

# Step 3: Create Expected Threat Grid
def create_xt_grid(n_cells_x=12, n_cells_y=8):
    """
    Create a simplified xT grid based on pitch location
    Higher values in attacking third, particularly central areas
    """
    # Create empty grid
    grid = np.zeros((n_cells_y, n_cells_x))
    
    # Populate with increasing threat values toward the goal
    # This is a simplified version - real models train on large datasets
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            # Base value increases as we get closer to goal (x increases)
            base_value = (x / n_cells_x) ** 2  # Quadratic increase
            
            # Central positions are more valuable
            centrality = 1 - abs((y - (n_cells_y-1)/2) / (n_cells_y/2)) ** 2
            
            # Combine factors
            grid[y, x] = base_value * (0.5 + 0.5 * centrality)
    
    return grid

# Step 4: Calculate xT values for actions
def calculate_xt_values(df, grid, n_cells_x=12, n_cells_y=8, pitch_length=105, pitch_width=68):
    """Calculate xT values for each action based on start and end positions"""
    # Check for coordinate ranges to determine scaling
    x_max = max(df['start_x'].max(), df['end_x'].max())
    y_max = max(df['start_y'].max(), df['end_y'].max())
    
    print(f"Maximum x coordinate: {x_max}, Maximum y coordinate: {y_max}")
    
    # Adjust pitch dimensions if necessary
    if x_max > 1 and x_max <= 100:
        # Assuming coordinates are in percentages or 0-100 scale
        pitch_length = 100
        pitch_width = 100
        print("Assuming coordinates are in percentages (0-100)")
    elif x_max <= 1:
        # Assuming coordinates are normalized (0-1)
        pitch_length = 1
        pitch_width = 1
        print("Assuming coordinates are normalized (0-1)")
    else:
        # Assuming coordinates are in meters
        print(f"Assuming coordinates are in meters with pitch size {pitch_length}m x {pitch_width}m")
    
    # Normalize coordinates to grid cells
    df['start_cell_x'] = (df['start_x'] / pitch_length * n_cells_x).astype(int).clip(0, n_cells_x-1)
    df['start_cell_y'] = (df['start_y'] / pitch_width * n_cells_y).astype(int).clip(0, n_cells_y-1)
    df['end_cell_x'] = (df['end_x'] / pitch_length * n_cells_x).astype(int).clip(0, n_cells_x-1)
    df['end_cell_y'] = (df['end_y'] / pitch_width * n_cells_y).astype(int).clip(0, n_cells_y-1)
    
    # Lookup xT values in the grid
    df['start_xt'] = df.apply(lambda row: grid[row['start_cell_y'], row['start_cell_x']], axis=1)
    df['end_xt'] = df.apply(lambda row: grid[row['end_cell_y'], row['end_cell_x']], axis=1)
    
    # Calculate xT contribution
    df['xt_contribution'] = df['end_xt'] - df['start_xt']
    
    return df

# Step 5: Player and Team Analysis
def analyze_xt_by_player_and_team(df):
    """Group and analyze xT contributions by player and team"""
    # Player analysis
    player_xt = df.groupby('player_id')['xt_contribution'].agg(['sum', 'mean', 'count']).reset_index()
    player_xt = player_xt.sort_values(by='sum', ascending=False)
    player_xt.columns = ['player_id', 'total_xt', 'avg_xt', 'num_actions']
    
    # Team analysis
    team_xt = df.groupby('team_id')['xt_contribution'].sum().reset_index()
    team_xt = team_xt.sort_values(by='xt_contribution', ascending=False)
    
    # Action type analysis
    if 'action_type_name' in df.columns:
        action_type_field = 'action_type_name'
    else:
        action_type_field = 'action_type_id'
        
    action_xt = df.groupby(action_type_field)['xt_contribution'].agg(['sum', 'mean', 'count']).reset_index()
    action_xt = action_xt.sort_values(by='sum', ascending=False)
    
    return player_xt, team_xt, action_xt

# Step 6: Visualization Functions
def plot_xt_grid(grid, n_cells_x=12, n_cells_y=8):
    """Visualize the xT grid as a heatmap on a football pitch"""
    pitch = Pitch(pitch_type='custom', pitch_length=n_cells_x, pitch_width=n_cells_y, 
                  line_color='black', pitch_color='grass')
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Plot heatmap
    cmap = plt.cm.viridis
    hm = ax.imshow(grid, cmap=cmap, extent=[0, n_cells_x, 0, n_cells_y], origin='lower')
    
    # Add colorbar
    plt.colorbar(hm, ax=ax, label='Expected Threat (xT) Value')
    
    plt.title('Expected Threat (xT) Grid', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_xt_contributions(df, pitch_length=105, pitch_width=68):
    """Visualize xT contributions as a scatter plot on the football pitch"""
    # Check for coordinate ranges to adjust pitch dimensions
    x_max = max(df['start_x'].max(), df['end_x'].max())
    y_max = max(df['start_y'].max(), df['end_y'].max())
    
    if x_max > 1 and x_max <= 100:
        # Assuming coordinates are in percentages
        pitch_length = 100
        pitch_width = 100
    elif x_max <= 1:
        # Assuming coordinates are normalized
        pitch_length = 1
        pitch_width = 1
    
    pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width, 
                  line_color='black', pitch_color='grass')
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Filter for positive xT contributions for better visualization
    df_positive = df[df['xt_contribution'] > 0]
    
    # Plot arrows for actions (sample if too many)
    if len(df_positive) > 0:
        sample_size = min(1000, len(df_positive))
        for _, row in df_positive.sample(sample_size).iterrows():
            # Scale alpha by xT contribution (with a minimum to ensure visibility)
            alpha = min(1, max(0.1, row['xt_contribution']*5))
            ax.arrow(row['start_x'], row['start_y'], 
                    row['end_x'] - row['start_x'], row['end_y'] - row['start_y'],
                    head_width=pitch_width/50, head_length=pitch_length/50, 
                    fc='blue', ec='blue', alpha=alpha)
    
        # Plot scatter for end points, sized by xT contribution
        sc = ax.scatter(df_positive['end_x'], df_positive['end_y'], 
                        s=df_positive['xt_contribution']*100, c=df_positive['xt_contribution'], 
                        cmap='coolwarm', alpha=0.6, edgecolors='black')
        
        plt.colorbar(sc, ax=ax, label='xT Contribution')
    else:
        print("No positive xT contributions found for visualization")
        
    plt.title('Expected Threat (xT) Contributions', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_player_xt_contributions(player_xt, top_n=20):
    """Plot top players by xT contribution"""
    plt.figure(figsize=(12, 8))
    
    # Get top N players
    top_players = player_xt.head(top_n)
    
    # Create horizontal bar chart
    plt.barh(top_players['player_id'], top_players['total_xt'], color='darkblue')
    
    plt.xlabel('Total xT Contribution', fontsize=12)
    plt.ylabel('Player ID', fontsize=12)
    plt.title(f'Top {top_n} Players by xT Contribution', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_team_heatmap(df, pitch_length=105, pitch_width=68):
    """Create heatmap of xT contributions by field position"""
    # Adjust pitch dimensions based on coordinate ranges
    x_max = max(df['start_x'].max(), df['end_x'].max())
    if x_max > 1 and x_max <= 100:
        pitch_length = 100
        pitch_width = 100
    elif x_max <= 1:
        pitch_length = 1
        pitch_width = 1
    
    # Create pitch
    pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width, 
                  line_color='black', pitch_color='grass')
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Create bins for heatmap
    n_bins_x = 20
    n_bins_y = 14
    
    # Create 2D histogram of xT contributions
    bin_statistic = pitch.bin_statistic(
        df['end_x'], df['end_y'],
        values=df['xt_contribution'],
        statistic='sum',
        bins=(n_bins_x, n_bins_y)
    )
    
    # Plot heatmap
    hm = pitch.heatmap(bin_statistic, ax=ax, cmap='viridis', edgecolors='gray')
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label('Total xT Contribution')
    
    plt.title('Expected Threat (xT) Contribution Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Step 7: Main Function to Run Analysis
def run_xt_analysis(dbname, user, password, host, port, sslmode):
    """Complete Expected Threat analysis workflow"""
    # Connect to database
    conn = connect_to_database(dbname, user, password, host, port,sslmode)
    if conn is None:
        return
    
    try:
        # Check available tables in the database
        check_schema_query = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        schema_df = pd.read_sql(check_schema_query, conn)
        print("Available tables in the database:")
        print(schema_df)
        
        # Load data with event type mapping
        actions_df = load_spadl_actions(conn)
        if actions_df.empty:
            print("No data loaded. Check your database connection and query.")
            conn.close()
            return
        
        # Print data sample and column info for debugging
        print("\nLoaded data sample:")
        print(actions_df.head())
        print("\nData columns:")
        print(actions_df.columns.tolist())
        print("\nData types:")
        print(actions_df.dtypes)
        
        # Create xT grid
        xt_grid = create_xt_grid()
        
        # Calculate xT values
        actions_with_xt = calculate_xt_values(actions_df, xt_grid)
        
        # Player and team analysis
        player_xt, team_xt, action_xt = analyze_xt_by_player_and_team(actions_with_xt)
        
        # Print top players and teams
        print("\nTop 10 Players by xT Contribution:")
        print(player_xt.head(10))
        
        print("\nTeam xT Contributions:")
        print(team_xt)
        
        print("\nAction Type xT Contributions:")
        print(action_xt)
        
        # Visualizations
        plot_xt_grid(xt_grid)
        plot_xt_contributions(actions_with_xt)
        plot_player_xt_contributions(player_xt)
        plot_team_heatmap(actions_with_xt)
        
        # Save results if needed
        player_xt.to_csv('player_xt_contributions.csv', index=False)
        team_xt.to_csv('team_xt_contributions.csv', index=False)
        action_xt.to_csv('action_type_xt_contributions.csv', index=False)
        
        print("\nAnalysis complete! Results saved to CSV files.")
        
    finally:
        # Close database connection
        conn.close()

# Example usage
if __name__ == "__main__":
    # Replace with your actual database credentials
    run_xt_analysis(
        dbname="internation_week",
        user="busit_79",
        password="[KSiO^`9??V\K=?W2>Z:`",
        host="fuji.ucll.be",  # or your server address
        port="52425",       # default PostgreSQL port
        sslmode="require",
    

    )