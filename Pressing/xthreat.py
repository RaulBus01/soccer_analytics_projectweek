import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from scipy.ndimage import gaussian_filter
# from expected_threat import ExpectedThreat, connect_to_db

def connect_to_db():
    """
    Connect to the database using psycopg2
    
    Returns:
    --------
    conn : psycopg2.extensions.connection
        Database connection
    """
    conn = psycopg2.connect(
        host="fuji.ucll.be",
        database="international_week",
        user="busit_79",
        password="[KSiO^`9??V\K=?W2>Z:`",
        port=52425,
        sslmode="require",
    )
    return conn

class ExpectedThreat:
    def __init__(self, n_rows=12, n_cols=16, smoothing=True, sigma=1.0):
        """
        Initialize the Expected Threat model
        
        Parameters:
        -----------
        n_rows : int
            Number of grid rows for pitch division
        n_cols : int
            Number of grid columns for pitch division
        smoothing : bool
            Whether to apply Gaussian smoothing to the grid
        sigma : float
            Sigma parameter for Gaussian smoothing
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.smoothing = smoothing
        self.sigma = sigma
        self.grid = None
        self.transition_matrix = None
        self.shot_probability = None
        
    def _preprocess_data(self, actions_df):
        """
        Preprocess SPADL actions data for xT calculation - now without action type filtering
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            DataFrame containing SPADL actions
            
        Returns:
        --------
        preprocessed_df : pandas.DataFrame
            Preprocessed actions for xT calculation
        """
        # Create a copy to work with
        df = actions_df.copy()
        
        # Filter actions based on having valid start and end coordinates
        # This approach doesn't rely on action types at all
        df = df.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
        
        # Convert coordinates to grid cells
        df['start_cell_x'] = pd.cut(df['start_x'], self.n_cols, labels=False)
        df['start_cell_y'] = pd.cut(df['start_y'], self.n_rows, labels=False)
        df['end_cell_x'] = pd.cut(df['end_x'], self.n_cols, labels=False)
        df['end_cell_y'] = pd.cut(df['end_y'], self.n_rows, labels=False)
        
        # Calculate start and end cell indices
        df['start_cell'] = df['start_cell_y'] * self.n_cols + df['start_cell_x']
        df['end_cell'] = df['end_cell_y'] * self.n_cols + df['end_cell_x']
        
        # Filter actions where the ball moves to a different cell
        df = df[df['start_cell'] != df['end_cell']]
        
        # Add success flag - assuming 'success' is a standard value in the result column
        # If not, we can modify this logic
        df['success'] = df['result'] == 'success'
        
        return df
    
    def _preprocess_shots(self, actions_df):
        """
        Preprocess shots data for xT calculation, now identifying shots based on position
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            DataFrame containing SPADL actions
            
        Returns:
        --------
        shots_df : pandas.DataFrame
            Preprocessed shots for xT calculation
        """
        # Create a copy
        df = actions_df.copy()
        
        # Instead of filtering by action_type, we'll identify potential shots 
        # based on their end location near the goal
        # Assuming pitch coordinates where the goal is at x=100
        goal_x_threshold = 90  # Actions ending very close to goal line
        penalty_area_width = 44  # Standard width of penalty area
        
        # Identify potential shots by position (close to goal, within width of penalty area)
        potential_shots = df[
            (df['end_x'] >= goal_x_threshold) & 
            (df['end_y'] >= (100 - penalty_area_width)/2) & 
            (df['end_y'] <= (100 + penalty_area_width)/2)
        ].copy()
        
        # Convert coordinates to grid cells
        potential_shots['cell_x'] = pd.cut(potential_shots['start_x'], self.n_cols, labels=False)
        potential_shots['cell_y'] = pd.cut(potential_shots['start_y'], self.n_rows, labels=False)
        potential_shots['cell'] = potential_shots['cell_y'] * self.n_cols + potential_shots['cell_x']
        
        # Add goal flag - assuming 'success' means a goal for these actions
        potential_shots['goal'] = potential_shots['result'] == 'success'
        
        return potential_shots
    
    def _calculate_transition_matrix(self, actions_df):
        """
        Calculate the transition matrix between grid cells
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            Preprocessed actions DataFrame
            
        Returns:
        --------
        transition_matrix : numpy.ndarray
            Matrix of transition probabilities between grid cells
        """
        n_cells = self.n_rows * self.n_cols
        transition_matrix = np.zeros((n_cells, n_cells))
        
        # Group by start cell and calculate transition probabilities
        for start_cell in range(n_cells):
            # Filter actions starting from this cell
            cell_actions = actions_df[actions_df['start_cell'] == start_cell]
            
            if len(cell_actions) > 0:
                # Count successful actions to each end cell
                success_counts = cell_actions[cell_actions['success']].groupby('end_cell').size()
                
                # Total actions from this cell
                total_actions = len(cell_actions)
                
                # Calculate transition probabilities
                for end_cell, count in success_counts.items():
                    transition_matrix[start_cell, end_cell] = count / total_actions
        
        return transition_matrix
    
    def _calculate_shot_probability(self, shots_df):
        """
        Calculate the probability of scoring from each grid cell
        
        Parameters:
        -----------
        shots_df : pandas.DataFrame
            Preprocessed shots DataFrame
            
        Returns:
        --------
        shot_probability : numpy.ndarray
            Array of shot success probabilities for each grid cell
        """
        n_cells = self.n_rows * self.n_cols
        shot_probability = np.zeros(n_cells)
        
        # Group by cell and calculate goal probability
        cell_shots = shots_df.groupby('cell')
        
        for cell, group in cell_shots:
            # Count goals and total shots
            goals = group['goal'].sum()
            total_shots = len(group)
            
            # Calculate probability
            shot_probability[cell] = goals / total_shots if total_shots > 0 else 0
            
        # Apply smoothing if enabled
        if self.smoothing:
            # Reshape to 2D grid
            grid_2d = shot_probability.reshape(self.n_rows, self.n_cols)
            # Apply Gaussian filter
            smoothed_grid = gaussian_filter(grid_2d, sigma=self.sigma)
            # Reshape back to 1D
            shot_probability = smoothed_grid.flatten()
            
        return shot_probability
    
    def _calculate_xt_values(self):
        """
        Calculate xT values for each grid cell using the Markov model
        
        Returns:
        --------
        xt_values : numpy.ndarray
            Array of xT values for each grid cell
        """
        n_cells = self.n_rows * self.n_cols
        
        # Initialize xT values with shot probabilities
        xt_values = self.shot_probability.copy()
        
        # Iterative value calculation (value iteration)
        max_iterations = 100
        epsilon = 1e-6
        
        for _ in range(max_iterations):
            old_values = xt_values.copy()
            
            # Update each cell's value
            for cell in range(n_cells):
                # Expected value from transitions
                expected_value = np.sum(self.transition_matrix[cell] * xt_values)
                
                # Choose maximum of shooting or moving
                xt_values[cell] = max(self.shot_probability[cell], expected_value)
            
            # Check convergence
            if np.max(np.abs(xt_values - old_values)) < epsilon:
                break
                
        return xt_values
    
    def fit(self, actions_df):
        """
        Fit the xT model using the provided actions data
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            DataFrame containing SPADL actions
            
        Returns:
        --------
        self : ExpectedThreat
            The fitted xT model
        """
        # Preprocess data
        preprocessed_df = self._preprocess_data(actions_df)
        shots_df = self._preprocess_shots(actions_df)
        
        # Calculate transition matrix
        self.transition_matrix = self._calculate_transition_matrix(preprocessed_df)
        
        # Calculate shot probability
        self.shot_probability = self._calculate_shot_probability(shots_df)
        
        # Calculate xT values
        self.grid = self._calculate_xt_values()
        
        return self
    
    def calculate_action_value(self, start_x, start_y, end_x, end_y):
        """
        Calculate the xT value of a specific action
        
        Parameters:
        -----------
        start_x, start_y : float
            Starting coordinates of the action
        end_x, end_y : float
            Ending coordinates of the action
            
        Returns:
        --------
        xt_value : float
            The xT value of the action
        """
        # Convert coordinates to grid indices
        start_cell_x = min(int(start_x * self.n_cols / 100), self.n_cols - 1)
        start_cell_y = min(int(start_y * self.n_rows / 100), self.n_rows - 1)
        end_cell_x = min(int(end_x * self.n_cols / 100), self.n_cols - 1)
        end_cell_y = min(int(end_y * self.n_rows / 100), self.n_rows - 1)
        
        # Calculate cell indices
        start_cell = start_cell_y * self.n_cols + start_cell_x
        end_cell = end_cell_y * self.n_cols + end_cell_x
        
        # Calculate xT difference
        return self.grid[end_cell] - self.grid[start_cell]
    
    def add_xt_to_actions(self, actions_df):
        """
        Add xT values to a DataFrame of actions
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            DataFrame containing SPADL actions
            
        Returns:
        --------
        actions_with_xt : pandas.DataFrame
            Actions DataFrame with added xT column
        """
        result_df = actions_df.copy()
        
        # Initialize xT column
        result_df['xt_value'] = 0.0
        
        # Calculate xT for all actions that have valid coordinates
        valid_actions = result_df.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
        
        for idx, row in valid_actions.iterrows():
            xt_value = self.calculate_action_value(
                row['start_x'], row['start_y'], row['end_x'], row['end_y']
            )
            result_df.at[idx, 'xt_value'] = xt_value
        
        return result_df
    
    def plot_xt_grid(self, ax=None, title="Expected Threat (xT) Grid", cmap="viridis"):
        """
        Plot the xT grid as a heatmap
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        title : str
            Title for the plot
        cmap : str
            Colormap for the heatmap
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Reshape grid to 2D
        grid_2d = self.grid.reshape(self.n_rows, self.n_cols)
        
        # Create heatmap
        sns.heatmap(grid_2d, ax=ax, cmap=cmap, annot=False)
        
        # Set labels and title
        ax.set_title(title)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        
        return ax
    
    def get_player_xt_contributions(self, actions_df, player_id=None):
        """
        Calculate xT contributions for players
        
        Parameters:
        -----------
        actions_df : pandas.DataFrame
            DataFrame containing SPADL actions with xT values
        player_id : str, optional
            If provided, return only this player's contributions
            
        Returns:
        --------
        player_contributions : pandas.DataFrame
            DataFrame with player xT contributions
        """
        # Ensure actions have xT values
        if 'xt_value' not in actions_df.columns:
            actions_df = self.add_xt_to_actions(actions_df)
        
        # Filter for successful actions
        valid_actions = actions_df[actions_df['result'] == 'success'].copy()
        
        # Group by player and calculate total xT
        player_xt = valid_actions.groupby('player_id')['xt_value'].agg(['sum', 'count', 'mean'])
        player_xt.columns = ['total_xt', 'actions_count', 'avg_xt_per_action']
        
        # Sort by total xT
        player_xt = player_xt.sort_values('total_xt', ascending=False)
        
        if player_id:
            return player_xt.loc[[player_id]]
        
        return player_xt
    

def load_spadl_actions_from_db(conn, match_id=None):
    """
    Load SPADL actions from the database using psycopg2
    
    Parameters:
    -----------
    conn : psycopg2.extensions.connection
        Database connection
    match_id : str, optional
        If provided, load actions only for this match
        
    Returns:
    --------
    actions_df : pandas.DataFrame
        DataFrame containing SPADL actions
    """
    # Build the SQL query
    query = """
    SELECT 
        id, game_id, period_id, seconds, player_id, team_id,
        start_x, start_y, end_x, end_y, action_type, result, bodypart
    FROM spadl_actions
    """
    
    if match_id:
        query += f" WHERE game_id = '{match_id}'"
    
    # Execute the query
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query)
    
    # Fetch all records
    records = cursor.fetchall()
    
    # Convert to DataFrame
    actions_df = pd.DataFrame(records)
    

    
    return actions_df

def load_player_data(conn):
    """
    Load player data from the database
    """
    query = """
    SELECT 
        player_id, player_name, team_id, jersey_number
    FROM players
    """
    
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query)
    
    records = cursor.fetchall()
    players_df = pd.DataFrame(records)
    
    
    return players_df

def load_team_data(conn):
    """
    Load team data from the database
    """
    query = """
    SELECT 
        team_id, team_name
    FROM teams
    """
    
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query)
    
    records = cursor.fetchall()
    teams_df = pd.DataFrame(records)
    

    
    return teams_df

def main():
    """
    Main function to run the xT model
    """
    # Connect to the database
    conn = connect_to_db()
    
    try:
        # Load data
        print("Loading actions data from database...")
        actions_df = load_spadl_actions_from_db(conn)
        
        # Check if data was loaded successfully
        if actions_df.empty:
            print("No actions data found in the database!")
            return
            
        print(f"Loaded {len(actions_df)} actions from database")
        
        # Print data sample
        print("\nSample of loaded actions data:")
        print(actions_df.head())
        
        # Print column information
        print("\nColumns in actions data:")
        print(actions_df.columns.tolist())
        
        # Print data types
        print("\nData types:")
        print(actions_df.dtypes)
        
        # Verify essential columns exist
        essential_columns = ['start_x', 'start_y', 'end_x', 'end_y', 'result', 'player_id']
        missing_columns = [col for col in essential_columns if col not in actions_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing essential columns: {missing_columns}")
            print("Please ensure these columns exist in your database.")
            return
            
        # Check for null values in essential columns
        null_counts = actions_df[essential_columns].isnull().sum()
        print("\nNull values in essential columns:")
        print(null_counts)
        
        # Create and fit the xT model
        print("\nFitting Expected Threat model...")
        xt_model = ExpectedThreat(n_rows=12, n_cols=16, smoothing=True)
        xt_model.fit(actions_df)
        
        # Add xT values to actions
        print("Calculating xT values for all actions...")
        actions_with_xt = xt_model.add_xt_to_actions(actions_df)
        
        # Print some xT values
        print("\nSample of actions with xT values:")
        print(actions_with_xt[['id', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y', 'xt_value']].head(10))
        
        # Calculate player contributions
        print("\nCalculating player xT contributions...")
        player_contributions = xt_model.get_player_xt_contributions(actions_with_xt)
        
        # Print top player contributions
        print("\nTop 10 players by xT contribution:")
        print(player_contributions.head(10))
        
        # Try to add player names if available
        try:
            players_df = load_player_data(conn)
            teams_df = load_team_data(conn)
            
            player_contributions_with_names = player_contributions.reset_index().merge(
                players_df, on='player_id', how='left'
            )
            
            player_contributions_with_names = player_contributions_with_names.merge(
                teams_df, on='team_id', how='left'
            )
            
            print("\nTop 10 players by xT contribution (with names):")
            print(player_contributions_with_names[['player_name', 'team_name', 'total_xt', 'actions_count', 'avg_xt_per_action']].head(10))
            
            # Save to CSV
            player_contributions_with_names.to_csv('player_xt_contributions.csv', index=False)
            print("\nSaved player contributions to 'player_xt_contributions.csv'")
        except Exception as e:
            print(f"\nCouldn't load player/team data: {e}")
            print("Saving contributions without names...")
            player_contributions.to_csv('player_xt_contributions.csv')
        
        # Visualization
        print("\nCreating xT grid visualization...")
        fig, ax = plt.subplots(figsize=(12, 8))
        xt_model.plot_xt_grid(ax=ax)
        plt.tight_layout()
        plt.savefig('xt_grid.png')
        print("Saved xT grid visualization to 'xt_grid.png'")
        
        # Optionally display the visualization
        plt.show()
        
        print("\nAnalysis complete!")
        
        return xt_model, actions_with_xt
        
    except Exception as e:
        print(f"Error running the xT model: {e}")
        raise
        
    finally:
        # Close the connection
        print("Database connection closed")

if __name__ == "__main__":
    main()