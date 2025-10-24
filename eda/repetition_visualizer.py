"""
Repetition Visualizer - Integrates with existing run_3d_visualization
Shows all repetitions for a subject-gesture combination
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Import our existing functions
import os
import glob
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Copy the necessary functions directly to avoid import issues
def load_gesture_labels(labels_path: str = "../data/labels.csv") -> dict:
    """Load gesture labels from CSV file."""
    try:
        labels_df = pd.read_csv(labels_path)
        return dict(zip(labels_df['GestureIndex'].astype(str), labels_df['GestureName']))
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {labels_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load labels: {e}")
        return {}

def load_irds_data(folder_path: str = "../data",
                   file_pattern: str = "*.txt",
                   has_header: bool = False,
                   add_metadata: bool = True,
                   columns: Optional[List[str]] = None,
                   include_source_file: bool = True,
                   max_files: Optional[int] = None,
                   subject_id: Optional[str] = None,
                   gesture_label: Optional[str] = None) -> pd.DataFrame:
    """Load IRDS dataset files into a combined pandas DataFrame."""
    search_path = os.path.join(folder_path, file_pattern)
    all_files = glob.glob(search_path)
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found at {search_path}")
    
    # Filter files by subject and gesture if specified
    if subject_id is not None or gesture_label is not None:
        filtered_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            name, _ext = os.path.splitext(filename)
            parts = name.split("_")
            
            if len(parts) >= 3:
                file_subject = parts[0]
                file_gesture = parts[2]
                
                # Check if this file matches our criteria
                subject_match = (subject_id is None) or (file_subject == str(subject_id))
                gesture_match = (gesture_label is None) or (file_gesture == str(gesture_label))
                
                if subject_match and gesture_match:
                    filtered_files.append(file_path)
        
        if filtered_files:
            all_files = filtered_files
            print(f"Found {len(all_files)} files matching subject={subject_id}, gesture={gesture_label}")
        else:
            print(f"No files found matching subject={subject_id}, gesture={gesture_label}")
            return pd.DataFrame()
    
    # Limit number of files for faster loading
    if max_files is not None and len(all_files) > max_files:
        print(f"Loading only first {max_files} files out of {len(all_files)} for faster startup")
        all_files = all_files[:max_files]

    header = 0 if has_header else None
    list_of_dfs: List[pd.DataFrame] = []
    
    for file_path in all_files:
        df = pd.read_csv(file_path, header=header)

        # Assign column names if provided and no header present
        if not has_header and columns is not None:
            if len(columns) != df.shape[1]:
                raise ValueError(
                    f"Provided columns length {len(columns)} does not match file columns {df.shape[1]} for {file_path}"
                )
            df.columns = columns

        if add_metadata:
            filename = os.path.basename(file_path)
            name, _ext = os.path.splitext(filename)
            parts = name.split("_")
            # Expected: subject_id, date_id, gesture_label, rep_number, correct_label, position
            if len(parts) >= 6:
                subject_id, date_id, gesture_label, rep_number, correct_label, position = parts[:6]
            else:
                # Fallback: pad missing parts with None
                subject_id = parts[0] if len(parts) > 0 else None
                date_id = parts[1] if len(parts) > 1 else None
                gesture_label = parts[2] if len(parts) > 2 else None
                rep_number = parts[3] if len(parts) > 3 else None
                correct_label = parts[4] if len(parts) > 4 else None
                position = parts[5] if len(parts) > 5 else None

            df["subject_id"] = subject_id
            df["date_id"] = date_id
            df["gesture_label"] = gesture_label
            df["rep_number"] = rep_number
            df["correct_label"] = correct_label
            df["position"] = position

        if include_source_file:
            df["source_file"] = os.path.basename(file_path)

        list_of_dfs.append(df)

    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    return combined_df

def filter_dataframe(df: pd.DataFrame,
                    *,
                    source_file: Optional[str] = None,
                    subject_id: Optional[str] = None,
                    date_id: Optional[str] = None,
                    gesture_label: Optional[str] = None,
                    rep_number: Optional[str] = None,
                    correct_label: Optional[str] = None,
                    position: Optional[str] = None) -> pd.DataFrame:
    """Filter the combined DataFrame by optional metadata fields.

    All comparisons are done as strings to avoid type-mismatch issues (e.g., 0 vs "0").
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure string comparison on both sides
    def col_str(name: str) -> Optional[pd.Series]:
        return df[name].astype(str) if name in df.columns else None

    mask = pd.Series(True, index=df.index)

    if source_file is not None:
        col = col_str("source_file")
        if col is not None:
            mask &= col == str(source_file)
    if subject_id is not None:
        col = col_str("subject_id")
        if col is not None:
            mask &= col == str(subject_id)
    if date_id is not None:
        col = col_str("date_id")
        if col is not None:
            mask &= col == str(date_id)
    if gesture_label is not None:
        col = col_str("gesture_label")
        if col is not None:
            mask &= col == str(gesture_label)
    if rep_number is not None:
        col = col_str("rep_number")
        if col is not None:
            mask &= col == str(rep_number)
    if correct_label is not None:
        col = col_str("correct_label")
        if col is not None:
            mask &= col == str(correct_label)
    if position is not None:
        col = col_str("position")
        if col is not None:
            mask &= col == str(position)

    return df.loc[mask]


def visualize_all_repetitions(subject_id: str, gesture_label: str, 
                             folder_path: str = "../data",
                             max_files: int = 20, max_reps: int = 5):
    """
    Visualize all repetitions for a subject-gesture combination
    
    Args:
        subject_id: Subject ID to analyze
        gesture_label: Gesture label to analyze  
        folder_path: Path to data folder
        max_files: Maximum files to load
        max_reps: Maximum repetitions to show
    """
    
    print(f"Loading data for Subject {subject_id}, Gesture {gesture_label}...")
    
    try:
        # Load data specifically for the subject and gesture
        df = load_irds_data(folder_path=folder_path, 
                           max_files=max_files,
                           subject_id=subject_id,
                           gesture_label=gesture_label)
        
        if df is None or df.empty:
            print(f"No data loaded from {folder_path}")
            return
            
        print(f"Loaded {len(df)} total rows")
        print(f"Columns: {list(df.columns)}")
        
        # Since we already filtered by subject and gesture, use the data directly
        filtered_df = df
        
        if filtered_df is None or filtered_df.empty:
            print(f"No data found for Subject {subject_id}, Gesture {gesture_label}")
            return
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load gesture labels
    gesture_labels = load_gesture_labels()
    gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
    
    # Get unique repetitions, prioritizing both correct and incorrect
    unique_reps = sorted(filtered_df['rep_number'].unique())
    
    # Separate correct and incorrect repetitions
    correct_reps = filtered_df[filtered_df['correct_label'] == '1']['rep_number'].unique()
    incorrect_reps = filtered_df[filtered_df['correct_label'] == '0']['rep_number'].unique()
    
    # Mix correct and incorrect repetitions for variety
    mixed_reps = []
    max_correct = min(len(correct_reps), max_reps // 2 + 1)
    max_incorrect = min(len(incorrect_reps), max_reps // 2 + 1)
    
    # Add correct repetitions
    mixed_reps.extend(sorted(correct_reps)[:max_correct])
    # Add incorrect repetitions
    mixed_reps.extend(sorted(incorrect_reps)[:max_incorrect])
    
    # If we still need more, fill with remaining repetitions
    remaining_reps = [rep for rep in unique_reps if rep not in mixed_reps]
    mixed_reps.extend(remaining_reps[:max_reps - len(mixed_reps)])
    
    unique_reps = mixed_reps[:max_reps]
    print(f"Found {len(unique_reps)} repetitions: {unique_reps}")
    print(f"Correct repetitions: {[rep for rep in unique_reps if rep in correct_reps]}")
    print(f"Incorrect repetitions: {[rep for rep in unique_reps if rep in incorrect_reps]}")
    
    # Create single figure with multiple animated subplots
    n_reps = len(unique_reps)
    
    # Calculate grid layout: 2 rows if more than 5 repetitions, otherwise 1 row
    if n_reps > 5:
        n_rows = 2
        n_cols = (n_reps + 1) // 2  # Ceiling division
        figsize = (5 * n_cols, 8 * n_rows)  # Increased size for better spacing
    else:
        n_rows = 1
        n_cols = n_reps
        figsize = (7 * n_cols, 8)  # Increased size for better spacing
    
    fig = plt.figure(figsize=figsize)
    
    # Store all repetition data
    all_repetitions_data = []
    
    for i, rep_num in enumerate(unique_reps):
        # Get data for this repetition
        rep_data = filtered_df[filtered_df['rep_number'] == rep_num]
        
        if rep_data.empty:
            continue
            
        print(f"Preparing animation for Repetition {rep_num}: {len(rep_data)} frames")
        
        # Get skeleton columns
        numeric_cols = rep_data.select_dtypes(include=[np.number]).columns.tolist()
        metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                        'correct_label', 'position']
        skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
        
        # Reshape skeleton data
        skeleton_data = rep_data[skeleton_cols].values
        num_frames = len(skeleton_data)
        num_joints = 25
        coords_per_joint = 3
        
        skeleton_reshaped = skeleton_data.reshape(num_frames, num_joints, coords_per_joint)
        
        # Store data for animation
        all_repetitions_data.append({
            'rep_num': rep_num,
            'data': skeleton_reshaped,
            'correct_label': rep_data['correct_label'].iloc[0],
            'position': rep_data['position'].iloc[0],
            'num_frames': num_frames
        })
    
    # Create subplots for each repetition with proper grid layout
    axes = []
    for i, rep_data in enumerate(all_repetitions_data):
        if n_rows == 2:
            # Two-row layout
            row = i // n_cols
            col = i % n_cols
            subplot_idx = row * n_cols + col + 1
        else:
            # Single row layout
            subplot_idx = i + 1
        
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection='3d')
        axes.append(ax)
        
        # Set up the plot with title (without correct flag)
        title = f'Rep {rep_data["rep_num"]} ({rep_data["num_frames"]} frames)\nPos: {rep_data["position"]}'
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Add visual indicator for correct/incorrect
        correct_label = rep_data["correct_label"]
        if correct_label == '1':
            indicator = "✓"
            color = "green"
        else:
            indicator = "✗"
            color = "red"
        
        # Add indicator text positioned beside the chart
        ax.text2D(0.02, 0.98, indicator, transform=ax.transAxes, fontsize=20, 
                 color=color, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        
        # Remove axis labels and titles
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set consistent view
        ax.view_init(elev=20, azim=90)
        
        # Remove all axes lines and panes
        ax.grid(False)  # Remove grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Remove axis lines
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color('none')
    
    # Set common axis limits
    all_skeleton_data = np.concatenate([rep['data'] for rep in all_repetitions_data])
    all_X = all_skeleton_data[:, :, 0]
    all_Y = all_skeleton_data[:, :, 2]  # Swapped
    all_Z = all_skeleton_data[:, :, 1]  # Swapped
    
    for ax in axes:
        ax.set_xlim(np.nanmin(all_X), np.nanmax(all_X))
        ax.set_ylim(np.nanmin(all_Y), np.nanmax(all_Y))
        ax.set_zlim(np.nanmin(all_Z), np.nanmax(all_Z))
    
    # Create synchronized animations for all repetitions
    create_multi_skeleton_animation(fig, axes, all_repetitions_data, subject_id, gesture_name)
    
    # Add main title with more space
    fig.suptitle(f'Subject {subject_id} - {gesture_name} - Animated Repetitions', 
                 fontsize=12, fontweight='bold')  # Positioned higher
    plt.tight_layout()
   
    plt.savefig(f'outputs/animated_repetitions_subject_{subject_id}_gesture_{gesture_label}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return filtered_df


def create_multi_skeleton_animation(fig, axes, all_repetitions_data, subject_id, gesture_name):
    """Create synchronized animated skeleton visualizations for multiple repetitions"""
    connections = get_skeleton_connections()
    
    # Initialize artists for each subplot
    all_artists = []
    for i, ax in enumerate(axes):
        # Initialize empty scatter and line objects with new colors
        scatter = ax.scatter([], [], [], c='darkblue', s=8, alpha=0.9)  # Dark blue joints
        line_artists = []
        for _ in connections:
            line, = ax.plot([], [], [], c='red', alpha=0.8, linewidth=0.8)  # Red lines
            line_artists.append(line)
        
        all_artists.append({
            'scatter': scatter,
            'lines': line_artists
        })
    
    def init():
        """Initialize animation"""
        for artists in all_artists:
            # For 3D scatter, we need to set empty 3D coordinates
            artists['scatter']._offsets3d = ([], [], [])
            for line in artists['lines']:
                line.set_data([], [])
                line.set_3d_properties([])
        return [artists['scatter'] for artists in all_artists] + \
               [line for artists in all_artists for line in artists['lines']]
    
    def update(frame_idx):
        """Update animation frame for all repetitions"""
        all_return_artists = []
        
        for i, (ax, rep_data, artists) in enumerate(zip(axes, all_repetitions_data, all_artists)):
            skeleton_reshaped = rep_data['data']
            num_frames = rep_data['num_frames']
            
            if frame_idx >= num_frames:
                # Keep showing last frame
                frame_idx = num_frames - 1
            
            frame_data = skeleton_reshaped[frame_idx]
            
            # Apply coordinate transformation
            X = frame_data[:, 0]
            Y = frame_data[:, 2]  # Swap Y and Z
            Z = frame_data[:, 1]
            
            # Update joints
            artists['scatter']._offsets3d = (X, Y, Z)
            
            # Update skeleton connections
            for j, (start_joint, end_joint) in enumerate(connections):
                if start_joint < len(frame_data) and end_joint < len(frame_data):
                    start_point = [X[start_joint], Y[start_joint], Z[start_joint]]
                    end_point = [X[end_joint], Y[end_joint], Z[end_joint]]
                    artists['lines'][j].set_data([start_point[0], end_point[0]], 
                                               [start_point[1], end_point[1]])
                    artists['lines'][j].set_3d_properties([start_point[2], end_point[2]])
                else:
                    artists['lines'][j].set_data([], [])
                    artists['lines'][j].set_3d_properties([])
            
            all_return_artists.extend([artists['scatter']] + artists['lines'])
        
        return all_return_artists
    
    # Find the maximum number of frames across all repetitions
    max_frames = max(rep['num_frames'] for rep in all_repetitions_data)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=max_frames, interval=150, blit=False, repeat=True
    )
    
    # Add keyboard controls
    is_playing = True
    
    def on_key(event):
        nonlocal is_playing
        if event.key == ' ':  # Spacebar toggles play/pause
            if is_playing:
                anim.pause()
                is_playing = False
            else:
                anim.resume()
                is_playing = True
        elif event.key == 'r':  # Reset to first frame
            anim.pause()
            is_playing = False
            update(0)
            fig.canvas.draw()
        elif event.key == 'p':  # Replay from beginning
            anim.pause()
            update(0)
            fig.canvas.draw()
            anim.event_source.start()
            is_playing = True
        elif event.key == 'escape':  # Close window
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Controls removed as requested
    
    return anim


def get_skeleton_connections():
    """Get skeleton joint connections"""
    return [
        (3, 2), (2, 20), (20, 1), (1, 0),  # Head to Neck to Spine
        (20, 4), (20, 8),  # Shoulder connections
        (0, 12), (0, 16),  # Hip connections
        (16, 17), (17, 18), (18, 19),  # Right leg
        (12, 13), (13, 14), (14, 15),  # Left leg
        (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Left arm
        (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Right arm
    ]


def run_repetition_analysis(subject_id: str, gesture_label: str, 
                           folder_path: str = "../data",
                           max_files: int = 20, max_reps: int = 5):
    """
    Run complete repetition analysis for a subject-gesture combination
    
    Args:
        subject_id: Subject ID to analyze
        gesture_label: Gesture label to analyze
        folder_path: Path to data folder
        max_files: Maximum files to load
    """
    
    print("=" * 60)
    print(f"REPETITION ANALYSIS")
    print(f"Subject: {subject_id}")
    print(f"Gesture: {gesture_label}")
    print("=" * 60)
    
    # Step 1: Visualize all repetitions
    print("\n1. Visualizing all repetitions...")
    try:
        filtered_df = visualize_all_repetitions(subject_id, gesture_label, folder_path, max_files, max_reps)
        
        if filtered_df is None or filtered_df.empty:
            print("No data found for analysis")
            return
    except Exception as e:
        print(f"Error in visualization: {e}")
        return
    
    # Step 2: Show summary statistics
    print("\n2. Summary Statistics:")
    print("-" * 30)
    
    unique_reps = sorted(filtered_df['rep_number'].unique())
    print(f"Total repetitions: {len(unique_reps)}")
    print(f"Repetition numbers: {unique_reps}")
    
    # Count correct vs incorrect
    correct_count = sum(filtered_df['correct_label'] == '1')
    total_count = len(filtered_df)
    print(f"Correct frames: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    
    # Show positions
    positions = filtered_df['position'].unique()
    print(f"Positions: {list(positions)}")
    
    # Show frame counts per repetition
    print("\nFrames per repetition:")
    for rep_num in unique_reps:
        rep_data = filtered_df[filtered_df['rep_number'] == rep_num]
        correct_label = rep_data['correct_label'].iloc[0]
        position = rep_data['position'].iloc[0]
        print(f"  Rep {rep_num}: {len(rep_data)} frames, Correct: {correct_label}, Position: {position}")
    
    # Step 3: Show individual repetition details
    print(f"\n3. Individual repetition details:")
    print("-" * 40)
    
    for rep_num in unique_reps[:3]:  # Show first 3 repetitions
        rep_data = filtered_df[filtered_df['rep_number'] == rep_num]
        if not rep_data.empty:
            correct_label = rep_data['correct_label'].iloc[0]
            position = rep_data['position'].iloc[0]
            num_frames = len(rep_data)
            print(f"Repetition {rep_num}: {num_frames} frames, Correct: {correct_label}, Position: {position}")
            
            # Show file source
            source_files = rep_data['source_file'].unique()
            print(f"  Source files: {list(source_files)}")
    
    print(f"\nTo run individual 3D visualizations, use:")
    print(f"  ./run_3d_visualization config_single_file.yaml")
    print(f"  (Update the config file to filter for subject {subject_id}, gesture {gesture_label}, specific rep)")
    
    print(f"\nRepetition analysis completed for Subject {subject_id}, Gesture {gesture_label}")


if __name__ == "__main__":
    # Example usage
    print("Gesture Repetition Analysis")
    print("=" * 40)
    
    # Analyze repetitions for a specific subject and gesture
    subject_id = "202"  # Change this to your desired subject
    gesture_label = "4"  # Change this to your desired gesture
    
    run_repetition_analysis(subject_id, gesture_label, max_files=100, max_reps=10)  # Show all 10 repetitions
