import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection


def load_gesture_labels(labels_path: str = "../data/labels.csv") -> dict:
    """
    Load gesture labels from CSV file.
    
    Args:
        labels_path: Path to labels.csv file
        
    Returns:
        Dictionary mapping gesture_label to gesture_name
    """
    try:
        labels_df = pd.read_csv(labels_path)
        return dict(zip(labels_df['GestureIndex'].astype(str), labels_df['GestureName']))
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {labels_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load labels: {e}")
        return {}


def load_irds_data(
    folder_path: str = "../data",
    file_pattern: str = "*.txt",
    has_header: bool = False,
    add_metadata: bool = True,
    columns: Optional[List[str]] = None,
    include_source_file: bool = True,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load IRDS dataset files into a combined pandas DataFrame.

    - Reads all files matching `file_pattern` in `folder_path`.
    - Assumes delimited text files readable by pandas `read_csv` (default no header).
    - Optionally parses filename metadata of the form:
      subject_date_gesture_rep_correct_position.txt

    Args:
        folder_path: Directory containing data files.
        file_pattern: Glob pattern for files (e.g., '*.txt').
        has_header: If True, use first row as header; else create numeric columns.
        add_metadata: If True, parse and add filename-derived metadata columns.
        columns: Optional explicit column names for the sensor columns; must match
                 number of columns in the files when `has_header` is False.

    Returns:
        Combined pandas DataFrame with all rows and optional metadata columns.
    """
    search_path = os.path.join(folder_path, file_pattern)
    all_files = glob.glob(search_path)
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found at {search_path}")
    
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


def filter_dataframe(
    df: pd.DataFrame,
    *,
    source_file: Optional[str] = None,
    subject_id: Optional[str] = None,
    date_id: Optional[str] = None,
    gesture_label: Optional[str] = None,
    rep_number: Optional[str] = None,
    correct_label: Optional[str] = None,
    position: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter the combined DataFrame by optional metadata fields.
    Values must match exactly; pass None to skip a filter.
    """
    mask = pd.Series(True, index=df.index)
    if source_file is not None and "source_file" in df.columns:
        mask &= df["source_file"] == source_file
    if subject_id is not None and "subject_id" in df.columns:
        mask &= df["subject_id"] == subject_id
    if date_id is not None and "date_id" in df.columns:
        mask &= df["date_id"] == date_id
    if gesture_label is not None and "gesture_label" in df.columns:
        mask &= df["gesture_label"] == gesture_label
    if rep_number is not None and "rep_number" in df.columns:
        mask &= df["rep_number"] == rep_number
    if correct_label is not None and "correct_label" in df.columns:
        mask &= df["correct_label"] == correct_label
    if position is not None and "position" in df.columns:
        mask &= df["position"] == position
    return df.loc[mask]


def animate_3d_sequence(
    df: pd.DataFrame,
    x_cols: Tuple[str, str, str],
    frame_col: Optional[str] = None,
    max_rows: Optional[int] = 500,
    interval_ms: int = 100,
    point_size: int = 20,
    elev: int = 20,
    azim: int = -60,
    save_path: Optional[str] = None,
    dpi: int = 100,
    gesture_labels: Optional[dict] = None,
) -> animation.FuncAnimation:
    """
    Animate 3D points from a DataFrame as a time sequence.

    Args:
        df: Input DataFrame containing at least the x/y/z columns.
        x_cols: Tuple of column names for (x, y, z).
        frame_col: Optional column that groups rows by frame/time; if None,
                   the first axis (row order) is used as the sequence.
        max_rows: Maximum number of sequential rows/frames to animate.
        interval_ms: Delay between frames in milliseconds.
        point_size: Matplotlib scatter point size.
        elev: 3D elevation angle.
        azim: 3D azimuth angle.
        save_path: If provided, save the animation as an mp4/gif by extension.
        dpi: Resolution used when saving.

    Returns:
        Matplotlib FuncAnimation object.
    """
    if max_rows is not None:
        df_seq = df.iloc[:max_rows].copy()
    else:
        df_seq = df.copy()

    x_name, y_name, z_name = x_cols
    if frame_col is not None:
        # Ensure frame ordering is stable
        df_seq = df_seq.sort_values(by=frame_col, kind="stable")

    # Keep only rows with valid numeric values in the required columns
    df_seq = df_seq[pd.to_numeric(df_seq[x_name], errors="coerce").notna()]
    df_seq = df_seq[pd.to_numeric(df_seq[y_name], errors="coerce").notna()]
    df_seq = df_seq[pd.to_numeric(df_seq[z_name], errors="coerce").notna()]
    if len(df_seq) == 0:
        raise ValueError(
            f"No valid numeric data to plot for columns {x_name}, {y_name}, {z_name}."
        )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # Axis limits from data for a stable viewport
    x_min, x_max = np.nanmin(df_seq[x_name].values), np.nanmax(df_seq[x_name].values)
    y_min, y_max = np.nanmin(df_seq[y_name].values), np.nanmax(df_seq[y_name].values)
    z_min, z_max = np.nanmin(df_seq[z_name].values), np.nanmax(df_seq[z_name].values)
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    # Initialize with the first point to ensure something renders immediately
    first_row = df_seq.iloc[0]
    scatter = ax.scatter([first_row[x_name]], [first_row[y_name]], [first_row[z_name]], s=point_size, c="tab:blue")

    # Determine frames
    if frame_col is not None:
        frame_values = df_seq[frame_col].values
    else:
        frame_values = np.arange(len(df_seq))

    def init():
        scatter._offsets3d = (
            np.array([first_row[x_name]]),
            np.array([first_row[y_name]]),
            np.array([first_row[z_name]]),
        )
        return (scatter,)

    def update(frame_idx):
        if frame_col is not None:
            row = df_seq.iloc[frame_idx]
        else:
            row = df_seq.iloc[frame_idx]
        x = row[x_name]
        y = row[y_name]
        z = row[z_name]
        scatter._offsets3d = (np.array([x]), np.array([y]), np.array([z]))
        
        # Create title with gesture label, subject ID, repetition, and position if available
        title = f"Frame {frame_idx}"
        if gesture_labels and "gesture_label" in row:
            gesture_label = str(row["gesture_label"])
            gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
            
            # Build title components
            title_parts = []
            if "subject_id" in row:
                title_parts.append(f"Subject {row['subject_id']}")
            title_parts.append(gesture_name)
            if "rep_number" in row:
                title_parts.append(f"Rep {row['rep_number']}")
            if "position" in row:
                title_parts.append(f"Position: {row['position']}")
            title_parts.append(f"Frame {frame_idx}")
            
            title = " -- ".join(title_parts)
        ax.set_title(title)
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(df_seq),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = None
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=max(1, int(1000 / max(interval_ms, 1))))
        elif ext == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(1000 / max(interval_ms, 1))))
        anim.save(save_path, writer=writer, dpi=dpi)
    else:
        # If no save path and not showing, just create a static plot to avoid animation warning
        # This prevents the "Animation was deleted without rendering" warning
        # Create a static frame instead of animation
        update(0)  # Render the first frame
        plt.close(fig)

    return anim


def infer_xyz_triplets(
    df: pd.DataFrame,
    num_joints: int = 25,
    start_col: int = 0,
    order: str = "xyz",
) -> List[Tuple[str, str, str]]:
    """
    Infer (x,y,z) triplets for a skeleton from numeric columns.

    Args:
        df: DataFrame containing numeric columns for joints.
        num_joints: Number of joints to use (25 => 75 columns expected).
        start_col: Index of the first numeric column to start grouping from.
        order: Order of axes in triplets: one of {'xyz','xzy','yxz','yzx','zxy','zyx'}.

    Returns:
        List of (x_col, y_col, z_col) tuples.
    """
    if order not in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}:
        raise ValueError("order must be a permutation of 'x', 'y', 'z'")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) < start_col + num_joints * 3:
        raise ValueError(
            f"Not enough numeric columns: have {len(numeric_cols)}, need at least {start_col + num_joints * 3}"
        )

    axis_indices = {"x": 0, "y": 1, "z": 2}
    order_indices = [axis_indices[c] for c in order]

    triplets: List[Tuple[str, str, str]] = []
    base = start_col
    for _ in range(num_joints):
        cols = numeric_cols[base:base + 3]
        # Reorder according to desired order
        x_col, y_col, z_col = cols[order_indices[0]], cols[order_indices[1]], cols[order_indices[2]]
        triplets.append((x_col, y_col, z_col))
        base += 3
    return triplets


def get_human_skeleton_edges() -> List[Tuple[int, int]]:
    """
    Define human skeleton bone connections for 25 joints based on the provided connections.txt.
    Joint mapping from joints.txt:
    0=SpineBase, 1=SpineMid, 2=Neck, 3=Head, 4=ShoulderLeft, 5=ElbowLeft, 6=WristLeft, 7=HandLeft,
    8=ShoulderRight, 9=ElbowRight, 10=WristRight, 11=HandRight, 12=HipLeft, 13=KneeLeft, 14=AnkleLeft,
    15=FootLeft, 16=HipRight, 17=KneeRight, 18=AnkleRight, 19=FootRight, 20=SpineShoulder,
    21=HandTipLeft, 22=ThumbLeft, 23=HandTipRight, 24=ThumbRight
    """
    connections = [
        (3, 2), (2, 20), (20, 1), (1, 0),  # Head to Neck to Spine
        (20, 4), (20, 8),  # Shoulder connections
        (0, 12), (0, 16),  # Hip connections
        (16, 17), (17, 18), (18, 19),  # Right leg
        (12, 13), (13, 14), (14, 15),  # Left leg
        (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Left arm
        (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Right arm
    ]
    return connections


def animate_3d_skeleton_sequence(
    df: pd.DataFrame,
    xyz_cols: List[Tuple[str, str, str]],
    frame_col: Optional[str] = None,
    max_rows: Optional[int] = 500,
    interval_ms: int = 100,
    point_size: int = 20,
    elev: int = 20,
    azim: int = -60,
    edges: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    dpi: int = 100,
    gesture_labels: Optional[dict] = None,
    interactive: bool = True,
) -> animation.FuncAnimation:
    """
    Animate a skeleton as multiple 3D joints per frame.

    Args:
        df: Input DataFrame containing numeric joint columns.
        xyz_cols: List of (x,y,z) column names per joint in order.
        frame_col: Optional frame/time column.
        max_rows: Number of frames to animate.
        interval_ms: Delay between frames.
        point_size: Scatter size for joints.
        elev, azim: Camera angles.
        edges: Optional list of (i,j) pairs to draw bones between joints.
        save_path: Optional output path for animation (.gif/.mp4).
        dpi: Output resolution when saving.
    """
    if max_rows is not None:
        df_seq = df.iloc[:max_rows].copy()
    else:
        df_seq = df.copy()

    if frame_col is not None:
        df_seq = df_seq.sort_values(by=frame_col, kind="stable")

    # Drop rows with NaNs across any required columns
    needed_cols = [c for trip in xyz_cols for c in trip]
    df_seq = df_seq.dropna(subset=needed_cols)
    if len(df_seq) == 0:
        raise ValueError("No valid rows after dropping NaNs for skeleton columns.")

    # Precompute arrays per axis
    X = np.stack([df_seq[x].to_numpy(dtype=float) for (x, _, _) in xyz_cols], axis=1)  # (T, J)
    Y = np.stack([df_seq[y].to_numpy(dtype=float) for (_, y, _) in xyz_cols], axis=1)
    Z = np.stack([df_seq[z].to_numpy(dtype=float) for (_, _, z) in xyz_cols], axis=1)
    
    # Try coordinate transformation to get upright human pose
    # Swap Y and Z if human is facing down
    Y, Z = Z, Y

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Set better default orientation for human figure display
    # Rotate horizontally clockwise so human faces the screen
    # elev=20 (slightly above), azim=90 (side view, rotated clockwise)
    ax.view_init(elev=20, azim=90)
    
    # Enable interactive rotation and better visualization
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Z (swapped)', fontsize=12)  # Note: Y and Z were swapped
    ax.set_zlabel('Y (swapped)', fontsize=12)  # Note: Y and Z were swapped
    ax.grid(True, alpha=0.3)
    
    # No on-figure buttons; keep plot area larger

    # Compute limits across all frames/joints for stable viewport
    ax.set_xlim3d(np.nanmin(X), np.nanmax(X))
    ax.set_ylim3d(np.nanmin(Y), np.nanmax(Y))
    ax.set_zlim3d(np.nanmin(Z), np.nanmax(Z))

    scatter = ax.scatter(X[0], Y[0], Z[0], s=point_size, c="red")
    
    # Add correctness indicator - positioned lower to avoid title overlap
    correctness_text = ax.text2D(0.02, 0.85, "", transform=ax.transAxes, fontsize=24, 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

    # Use proper human skeleton connections
    edges = get_human_skeleton_edges()
    
    # Create bone lines connecting joints
    line_artists = []
    for (i, j) in edges:
        line, = ax.plot([X[0, i], X[0, j]], [Y[0, i], Y[0, j]], [Z[0, i], Z[0, j]], c="darkred", lw=2, alpha=0.8)
        line_artists.append(line)

    def init():
        scatter._offsets3d = (X[0], Y[0], Z[0])
        if edges:
            for line, (i, j) in zip(line_artists, edges):
                line.set_data([X[0, i], X[0, j]], [Y[0, i], Y[0, j]])
                line.set_3d_properties([Z[0, i], Z[0, j]])
        
        # Set initial correctness indicator
        if len(df_seq) > 0:
            row = df_seq.iloc[0]
            if "correct_label" in row:
                correct_label = str(row["correct_label"])
                if correct_label == "1":
                    correctness_text.set_text("✓")
                    correctness_text.set_color("green")
                else:
                    correctness_text.set_text("✗")
                    correctness_text.set_color("red")
            else:
                correctness_text.set_text("")
        return (scatter, *line_artists, correctness_text)

    def update(frame_idx):
        scatter._offsets3d = (X[frame_idx], Y[frame_idx], Z[frame_idx])
        if edges:
            for line, (i, j) in zip(line_artists, edges):
                line.set_data([X[frame_idx, i], X[frame_idx, j]], [Y[frame_idx, i], Y[frame_idx, j]])
                line.set_3d_properties([Z[frame_idx, i], Z[frame_idx, j]])
        
        # Create title with gesture label, subject ID, repetition, and position if available
        title = f"Frame {frame_idx}"
        if gesture_labels and len(df_seq) > frame_idx:
            row = df_seq.iloc[frame_idx]
            if "gesture_label" in row:
                gesture_label = str(row["gesture_label"])
                gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
                
                # Build title components
                title_parts = []
                if "subject_id" in row:
                    title_parts.append(f"Subject {row['subject_id']}")
                title_parts.append(gesture_name)
                if "rep_number" in row:
                    title_parts.append(f"Rep {row['rep_number']}")
                if "position" in row:
                    title_parts.append(f"Position: {row['position']}")
                title_parts.append(f"Frame {frame_idx}")
                
                title = " -- ".join(title_parts)
        
        # Update correctness indicator
        if len(df_seq) > frame_idx:
            row = df_seq.iloc[frame_idx]
            if "correct_label" in row:
                correct_label = str(row["correct_label"])
                if correct_label == "1":
                    correctness_text.set_text("✓")
                    correctness_text.set_color("green")
                else:
                    correctness_text.set_text("✗")
                    correctness_text.set_color("red")
            else:
                correctness_text.set_text("")
        
        ax.set_title(title)
        return (scatter, *line_artists, correctness_text)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=X.shape[0],
        interval=interval_ms,
        blit=False,
        repeat=True,  # keep loop enabled
    )
    # Keyboard controls only
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

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = None
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=max(1, int(1000 / max(interval_ms, 1))))
        elif ext == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(1000 / max(interval_ms, 1))))
        anim.save(save_path, writer=writer, dpi=dpi)
    else:
        # If no save path and not showing, just create a static plot to avoid animation warning
        # This prevents the "Animation was deleted without rendering" warning
        # Create a static frame instead of animation
        update(0)  # Render the first frame
        plt.close(fig)

    return anim


if __name__ == "__main__":
    # Example usage (will not run if imported):
    # df_all = load_irds_data()
    # anim = animate_3d_sequence(df_all, ("x", "y", "z"), max_rows=200)
    # plt.show()
    pass

def run_3d_visualization(
    folder_path: str = "../data",
    file_pattern: str = "*.txt",
    has_header: bool = False,
    add_metadata: bool = True,
    columns: Optional[List[str]] = None,
    x_cols: Optional[Tuple[str, str, str]] = None,
    frame_col: Optional[str] = None,
    max_rows: Optional[int] = 500,
    interval_ms: int = 100,
    point_size: int = 20,
    elev: int = 20,
    azim: int = -60,
    save_path: Optional[str] = None,
    dpi: int = 100,
    show: bool = True,
    skeleton: bool = False,
    num_joints: int = 25,
    start_col: int = 0,
    order: str = "xyz",
    connect: bool = False,
    # Filters
    source_file: Optional[str] = None,
    subject_id: Optional[str] = None,
    date_id: Optional[str] = None,
    gesture_label: Optional[str] = None,
    rep_number: Optional[str] = None,
    correct_label: Optional[str] = None,
    position: Optional[str] = None,
    # Gesture labels
    labels_path: str = "../data/labels.csv",
    # Performance optimization
    max_files: Optional[int] = 10,
):
    """
    Convenience wrapper to load the dataset and run the 3D animation.

    Args mirror those of `load_irds_data` and `animate_3d_sequence`.

    Returns:
        (animation.FuncAnimation, pd.DataFrame): The animation and loaded DataFrame.
    """
    df = load_irds_data(
        folder_path=folder_path,
        file_pattern=file_pattern,
        has_header=has_header,
        add_metadata=add_metadata,
        columns=columns,
        include_source_file=True,
        max_files=max_files,
    )

    # Load gesture labels
    gesture_labels = load_gesture_labels(labels_path)

    # Apply optional filters to focus on a single sequence/file
    original_count = len(df)
    df = filter_dataframe(
        df,
        source_file=source_file,
        subject_id=subject_id,
        date_id=date_id,
        gesture_label=gesture_label,
        rep_number=rep_number,
        correct_label=correct_label,
        position=position,
    )
    
    if df.empty:
        print(f"Warning: No rows left after applying filters (started with {original_count} rows)")
        print("Available data summary:")
        if original_count > 0:
            # Show what data is available
            df_original = load_irds_data(
                folder_path=folder_path,
                file_pattern=file_pattern,
                has_header=has_header,
                add_metadata=add_metadata,
                columns=columns,
                include_source_file=True,
                max_files=max_files,
            )
            print(f"  - Unique subjects: {df_original['subject_id'].unique()[:5]}")
            print(f"  - Unique gestures: {df_original['gesture_label'].unique()[:5]}")
            print(f"  - Unique positions: {df_original['position'].unique()[:5]}")
            print(f"  - Sample files: {df_original['source_file'].unique()[:3]}")
        print("Try removing some filters or using different values.")
        # Use original data without filters as fallback
        df = load_irds_data(
            folder_path=folder_path,
            file_pattern=file_pattern,
            has_header=has_header,
            add_metadata=add_metadata,
            columns=columns,
            include_source_file=True,
            max_files=max_files,
        )
        print(f"Using unfiltered data with {len(df)} rows")

    # Choose default x/y/z columns if not provided: first three numeric columns
    if x_cols is None:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        # Auto-enable skeleton mode if we have at least 75 numeric columns (25 joints * 3)
        if not skeleton and len(numeric_cols) >= 75:
            skeleton = True
            num_joints = 25
        if skeleton:
            # When skeleton is chosen, x_cols is ignored in favor of triplets
            x_cols = None
        else:
            if len(numeric_cols) < 3:
                # Fallback to first three columns if numeric inference fails
                fallback = list(df.columns[:3])
                if len(fallback) < 3:
                    raise ValueError("DataFrame has fewer than 3 columns to plot.")
                x_cols = (fallback[0], fallback[1], fallback[2])
            else:
                x_cols = (numeric_cols[0], numeric_cols[1], numeric_cols[2])

    if skeleton:
        xyz_cols = infer_xyz_triplets(df, num_joints=num_joints, start_col=start_col, order=order)
        edges = None
        if connect:
            # A simple chain connection by index as a default; users can extend later
            edges = [(i, i + 1) for i in range(len(xyz_cols) - 1)]
        anim = animate_3d_skeleton_sequence(
            df=df,
            xyz_cols=xyz_cols,
            frame_col=frame_col,
            max_rows=max_rows,
            interval_ms=interval_ms,
            point_size=point_size,
            elev=elev,
            azim=azim,
            edges=edges,
            save_path=save_path,
            dpi=dpi,
            gesture_labels=gesture_labels,
        )
    else:
        anim = animate_3d_sequence(
            df=df,
            x_cols=x_cols,
            frame_col=frame_col,
            max_rows=max_rows,
            interval_ms=interval_ms,
            point_size=point_size,
            elev=elev,
            azim=azim,
            save_path=save_path,
            dpi=dpi,
            gesture_labels=gesture_labels,
        )

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display animation: {e}")
            print("Animation created but cannot be displayed in this environment")
            plt.close(plt.gcf())
    else:
        # If not showing, close the figure to prevent warnings
        plt.close(plt.gcf())

    return anim, df
