"""
Gesture Repetition Analysis
Visualize all repetitions for a subject-gesture combination
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Tuple, Optional
import sys

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Import our existing functions
sys.path.append('.')
import importlib.util
spec = importlib.util.spec_from_file_location("irds_eda", os.path.join(os.path.dirname(__file__), "irds-eda.py"))
irds_eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(irds_eda)
load_irds_data = irds_eda.load_irds_data
load_gesture_labels = irds_eda.load_gesture_labels
filter_dataframe = irds_eda.filter_dataframe


def get_gesture_repetitions(df: pd.DataFrame, subject_id: str, gesture_label: str) -> pd.DataFrame:
    """
    Get all repetitions for a specific subject and gesture
    
    Args:
        df: DataFrame with IRDS data
        subject_id: Subject ID to filter
        gesture_label: Gesture label to filter
        
    Returns:
        DataFrame with all repetitions for the subject-gesture combination
    """
    filtered_df = filter_dataframe(
        df, 
        subject_id=subject_id, 
        gesture_label=gesture_label
    )
    
    if filtered_df.empty:
        print(f"No data found for Subject {subject_id}, Gesture {gesture_label}")
        return pd.DataFrame()
    
    # Sort by repetition number and frame order
    filtered_df = filtered_df.sort_values(['rep_number', filtered_df.index])
    
    return filtered_df


def visualize_gesture_repetitions(df: pd.DataFrame, subject_id: str, gesture_label: str,
                                max_reps: int = 5, save_path: Optional[str] = None):
    """
    Visualize all repetitions for a subject-gesture combination
    
    Args:
        df: DataFrame with IRDS data
        subject_id: Subject ID
        gesture_label: Gesture label
        max_reps: Maximum number of repetitions to show
        save_path: Optional path to save the plot
    """
    
    # Get repetitions data
    reps_df = get_gesture_repetitions(df, subject_id, gesture_label)
    
    if reps_df.empty:
        return
    
    # Load gesture labels for display
    gesture_labels = load_gesture_labels()
    gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
    
    # Get unique repetitions
    unique_reps = sorted(reps_df['rep_number'].unique())
    if len(unique_reps) > max_reps:
        unique_reps = unique_reps[:max_reps]
        print(f"Showing first {max_reps} repetitions out of {len(reps_df['rep_number'].unique())}")
    
    # Get skeleton columns
    numeric_cols = [col for col in reps_df.columns if reps_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Create figure with subplots
    n_reps = len(unique_reps)
    fig = plt.figure(figsize=(5 * n_reps, 12))
    
    # Colors for different repetitions
    colors = plt.cm.Set3(torch.linspace(0, 1, n_reps).numpy())
    
    for i, rep_num in enumerate(unique_reps):
        # Get data for this repetition
        rep_data = reps_df[reps_df['rep_number'] == rep_num]
        
        if rep_data.empty:
            continue
            
        # Reshape skeleton data to (frames, joints, coords)
        skeleton_data = rep_data[skeleton_cols].values
        num_frames = len(skeleton_data)
        num_joints = 25
        coords_per_joint = 3
        
        # Reshape to (frames, joints, coords)
        skeleton_reshaped = skeleton_data.reshape(num_frames, num_joints, coords_per_joint)
        
        # Create 3D subplot
        ax = fig.add_subplot(2, n_reps, i + 1, projection='3d')
        
        # Plot skeleton for each frame (overlay all frames)
        for frame_idx in range(0, num_frames, max(1, num_frames // 10)):  # Sample frames
            frame_data = skeleton_reshaped[frame_idx]
            
            # Apply coordinate transformation (same as in visualization)
            X = frame_data[:, 0]
            Y = frame_data[:, 2]  # Swap Y and Z
            Z = frame_data[:, 1]
            
            # Plot joints
            ax.scatter(X, Y, Z, c=colors[i], s=20, alpha=0.6)
            
            # Plot skeleton connections
            connections = get_skeleton_connections()
            for (start_joint, end_joint) in connections:
                if start_joint < len(frame_data) and end_joint < len(frame_data):
                    start_point = [X[start_joint], Y[start_joint], Z[start_joint]]
                    end_point = [X[end_joint], Y[end_joint], Z[end_joint]]
                    ax.plot([start_point[0], end_point[0]], 
                           [start_point[1], end_point[1]], 
                           [start_point[2], end_point[2]], 
                           c=colors[i], alpha=0.4, linewidth=1)
        
        # Set up the plot
        ax.set_title(f'Rep {rep_num} ({num_frames} frames)', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Z (swapped)')
        ax.set_zlabel('Y (swapped)')
        
        # Set consistent view
        ax.view_init(elev=20, azim=90)
        
        # Set axis limits based on all data
        all_data = torch.tensor(reps_df[skeleton_cols].values, dtype=torch.float32).reshape(-1, num_joints, coords_per_joint)
        all_X = all_data[:, :, 0]
        all_Y = all_data[:, :, 2]  # Swapped
        all_Z = all_data[:, :, 1]  # Swapped
        
        ax.set_xlim(torch.min(all_X[~torch.isnan(all_X)]).item(), torch.max(all_X[~torch.isnan(all_X)]).item())
        ax.set_ylim(torch.min(all_Y[~torch.isnan(all_Y)]).item(), torch.max(all_Y[~torch.isnan(all_Y)]).item())
        ax.set_zlim(torch.min(all_Z[~torch.isnan(all_Z)]).item(), torch.max(all_Z[~torch.isnan(all_Z)]).item())
    
    # Create trajectory plot
    ax_traj = fig.add_subplot(2, 1, 2)
    
    # Plot trajectory for each repetition
    for i, rep_num in enumerate(unique_reps):
        rep_data = reps_df[reps_df['rep_number'] == rep_num]
        
        if rep_data.empty:
            continue
            
        # Get center of mass trajectory
        skeleton_data = torch.tensor(rep_data[skeleton_cols].values, dtype=torch.float32)
        skeleton_reshaped = skeleton_data.reshape(len(skeleton_data), num_joints, coords_per_joint)
        
        # Calculate center of mass for each frame
        com_trajectory = torch.mean(skeleton_reshaped, dim=1)  # (frames, 3)
        
        # Apply coordinate transformation
        X_traj = com_trajectory[:, 0].numpy()
        Y_traj = com_trajectory[:, 2].numpy()  # Swapped
        Z_traj = com_trajectory[:, 1].numpy()  # Swapped
        
        # Plot trajectory
        ax_traj.plot(X_traj, Y_traj, c=colors[i], label=f'Rep {rep_num}', linewidth=2, alpha=0.8)
        ax_traj.scatter(X_traj[0], Y_traj[0], c=colors[i], s=100, marker='o', label=f'Start Rep {rep_num}')
        ax_traj.scatter(X_traj[-1], Y_traj[-1], c=colors[i], s=100, marker='s', label=f'End Rep {rep_num}')
    
    ax_traj.set_title(f'Center of Mass Trajectory - Subject {subject_id}, {gesture_name}')
    ax_traj.set_xlabel('X')
    ax_traj.set_ylabel('Z (swapped)')
    ax_traj.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_traj.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def analyze_repetition_variability(df: pd.DataFrame, subject_id: str, gesture_label: str):
    """
    Analyze variability across repetitions for a subject-gesture combination
    
    Args:
        df: DataFrame with IRDS data
        subject_id: Subject ID
        gesture_label: Gesture label
    """
    
    # Get repetitions data
    reps_df = get_gesture_repetitions(df, subject_id, gesture_label)
    
    if reps_df.empty:
        return
    
    # Load gesture labels
    gesture_labels = load_gesture_labels()
    gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
    
    # Get skeleton columns
    numeric_cols = [col for col in reps_df.columns if reps_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Analyze each repetition
    rep_stats = []
    
    for rep_num in sorted(reps_df['rep_number'].unique()):
        rep_data = reps_df[reps_df['rep_number'] == rep_num]
        
        if rep_data.empty:
            continue
            
        # Get skeleton data
        skeleton_data = rep_data[skeleton_cols].values
        num_frames = len(skeleton_data)
        
        # Calculate statistics
        rep_stat = {
            'rep_number': rep_num,
            'num_frames': num_frames,
            'correct_label': rep_data['correct_label'].iloc[0],
            'position': rep_data['position'].iloc[0]
        }
        
        # Calculate movement range (max - min for each joint)
        skeleton_reshaped = skeleton_data.reshape(num_frames, 25, 3)
        joint_ranges = []
        
        for joint_idx in range(25):
            joint_data = skeleton_reshaped[:, joint_idx, :]
            joint_range = torch.max(joint_data, dim=0)[0] - torch.min(joint_data, dim=0)[0]
            joint_ranges.append(torch.norm(joint_range).item())  # Euclidean norm
        
        rep_stat['avg_joint_range'] = torch.mean(torch.tensor(joint_ranges)).item()
        rep_stat['max_joint_range'] = torch.max(torch.tensor(joint_ranges)).item()
        rep_stat['min_joint_range'] = torch.min(torch.tensor(joint_ranges)).item()
        
        # Calculate center of mass movement
        com_trajectory = torch.mean(skeleton_reshaped, dim=1)
        com_range = torch.max(com_trajectory, dim=0)[0] - torch.min(com_trajectory, dim=0)[0]
        rep_stat['com_range'] = torch.norm(com_range).item()
        
        rep_stats.append(rep_stat)
    
    # Create analysis DataFrame
    stats_df = pd.DataFrame(rep_stats)
    
    # Print summary
    print(f"\nRepetition Analysis - Subject {subject_id}, {gesture_name}")
    print("=" * 60)
    print(f"Total repetitions: {len(stats_df)}")
    print(f"Average frames per repetition: {stats_df['num_frames'].mean():.1f}")
    print(f"Correct repetitions: {sum(stats_df['correct_label'] == '1')}")
    print(f"Positions: {stats_df['position'].unique()}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Frame count per repetition
    axes[0, 0].bar(stats_df['rep_number'], stats_df['num_frames'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Frames per Repetition')
    axes[0, 0].set_xlabel('Repetition Number')
    axes[0, 0].set_ylabel('Number of Frames')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Joint movement range
    axes[0, 1].plot(stats_df['rep_number'], stats_df['avg_joint_range'], 'o-', label='Average', linewidth=2)
    axes[0, 1].plot(stats_df['rep_number'], stats_df['max_joint_range'], 's-', label='Maximum', linewidth=2)
    axes[0, 1].plot(stats_df['rep_number'], stats_df['min_joint_range'], '^-', label='Minimum', linewidth=2)
    axes[0, 1].set_title('Joint Movement Range')
    axes[0, 1].set_xlabel('Repetition Number')
    axes[0, 1].set_ylabel('Movement Range')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Center of mass movement
    axes[1, 0].bar(stats_df['rep_number'], stats_df['com_range'], color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Center of Mass Movement Range')
    axes[1, 0].set_xlabel('Repetition Number')
    axes[1, 0].set_ylabel('COM Range')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correctness
    correctness = stats_df['correct_label'].map({'1': 'Correct', '0': 'Incorrect'})
    axes[1, 1].bar(stats_df['rep_number'], [1 if c == 'Correct' else 0 for c in correctness], 
                   color=['green' if c == 'Correct' else 'red' for c in correctness], alpha=0.7)
    axes[1, 1].set_title('Repetition Correctness')
    axes[1, 1].set_xlabel('Repetition Number')
    axes[1, 1].set_ylabel('Correct (1) / Incorrect (0)')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/repetition_analysis_subject_{subject_id}_gesture_{gesture_label}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df


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


def compare_subjects_gesture(df: pd.DataFrame, gesture_label: str, max_subjects: int = 3):
    """
    Compare the same gesture across different subjects
    
    Args:
        df: DataFrame with IRDS data
        gesture_label: Gesture label to compare
        max_subjects: Maximum number of subjects to show
    """
    
    # Get all subjects who performed this gesture
    gesture_df = filter_dataframe(df, gesture_label=gesture_label)
    subjects = gesture_df['subject_id'].unique()[:max_subjects]
    
    if len(subjects) == 0:
        print(f"No subjects found for gesture {gesture_label}")
        return
    
    # Load gesture labels
    gesture_labels = load_gesture_labels()
    gesture_name = gesture_labels.get(gesture_label, f"Gesture {gesture_label}")
    
    print(f"Comparing {gesture_name} across {len(subjects)} subjects: {subjects}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, len(subjects), figsize=(5 * len(subjects), 10))
    if len(subjects) == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.Set1(torch.linspace(0, 1, len(subjects)).numpy())
    
    for i, subject_id in enumerate(subjects):
        # Get data for this subject
        subject_df = get_gesture_repetitions(df, subject_id, gesture_label)
        
        if subject_df.empty:
            continue
        
        # Get skeleton columns
        numeric_cols = [col for col in subject_df.columns if subject_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                        'correct_label', 'position']
        skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
        
        # Plot first repetition
        first_rep = subject_df[subject_df['rep_number'] == subject_df['rep_number'].iloc[0]]
        skeleton_data = torch.tensor(first_rep[skeleton_cols].values, dtype=torch.float32)
        num_frames = len(skeleton_data)
        skeleton_reshaped = skeleton_data.reshape(num_frames, 25, 3)
        
        # 3D plot
        ax_3d = axes[0, i] if len(subjects) > 1 else axes[0]
        ax_3d = fig.add_subplot(2, len(subjects), i + 1, projection='3d')
        
        # Sample frames
        for frame_idx in range(0, num_frames, max(1, num_frames // 8)):
            frame_data = skeleton_reshaped[frame_idx]
            X = frame_data[:, 0].numpy()
            Y = frame_data[:, 2].numpy()  # Swapped
            Z = frame_data[:, 1].numpy()  # Swapped
            
            ax_3d.scatter(X, Y, Z, c=colors[i], s=20, alpha=0.6)
        
        ax_3d.set_title(f'Subject {subject_id}')
        ax_3d.view_init(elev=20, azim=90)
        
        # Trajectory plot
        ax_traj = axes[1, i] if len(subjects) > 1 else axes[1]
        
        # Plot trajectory for each repetition
        for rep_num in sorted(subject_df['rep_number'].unique()):
            rep_data = subject_df[subject_df['rep_number'] == rep_num]
            skeleton_data = torch.tensor(rep_data[skeleton_cols].values, dtype=torch.float32)
            skeleton_reshaped = skeleton_data.reshape(len(skeleton_data), 25, 3)
            
            com_trajectory = torch.mean(skeleton_reshaped, dim=1)
            X_traj = com_trajectory[:, 0].numpy()
            Y_traj = com_trajectory[:, 2].numpy()  # Swapped
            
            ax_traj.plot(X_traj, Y_traj, c=colors[i], alpha=0.7, linewidth=2)
        
        ax_traj.set_title(f'Subject {subject_id} - Trajectories')
        ax_traj.grid(True, alpha=0.3)
    
    plt.suptitle(f'{gesture_name} - Subject Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'outputs/gesture_comparison_{gesture_label}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Gesture Repetition Analysis")
    print("=" * 40)
    
    # Load data
    print("Loading IRDS data...")
    df = load_irds_data(folder_path="../data", max_files=20)
    
    # Example: Analyze repetitions for a specific subject and gesture
    subject_id = "201"  # Change this to your desired subject
    gesture_label = "1"  # Change this to your desired gesture
    
    print(f"\nAnalyzing repetitions for Subject {subject_id}, Gesture {gesture_label}")
    
    # Visualize repetitions
    visualize_gesture_repetitions(df, subject_id, gesture_label, max_reps=5)
    
    # Analyze variability
    stats_df = analyze_repetition_variability(df, subject_id, gesture_label)
    
    # Compare across subjects
    print(f"\nComparing gesture {gesture_label} across subjects...")
    compare_subjects_gesture(df, gesture_label, max_subjects=3)
    
    print("\nAnalysis complete!")

