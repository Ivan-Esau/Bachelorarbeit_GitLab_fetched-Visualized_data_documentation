"""
Overall Project Duration Visualization

Calculates and visualizes the total project duration (from first commit to last commit
across all branches in each project).

Data Source:
- Branch lifecycle durations CSV

Output:
    - project_overall_duration.png: Bar chart showing total project duration

Usage:
    python visualize_branch_duration_boxplot.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_datetime(dt_string):
    """Parse datetime string"""
    if pd.isna(dt_string):
        return None
    try:
        return pd.to_datetime(dt_string)
    except:
        return None


def main():
    """Create visualization of overall project durations"""

    print("=" * 100)
    print("OVERALL PROJECT DURATION VISUALIZATION")
    print("=" * 100)
    print()

    # Load data
    base_dir = Path(__file__).parent.parent.parent.parent
    data_file = base_dir / 'visualizations/summary/branch_lifecycle/branch_lifecycle_durations.csv'

    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    # Parse datetime columns
    df['first_activity'] = df['first_activity'].apply(parse_datetime)
    df['last_activity'] = df['last_activity'].apply(parse_datetime)

    print(f"Total branches: {len(df)}")
    print()

    # Calculate overall project duration
    projects = sorted(df['project'].unique())
    project_durations = []

    print("Calculating overall project durations...")
    print()

    for project in projects:
        project_df = df[df['project'] == project]

        # Get earliest first activity and latest last activity across all branches
        first_commit = project_df['first_activity'].min()
        last_commit = project_df['last_activity'].max()

        # Calculate duration in minutes
        duration = (last_commit - first_commit).total_seconds() / 60

        project_durations.append({
            'project': project,
            'first_commit': first_commit,
            'last_commit': last_commit,
            'duration_minutes': duration,
            'duration_hours': duration / 60,
            'duration_days': duration / (60 * 24),
            'total_branches': len(project_df)
        })

        print(f"{project}:")
        print(f"  First commit: {first_commit}")
        print(f"  Last commit:  {last_commit}")
        print(f"  Duration: {duration:.0f} minutes ({duration/60:.1f} hours)")
        print(f"  Branches: {len(project_df)}")
        print()

    project_df = pd.DataFrame(project_durations)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by project name (A01-A10) instead of duration
    project_df_sorted = project_df.sort_values('project')

    # Create bars
    bars = ax.barh(project_df_sorted['project'],
                   project_df_sorted['duration_minutes'],
                   color='#5A7A9B',
                   alpha=0.8,
                   edgecolor='#34495E',
                   linewidth=1.5)

    # Add value labels
    for i, (idx, row) in enumerate(project_df_sorted.iterrows()):
        duration_min = row['duration_minutes']
        duration_hrs = row['duration_hours']
        ax.text(duration_min + 10, i,
               f'{duration_min:.0f}m ({duration_hrs:.1f}h)',
               va='center', ha='left', fontsize=10, fontweight='bold')

    # Style the plot
    ax.set_xlabel('Overall Project Duration (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Project', fontsize=12, fontweight='bold')
    ax.set_title('Overall Project Duration\n' +
                 'Time from first commit to last commit/merge across all branches',
                 fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, axis='x')

    # Calculate statistics
    avg_duration = project_df['duration_minutes'].mean()
    median_duration = project_df['duration_minutes'].median()
    min_duration = project_df['duration_minutes'].min()
    max_duration = project_df['duration_minutes'].max()

    # Add vertical lines for average and median
    avg_line = ax.axvline(x=avg_duration, color='red', linestyle='--',
                          linewidth=2, alpha=0.7, label=f'Average: {avg_duration:.0f}m')
    median_line = ax.axvline(x=median_duration, color='green', linestyle='--',
                             linewidth=2, alpha=0.7, label=f'Median: {median_duration:.0f}m')

    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save
    output_dir = base_dir / 'visualizations/summary/branch_lifecycle'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'project_overall_duration.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Overall duration chart saved: {output_file}")
    # Print summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print()

    print(f"Average project duration: {avg_duration:.0f} minutes ({avg_duration/60:.1f} hours)")
    print(f"Median project duration: {median_duration:.0f} minutes ({median_duration/60:.1f} hours)")
    print(f"Shortest project: {min_duration:.0f} minutes ({min_duration/60:.1f} hours)")
    print(f"Longest project: {max_duration:.0f} minutes ({max_duration/60:.1f} hours)")

    print()
    print("=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
