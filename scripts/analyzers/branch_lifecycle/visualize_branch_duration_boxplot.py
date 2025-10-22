"""
Branch Duration Visualizations

Creates two visualizations:
1. Boxplot of individual feature branch durations per project
2. Bar chart of overall project durations (first to last commit)

Note: Master/main branches are EXCLUDED - they represent baseline templates, not feature work.

Data Source:
- Branch lifecycle durations CSV (feature branches only)

Output:
    - branch_duration_boxplot.png: Boxplot showing distribution of feature branch durations per project
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
    """Create branch duration visualizations (boxplot + bar chart)"""

    print("=" * 100)
    print("BRANCH DURATION VISUALIZATIONS")
    print("=" * 100)
    print()

    # Load data
    base_dir = Path(__file__).parent.parent.parent.parent

    # Load active project type from config
    from core.config_loader import load_project_types
    types_config = load_project_types()
    active_types = types_config.get('active_project_types', ['a'])
    project_type = active_types[0] if active_types else 'a'

    # Use the active project type to find the correct data file
    data_file = base_dir / f'visualizations/{project_type}/summary/branch_lifecycle/branch_lifecycle_durations.csv'

    if not data_file.exists():
        print(f"[ERROR] Could not find branch_lifecycle_durations.csv at: {data_file}")
        return 1

    print(f"Loading data from: {data_file}")
    print(f"Project type: {project_type.upper()}")
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

    # Determine project type and output directory first
    first_project = projects[0] if projects else None
    if first_project and len(first_project) >= 1:
        project_type = first_project[0].lower()  # Get first character (a, b, c, etc.)
    else:
        project_type = 'unknown'

    output_dir = base_dir / 'visualizations' / project_type / 'summary' / 'branch_lifecycle'
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, create boxplot of individual branch durations per project
    print("=" * 100)
    print("CREATING BRANCH DURATION BOXPLOT")
    print("=" * 100)
    print()

    fig_box, ax_box = plt.subplots(figsize=(14, 8))

    # Prepare data for boxplot (individual branch durations per project)
    # Convert from days to minutes
    branch_data_by_project = []
    project_labels = []
    all_branch_durations = []

    for project in projects:
        # Get duration in days and convert to minutes
        project_branches_days = df[df['project'] == project]['development_duration_days'].values
        project_branches_minutes = project_branches_days * 24 * 60  # Convert days to minutes
        branch_data_by_project.append(project_branches_minutes)
        all_branch_durations.extend(project_branches_minutes)
        project_labels.append(project)

    # Calculate overall statistics for the text box
    all_durations = np.array(all_branch_durations)
    stats_median = np.median(all_durations)
    stats_mean = np.mean(all_durations)
    stats_min = np.min(all_durations)
    stats_max = np.max(all_durations)
    total_branches = len(all_durations)

    # Create boxplot
    bp = ax_box.boxplot(branch_data_by_project,
                        tick_labels=project_labels,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
                        boxprops=dict(facecolor='#5A7A9B', alpha=0.7, linewidth=1.5),
                        medianprops=dict(color='#34495E', linewidth=2.5),
                        whiskerprops=dict(color='#34495E', linewidth=1.5),
                        capprops=dict(color='#34495E', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='none',
                                      markersize=5, markeredgecolor='black', markeredgewidth=1))

    # Style the boxplot
    ax_box.set_xlabel('Project', fontsize=12, fontweight='bold')
    ax_box.set_ylabel('Branch Duration (minutes)', fontsize=12, fontweight='bold')
    ax_box.set_title('Feature Branch Development Duration Distribution Across Projects\n' +
                     'Boxplot showing median, quartiles, and outliers (Master/Main branches excluded)',
                     fontsize=13, fontweight='bold', pad=15)
    ax_box.grid(True, alpha=0.3, axis='y')
    ax_box.set_axisbelow(True)

    # Add statistics text box in top right corner (inline format like Type A)
    stats_text = (
        f'Overall Statistics:\n'
        f'Median: {stats_median:.0f}m | Mean: {stats_mean:.0f}m\n'
        f'Min: {stats_min:.0f}m | Max: {stats_max:.0f}m\n'
        f'Total branches: {total_branches}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#34495E', linewidth=1.5)
    ax_box.text(0.98, 0.98, stats_text, transform=ax_box.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=props)

    # Add legend box in top left corner (match Type A style)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=6, markeredgecolor='red', linestyle='None', label='Mean'),
        Patch(facecolor='#5A7A9B', alpha=0.7, edgecolor='#34495E', linewidth=1.5, label='Q1/Q3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=5, markeredgecolor='black', markeredgewidth=1, linestyle='None', label='Outliers')
    ]
    ax_box.legend(handles=legend_elements, loc='upper left', fontsize=9,
                 framealpha=0.9, edgecolor='#34495E', title='Legend', title_fontsize=10)

    plt.tight_layout()

    # Save boxplot
    boxplot_file = output_dir / 'branch_duration_boxplot.png'
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Branch duration boxplot saved: {boxplot_file}")
    print()

    # Now create horizontal bar chart of overall project durations
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

    # Save bar chart (output_dir already defined above)
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
