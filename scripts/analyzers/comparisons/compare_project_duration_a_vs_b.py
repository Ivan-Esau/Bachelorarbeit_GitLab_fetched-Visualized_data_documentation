"""
Compare Project Duration: Type A vs Type B

Creates boxplot comparison of overall project durations between Type A and Type B projects.
Overall project duration = time from first commit to last commit/merge across all feature branches.

Data Source:
    - visualizations/a/summary/branch_lifecycle/branch_lifecycle_durations.csv
    - visualizations/b/summary/branch_lifecycle/branch_lifecycle_durations.csv

Output:
    - visualizations/comparisons/project_duration_a_vs_b.png: Boxplot comparison
    - visualizations/comparisons/project_duration_a_vs_b_statistics.csv: Statistical comparison

Usage:
    python compare_project_duration_a_vs_b.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")


def parse_datetime(dt_string):
    """Parse datetime string"""
    if pd.isna(dt_string):
        return None
    try:
        return pd.to_datetime(dt_string)
    except:
        return None


def calculate_project_durations(df, project_type):
    """
    Calculate overall project duration for each project

    Args:
        df: DataFrame with branch lifecycle data
        project_type: 'A' or 'B'

    Returns:
        Dict mapping project labels to duration in minutes
    """
    # Parse datetime columns
    df['first_activity'] = df['first_activity'].apply(parse_datetime)
    df['last_activity'] = df['last_activity'].apply(parse_datetime)

    projects = sorted(df['project'].unique())
    project_durations = {}

    for project in projects:
        project_df = df[df['project'] == project]

        # Get earliest first activity and latest last activity
        first_commit = project_df['first_activity'].min()
        last_commit = project_df['last_activity'].max()

        # Calculate duration in minutes
        duration_minutes = (last_commit - first_commit).total_seconds() / 60

        project_durations[project] = duration_minutes

    return project_durations


def perform_statistical_tests(type_a_durations, type_b_durations):
    """
    Perform statistical tests comparing Type A and Type B durations

    Args:
        type_a_durations: List of Type A project durations
        type_b_durations: List of Type B project durations

    Returns:
        Dict with statistical test results
    """
    # Mann-Whitney U test (non-parametric alternative to t-test)
    u_statistic, p_value_mw = stats.mannwhitneyu(type_a_durations, type_b_durations, alternative='two-sided')

    # Perform independent t-test (parametric)
    t_statistic, p_value_t = stats.ttest_ind(type_a_durations, type_b_durations)

    # Calculate effect size (Cohen's d)
    mean_a = np.mean(type_a_durations)
    mean_b = np.mean(type_b_durations)
    std_a = np.std(type_a_durations, ddof=1)
    std_b = np.std(type_b_durations, ddof=1)
    pooled_std = np.sqrt(((len(type_a_durations) - 1) * std_a**2 + (len(type_b_durations) - 1) * std_b**2) /
                         (len(type_a_durations) + len(type_b_durations) - 2))
    cohens_d = (mean_a - mean_b) / pooled_std

    return {
        'mann_whitney_u': u_statistic,
        'mann_whitney_p': p_value_mw,
        't_statistic': t_statistic,
        't_test_p': p_value_t,
        'cohens_d': cohens_d
    }


def create_comparison_boxplot(type_a_durations, type_b_durations, output_file):
    """
    Create boxplot comparing Type A and Type B project durations

    Args:
        type_a_durations: Dict of Type A project durations
        type_b_durations: Dict of Type B project durations
        output_file: Path to save the PNG file
    """
    # Prepare data
    data_a = list(type_a_durations.values())
    data_b = list(type_b_durations.values())

    # Calculate statistics
    stats_a = {
        'mean': np.mean(data_a),
        'median': np.median(data_a),
        'std': np.std(data_a, ddof=1),
        'min': np.min(data_a),
        'max': np.max(data_a),
        'q1': np.percentile(data_a, 25),
        'q3': np.percentile(data_a, 75),
        'n': len(data_a)
    }

    stats_b = {
        'mean': np.mean(data_b),
        'median': np.median(data_b),
        'std': np.std(data_b, ddof=1),
        'min': np.min(data_b),
        'max': np.max(data_b),
        'q1': np.percentile(data_b, 25),
        'q3': np.percentile(data_b, 75),
        'n': len(data_b)
    }

    # Perform statistical tests
    test_results = perform_statistical_tests(data_a, data_b)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create boxplot
    positions = [1, 2]
    box_data = [data_a, data_b]
    labels = ['Type A Projects', 'Type B Projects']

    bp = ax.boxplot(box_data,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
                    boxprops=dict(facecolor='#5A7A9B', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='#34495E', linewidth=2.5),
                    whiskerprops=dict(color='#34495E', linewidth=1.5),
                    capprops=dict(color='#34495E', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='none',
                                  markersize=5, markeredgecolor='black', markeredgewidth=1))

    # Customize plot
    ax.set_ylabel('Overall Project Duration (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Project Duration Comparison: Type A vs Type B\n' +
                 'Duration from first commit to last commit across all feature branches',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add combined statistics text box in top right corner
    stats_text = (
        f'Type A (n={stats_a["n"]}):\n'
        f'  Mean: {stats_a["mean"]:.0f}m | Median: {stats_a["median"]:.0f}m\n'
        f'  Range: {stats_a["min"]:.0f}-{stats_a["max"]:.0f}m\n'
        f'\n'
        f'Type B (n={stats_b["n"]}):\n'
        f'  Mean: {stats_b["mean"]:.0f}m | Median: {stats_b["median"]:.0f}m\n'
        f'  Range: {stats_b["min"]:.0f}-{stats_b["max"]:.0f}m'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#34495E', linewidth=1.5)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=props)

    # Add legend in top left corner
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
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.9, edgecolor='#34495E', title='Legend', title_fontsize=10)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Boxplot saved: {output_file}")

    return stats_a, stats_b, test_results


def create_statistics_table(type_a_durations, type_b_durations, stats_a, stats_b, test_results, output_file):
    """
    Create detailed statistics CSV table

    Args:
        type_a_durations: Dict of Type A project durations
        type_b_durations: Dict of Type B project durations
        stats_a: Statistics dict for Type A
        stats_b: Statistics dict for Type B
        test_results: Statistical test results dict
        output_file: Path to save the CSV file
    """
    # Create detailed project table
    rows = []

    # Add Type A projects
    for project, duration in sorted(type_a_durations.items()):
        rows.append({
            'Project': project,
            'Type': 'A',
            'Duration (minutes)': duration,
            'Duration (hours)': duration / 60,
            'Duration (days)': duration / (60 * 24)
        })

    # Add Type B projects
    for project, duration in sorted(type_b_durations.items()):
        rows.append({
            'Project': project,
            'Type': 'B',
            'Duration (minutes)': duration,
            'Duration (hours)': duration / 60,
            'Duration (days)': duration / (60 * 24)
        })

    df = pd.DataFrame(rows)

    # Add summary statistics rows
    summary_rows = [
        {'Project': '', 'Type': '', 'Duration (minutes)': '', 'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 'SUMMARY STATISTICS', 'Type': '', 'Duration (minutes)': '', 'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 'Type A Mean', 'Type': 'A', 'Duration (minutes)': stats_a['mean'],
         'Duration (hours)': stats_a['mean']/60, 'Duration (days)': stats_a['mean']/(60*24)},
        {'Project': 'Type A Median', 'Type': 'A', 'Duration (minutes)': stats_a['median'],
         'Duration (hours)': stats_a['median']/60, 'Duration (days)': stats_a['median']/(60*24)},
        {'Project': 'Type A Std', 'Type': 'A', 'Duration (minutes)': stats_a['std'],
         'Duration (hours)': stats_a['std']/60, 'Duration (days)': stats_a['std']/(60*24)},
        {'Project': 'Type B Mean', 'Type': 'B', 'Duration (minutes)': stats_b['mean'],
         'Duration (hours)': stats_b['mean']/60, 'Duration (days)': stats_b['mean']/(60*24)},
        {'Project': 'Type B Median', 'Type': 'B', 'Duration (minutes)': stats_b['median'],
         'Duration (hours)': stats_b['median']/60, 'Duration (days)': stats_b['median']/(60*24)},
        {'Project': 'Type B Std', 'Type': 'B', 'Duration (minutes)': stats_b['std'],
         'Duration (hours)': stats_b['std']/60, 'Duration (days)': stats_b['std']/(60*24)},
        {'Project': '', 'Type': '', 'Duration (minutes)': '', 'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 'STATISTICAL TESTS', 'Type': '', 'Duration (minutes)': '', 'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 'Mann-Whitney U statistic', 'Type': '', 'Duration (minutes)': test_results['mann_whitney_u'],
         'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 'Mann-Whitney p-value', 'Type': '', 'Duration (minutes)': test_results['mann_whitney_p'],
         'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 't-test statistic', 'Type': '', 'Duration (minutes)': test_results['t_statistic'],
         'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': 't-test p-value', 'Type': '', 'Duration (minutes)': test_results['t_test_p'],
         'Duration (hours)': '', 'Duration (days)': ''},
        {'Project': "Cohen's d (effect size)", 'Type': '', 'Duration (minutes)': test_results['cohens_d'],
         'Duration (hours)': '', 'Duration (days)': ''},
    ]

    df_summary = pd.DataFrame(summary_rows)
    df = pd.concat([df, df_summary], ignore_index=True)

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"[OK] Statistics table saved: {output_file}")


def main():
    """Compare overall project durations between Type A and Type B"""

    print("=" * 100)
    print("PROJECT DURATION COMPARISON: TYPE A vs TYPE B")
    print("=" * 100)
    print()

    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_a_file = base_dir / 'visualizations/a/summary/branch_lifecycle/branch_lifecycle_durations.csv'
    data_b_file = base_dir / 'visualizations/b/summary/branch_lifecycle/branch_lifecycle_durations.csv'

    # Check if files exist
    if not data_a_file.exists():
        print(f"[ERROR] Type A data not found: {data_a_file}")
        return 1
    if not data_b_file.exists():
        print(f"[ERROR] Type B data not found: {data_b_file}")
        return 1

    print("Loading data...")
    print(f"  Type A: {data_a_file}")
    print(f"  Type B: {data_b_file}")
    print()

    # Load data
    df_a = pd.read_csv(data_a_file)
    df_b = pd.read_csv(data_b_file)

    print(f"Type A: {len(df_a)} branches across {df_a['project'].nunique()} projects")
    print(f"Type B: {len(df_b)} branches across {df_b['project'].nunique()} projects")
    print()

    # Calculate project durations
    print("Calculating overall project durations...")
    type_a_durations = calculate_project_durations(df_a, 'A')
    type_b_durations = calculate_project_durations(df_b, 'B')

    print(f"  Type A: {len(type_a_durations)} projects")
    print(f"  Type B: {len(type_b_durations)} projects")
    print()

    # Create output directory
    output_dir = base_dir / 'visualizations/comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("Creating comparison boxplot...")
    boxplot_file = output_dir / 'project_duration_a_vs_b.png'
    stats_a, stats_b, test_results = create_comparison_boxplot(type_a_durations, type_b_durations, boxplot_file)
    print()

    # Create statistics table
    print("Creating statistics table...")
    stats_file = output_dir / 'project_duration_a_vs_b_statistics.csv'
    create_statistics_table(type_a_durations, type_b_durations, stats_a, stats_b, test_results, stats_file)
    print()

    # Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Type A Projects (n={stats_a['n']}):")
    print(f"  Mean:   {stats_a['mean']:.0f} minutes ({stats_a['mean']/60:.1f} hours)")
    print(f"  Median: {stats_a['median']:.0f} minutes ({stats_a['median']/60:.1f} hours)")
    print(f"  Std:    {stats_a['std']:.0f} minutes")
    print(f"  Range:  {stats_a['min']:.0f} - {stats_a['max']:.0f} minutes")
    print()
    print(f"Type B Projects (n={stats_b['n']}):")
    print(f"  Mean:   {stats_b['mean']:.0f} minutes ({stats_b['mean']/60:.1f} hours)")
    print(f"  Median: {stats_b['median']:.0f} minutes ({stats_b['median']/60:.1f} hours)")
    print(f"  Std:    {stats_b['std']:.0f} minutes")
    print(f"  Range:  {stats_b['min']:.0f} - {stats_b['max']:.0f} minutes")
    print()
    print("Statistical Tests:")
    print(f"  Mann-Whitney U test: p={test_results['mann_whitney_p']:.4f}")
    print(f"  t-test: p={test_results['t_test_p']:.4f}")
    print(f"  Cohen's d: {test_results['cohens_d']:.3f}")

    # Interpret results
    if test_results['mann_whitney_p'] < 0.05:
        print(f"\n  Conclusion: Significant difference between Type A and Type B (p < 0.05)")
        if stats_a['median'] > stats_b['median']:
            print(f"  Type A projects took longer on average")
        else:
            print(f"  Type B projects took longer on average")
    else:
        print(f"\n  Conclusion: No significant difference between Type A and Type B (p >= 0.05)")

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
