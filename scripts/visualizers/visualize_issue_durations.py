"""
Visualize Issue/Branch Development Durations

Creates a boxplot visualization comparing development durations across all projects.
Duration is calculated from first commit to last commit (or merge) for each issue/branch.

This script follows scientific research standards:
- Boxplots show median, quartiles, and outliers
- All projects displayed side-by-side for comparison
- Neutral color scheme suitable for publication

Input Data Requirements:
    - commits_by_mr.json: Commits grouped by merge request
    - merge_requests.json: Merge request metadata including branch names

Output:
    - issue_durations_boxplot.png: Comparative boxplot for all projects
    - issue_durations_statistics.csv: Summary statistics table

Usage:
    python visualize_issue_durations.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scripts.core import load_project_config

# Set style
sns.set_style("whitegrid")


def parse_datetime(dt_string):
    """Parse GitLab datetime string to datetime object"""
    if not dt_string:
        return None
    try:
        if dt_string.endswith('Z'):
            dt_string = dt_string[:-1]
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        return None


def get_issue_durations(project_name, data_base_dir='../../data_raw'):
    """
    Calculate duration for each issue/branch based on commit timestamps

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data

    Returns:
        List of dicts with: label, duration_hours, duration_minutes, num_commits
    """
    data_dir = os.path.join(data_base_dir, project_name)
    commits_by_mr_file = os.path.join(data_dir, 'commits_by_mr.json')
    merge_requests_file = os.path.join(data_dir, 'merge_requests.json')

    if not os.path.exists(commits_by_mr_file):
        return []

    # Load commits by MR
    with open(commits_by_mr_file, 'r', encoding='utf-8') as f:
        commits_by_mr = json.load(f)

    # Load merge requests for branch names
    mr_info = {}
    if os.path.exists(merge_requests_file):
        with open(merge_requests_file, 'r', encoding='utf-8') as f:
            merge_requests = json.load(f)
            for mr in merge_requests:
                mr_iid = str(mr.get('iid'))  # Use internal ID (iid) not global ID
                mr_info[mr_iid] = {
                    'title': mr.get('title', 'Unknown'),
                    'source_branch': mr.get('source_branch', 'Unknown')
                }

    issue_data = []

    for mr_id, commits in commits_by_mr.items():
        if len(commits) < 2:
            continue

        # Parse commit times
        commit_times = []
        for commit in commits:
            created_at = (commit.get('committed_date') or
                         commit.get('created_at') or
                         commit.get('authored_date'))
            dt = parse_datetime(created_at)
            if dt:
                commit_times.append(dt)

        if len(commit_times) < 2:
            continue

        first_commit = min(commit_times)
        last_commit = max(commit_times)
        duration_hours = (last_commit - first_commit).total_seconds() / 3600

        if duration_hours > 0:
            # Get branch info
            info = mr_info.get(mr_id, {})
            branch_name = info.get('source_branch', f'branch-{mr_id}')

            # Extract issue number from branch name for cleaner labels
            issue_label = branch_name
            if 'issue-' in branch_name.lower():
                # Extract issue number
                parts = branch_name.lower().split('issue-')
                if len(parts) > 1:
                    issue_num = parts[1].split('-')[0]
                    issue_label = f'Issue #{issue_num}'
            elif '/' in branch_name:
                # If it's a feature/bugfix branch, show just the last part
                issue_label = branch_name.split('/')[-1][:40]
            else:
                # Use branch name, truncate if too long
                issue_label = branch_name[:40]

            issue_data.append({
                'label': issue_label,
                'duration_hours': duration_hours,
                'duration_minutes': duration_hours * 60,
                'num_commits': len(commits)
            })

    return issue_data


def create_comparative_boxplot(all_project_data, output_file):
    """
    Create boxplot comparing all projects

    Args:
        all_project_data: Dict mapping project labels to duration data
        output_file: Path to save the PNG file
    """
    # Prepare data for plotting
    plot_data = []
    for project_label, durations in all_project_data.items():
        for duration in durations:
            plot_data.append({
                'Project': project_label,
                'Duration (minutes)': duration
            })

    df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get project order
    project_order = sorted(all_project_data.keys())

    # Create boxplot
    positions = range(1, len(project_order) + 1)
    box_data = [all_project_data[proj] for proj in project_order]

    bp = ax.boxplot(box_data,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='#5A7A9B', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='#34495E', linewidth=2.5),
                    whiskerprops=dict(color='#34495E', linewidth=1.5),
                    capprops=dict(color='#34495E', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                  markersize=6, alpha=0.7, markeredgecolor='#C0392B'))

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(project_order, fontsize=11)
    ax.set_xlabel('Projekt', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dauer (Minuten)', fontsize=12, fontweight='bold')
    ax.set_title('Dauer der Issue-/Branch-Entwicklung über alle Projekte\n(Erster bis letzter Commit)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add sample size annotations
    for i, proj in enumerate(project_order, 1):
        n = len(all_project_data[proj])
        ax.text(i, ax.get_ylim()[0], f'n={n}',
               ha='center', va='top', fontsize=9, color='#34495E')

    # Add legend explaining boxplot components (in German)
    legend_elements = [
        Patch(facecolor='#5A7A9B', alpha=0.7, edgecolor='#34495E', linewidth=1.5,
              label='Box (IQR: Q1-Q3)'),
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Median'),
        Line2D([0], [0], color='#34495E', linewidth=1.5, label='Whisker (1,5×IQR)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=6, alpha=0.7, markeredgecolor='#C0392B',
               label='Ausreißer', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             framealpha=0.95, edgecolor='#5A7A9B')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Boxplot saved: {output_file}")


def create_statistics_table(all_project_data, output_file):
    """
    Create summary statistics table for all projects

    Args:
        all_project_data: Dict mapping project labels to duration data
        output_file: Path to save the CSV file
    """
    stats_list = []

    for project_label in sorted(all_project_data.keys()):
        durations = all_project_data[project_label]

        stats = {
            'Project': project_label,
            'n': len(durations),
            'Median (min)': np.median(durations),
            'Mean (min)': np.mean(durations),
            'SD (min)': np.std(durations, ddof=1) if len(durations) > 1 else 0,
            'Q1 (min)': np.percentile(durations, 25),
            'Q3 (min)': np.percentile(durations, 75),
            'IQR (min)': np.percentile(durations, 75) - np.percentile(durations, 25),
            'Min (min)': min(durations),
            'Max (min)': max(durations)
        }
        stats_list.append(stats)

    # Create DataFrame and save
    df = pd.DataFrame(stats_list)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"  [OK] Statistics table saved: {output_file}")

    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    """Generate issue duration boxplot visualization for all projects"""

    print("="*80)
    print("ISSUE/BRANCH DURATION VISUALIZATION - BOXPLOT")
    print("="*80)
    print()

    # Load projects from config
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Processing {len(projects)} projects...\n")

    # Collect data for all projects
    all_project_data = {}
    total_issues = 0

    for project_id, project_name in projects:
        # Extract clean project label (e.g., 'ba_project_a01_battleship' -> 'A01')
        project_label = project_name.replace('ba_project_', '').replace('_battleship', '').upper()

        print(f"Loading {project_label}...", end=' ')

        # Get issue duration data
        issue_data = get_issue_durations(project_name)

        if issue_data:
            durations = [d['duration_minutes'] for d in issue_data]
            all_project_data[project_label] = durations
            total_issues += len(durations)
            print(f"[OK] ({len(durations)} issues, median={np.median(durations):.1f} min)")
        else:
            print(f"[SKIP] (no data)")

    if not all_project_data:
        print("\n[ERROR] No data found for any project")
        return 1

    print(f"\nTotal: {total_issues} issues across {len(all_project_data)} projects\n")

    # Create output directory
    output_dir = os.path.join('..', '..', 'visualizations', 'issue_durations')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations...\n")

    # Generate boxplot
    boxplot_file = os.path.join(output_dir, 'issue_durations_boxplot.png')
    create_comparative_boxplot(all_project_data, boxplot_file)

    # Generate statistics table
    stats_file = os.path.join(output_dir, 'issue_durations_statistics.csv')
    create_statistics_table(all_project_data, stats_file)

    print()
    print("="*80)
    print("COMPLETED")
    print(f"Location: {output_dir}")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
