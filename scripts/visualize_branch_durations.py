"""
Visualize Branch Durations - Boxplot of issue/branch durations across all projects

Shows the distribution of branch lifetimes (first commit to last commit) for each project.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from core import load_project_config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def parse_datetime(dt_string):
    """Parse GitLab datetime string"""
    if not dt_string:
        return None
    try:
        # Remove timezone suffix for parsing
        if dt_string.endswith('Z'):
            dt_string = dt_string[:-1]
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        return None


def calculate_branch_durations(project_name):
    """
    Calculate duration for each branch based on first and last commit

    Args:
        project_name: Name of the project

    Returns:
        List of durations in hours
    """
    data_dir = os.path.join('..', 'data_raw', project_name)
    branches_file = os.path.join(data_dir, 'branches.json')
    commits_file = os.path.join(data_dir, 'all_commits.json')

    if not os.path.exists(branches_file) or not os.path.exists(commits_file):
        print(f"  [SKIP] {project_name}: Missing data files")
        print(f"    Looking for: {branches_file}")
        print(f"    Looking for: {commits_file}")
        return []

    # Load data
    with open(branches_file, 'r', encoding='utf-8') as f:
        branches = json.load(f)

    with open(commits_file, 'r', encoding='utf-8') as f:
        commits = json.load(f)

    # Group commits by branch
    commits_by_branch = {}
    for commit in commits:
        branch_name = commit.get('branch_name')
        if branch_name:
            if branch_name not in commits_by_branch:
                commits_by_branch[branch_name] = []
            commits_by_branch[branch_name].append(commit)

    # Calculate durations
    durations = []

    for branch in branches:
        branch_name = branch.get('name')

        # Skip main/master branches
        if branch_name in ['main', 'master']:
            continue

        # Get commits for this branch
        branch_commits = commits_by_branch.get(branch_name, [])

        if len(branch_commits) < 2:
            # Need at least 2 commits to calculate duration
            continue

        # Parse commit timestamps
        commit_times = []
        for commit in branch_commits:
            created_at = commit.get('created_at') or commit.get('committed_date')
            dt = parse_datetime(created_at)
            if dt:
                commit_times.append(dt)

        if len(commit_times) < 2:
            continue

        # Calculate duration (first to last commit)
        first_commit = min(commit_times)
        last_commit = max(commit_times)
        duration_hours = (last_commit - first_commit).total_seconds() / 3600

        # Only include positive durations
        if duration_hours > 0:
            durations.append(duration_hours)

    return durations


def main():
    """Create boxplot visualization of branch durations"""

    print("="*80)
    print("BRANCH DURATION VISUALIZATION")
    print("="*80)
    print()

    # Load projects
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Processing {len(projects)} projects...\n")

    # Collect durations for all projects
    all_data = []

    for project_id, project_name in projects:
        print(f"Processing {project_name}...")
        durations = calculate_branch_durations(project_name)

        if durations:
            for duration in durations:
                all_data.append({
                    'project': project_name.replace('ba_project_', '').replace('_battleship', '').upper(),
                    'duration_hours': duration
                })
            print(f"  Found {len(durations)} branches with duration data")
        else:
            print(f"  No duration data found")

    if not all_data:
        print("\n[ERROR] No duration data found across all projects")
        return 1

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    print(f"\n{'='*80}")
    print("CREATING VISUALIZATION")
    print(f"{'='*80}\n")
    print(f"Total branches analyzed: {len(df)}")
    print(f"Projects with data: {df['project'].nunique()}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create boxplot
    box_plot = sns.boxplot(
        data=df,
        x='project',
        y='duration_hours',
        palette='Set2',
        ax=ax
    )

    # Customize plot
    ax.set_xlabel('Project', fontsize=14, fontweight='bold')
    ax.set_ylabel('Branch Duration (Hours)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Branch/Issue Duration Distribution Across All Projects\n'
        '(Time from First Commit to Last Commit per Branch)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add statistics text
    stats_text = []
    for project in sorted(df['project'].unique()):
        project_data = df[df['project'] == project]['duration_hours']
        median = project_data.median()
        stats_text.append(f"{project}: {median:.1f}h")

    # Add text box with median values
    textstr = "Median durations:\n" + "\n".join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        1.02, 0.5, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=props
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_file = os.path.join('..', 'visualizations', 'branch_durations_boxplot.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    for project in sorted(df['project'].unique()):
        project_data = df[df['project'] == project]['duration_hours']
        print(f"{project:5} | Count: {len(project_data):4} | "
              f"Median: {project_data.median():6.1f}h | "
              f"Mean: {project_data.mean():6.1f}h | "
              f"Min: {project_data.min():6.1f}h | "
              f"Max: {project_data.max():6.1f}h")

    print(f"\n{'='*80}")
    print(f"Overall: {len(df)} branches analyzed across {df['project'].nunique()} projects")
    print(f"{'='*80}")

    # Show plot
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
