"""
Branch Lifecycle Duration Analysis

Analyzes the full lifecycle of each feature branch from first commit to last commit/merge.
Shows how long each branch was in development.

Note: Master/main branches are EXCLUDED - they represent baseline templates, not feature work.

Data Sources:
- Pipelines: To determine first and last activity on each branch
- Merge Requests: To get merge timestamps
- Branches: To identify all feature branches

Metrics Calculated:
- Branch lifetime (first pipeline to last pipeline/merge)
- Development duration per feature branch
- Time to merge
- Average branch duration per project (feature branches only)

Output:
    - branch_lifecycle_durations.csv: Per-branch duration data
    - branch_lifecycle_timeline.png: Visual timeline
    - branch_lifecycle_statistics.png: Summary statistics

Usage:
    python analyze_branch_lifecycle.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime, timedelta
from core.path_helpers import get_data_dir
from core.config_loader import get_project_letter, get_project_number


def parse_datetime(dt_string):
    """Parse GitLab datetime string"""
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        return None


def analyze_branch_lifecycle(project_name):
    """
    Analyze branch lifecycle for a single project

    Returns:
        List of dictionaries with branch lifecycle data
    """
    # Use path helper to get correct letter-based path
    project_dir = get_data_dir(project_name)

    # Load data
    with open(project_dir / 'branches.json', encoding='utf-8') as f:
        branches = json.load(f)
    with open(project_dir / 'pipelines.json', encoding='utf-8') as f:
        pipelines = json.load(f)
    with open(project_dir / 'merge_requests.json', encoding='utf-8') as f:
        merge_requests = json.load(f)

    # Create MR lookup by source branch
    mr_by_branch = {}
    for mr in merge_requests:
        source_branch = mr.get('source_branch')
        if source_branch:
            mr_by_branch[source_branch] = mr

    results = []

    for branch in branches:
        branch_name = branch['name']

        # Skip master/main branches - they represent baseline template, not feature work
        if branch_name.lower() in ['master', 'main']:
            continue

        # Get all pipelines for this branch
        branch_pipelines = [p for p in pipelines if p.get('ref') == branch_name]

        if not branch_pipelines:
            continue

        # Sort by created_at
        branch_pipelines.sort(key=lambda p: p.get('created_at', ''))

        # Get first and last pipeline timestamps
        first_pipeline = branch_pipelines[0]
        last_pipeline = branch_pipelines[-1]

        first_time = parse_datetime(first_pipeline.get('created_at'))
        last_time = parse_datetime(last_pipeline.get('created_at'))

        # Get merge info if available
        mr_info = mr_by_branch.get(branch_name)
        merge_time = None
        mr_created = None
        mr_state = 'no_mr'

        if mr_info:
            mr_state = mr_info.get('state', 'unknown')
            mr_created = parse_datetime(mr_info.get('created_at'))
            if mr_info.get('merged_at'):
                merge_time = parse_datetime(mr_info.get('merged_at'))
                # Use merge time as end if available
                if merge_time and merge_time > last_time:
                    last_time = merge_time

                # Validate merge: get last pipeline BEFORE merge
                if mr_state == 'merged':
                    # Filter to pipelines before or at merge time
                    pipelines_before_merge = [
                        p for p in branch_pipelines
                        if parse_datetime(p.get('created_at')) <= merge_time
                    ]

                    if pipelines_before_merge:
                        validation_pipeline = pipelines_before_merge[-1]

                        # Check compile and test status in last pipeline before merge
                        compile_status = None
                        test_status = None

                        for job in validation_pipeline.get('jobs', []):
                            if job.get('stage') == 'compile':
                                compile_status = job.get('status')
                            elif job.get('stage') == 'test':
                                test_status = job.get('status')

                        # Validate: both compile and test must be success
                        is_valid = (compile_status == 'success' and test_status == 'success')
                        mr_state = 'valid_merge' if is_valid else 'invalid_merge'

        if not first_time or not last_time:
            continue

        # Calculate durations
        development_duration = (last_time - first_time).total_seconds()
        development_days = development_duration / 86400  # Convert to days

        # Extract issue number
        issue_num = None
        if 'issue-' in branch_name.lower():
            parts = branch_name.lower().split('issue-')
            if len(parts) > 1:
                issue_num = parts[1].split('-')[0]

        results.append({
            'branch_name': branch_name,
            'issue': f"Issue #{issue_num}" if issue_num else branch_name[:30],
            'first_activity': first_time,
            'last_activity': last_time,
            'development_duration_seconds': development_duration,
            'development_duration_days': development_days,
            'development_duration_hours': development_duration / 3600,
            'total_pipelines': len(branch_pipelines),
            'mr_state': mr_state,
            'mr_created_at': mr_created,
            'merged_at': merge_time
        })

    return results


def create_project_duration_chart(project_label, project_data, output_file):
    """
    Create duration bar chart for a single project
    Shows branch duration in minutes for each branch
    """
    # Sort by issue number
    def sort_key(item):
        label = item['issue']
        if label.startswith('Issue #'):
            try:
                return (0, int(label.split('#')[1]))
            except:
                return (1, label)
        else:
            return (1, label)

    project_data = sorted(project_data, key=sort_key)

    # Extract data - convert to minutes
    issues = [d['issue'] for d in project_data]
    durations_minutes = [d['development_duration_hours'] * 60 for d in project_data]
    mr_states = [d['mr_state'] for d in project_data]

    # Color mapping for MR states
    colors = {
        'valid_merge': '#27AE60',    # Green - valid merge
        'invalid_merge': '#E74C3C',  # Red - invalid merge
        'merged': '#27AE60',          # Green - fallback (shouldn't happen)
        'opened': '#3498DB',          # Blue - opened but not merged
        'closed': '#E74C3C',          # Red - closed
        'no_mr': '#95A5A6'           # Gray - no MR
    }

    bar_colors = [colors.get(state, '#95A5A6') for state in mr_states]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal bar chart
    bars = ax.barh(issues, durations_minutes, color=bar_colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Branch Duration (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Branch/Issue', fontsize=12, fontweight='bold')
    ax.set_title(f'Project {project_label}: Branch Development Durations\n'
                 f'(Time from First Pipeline to Last Activity/Merge)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, durations_minutes)):
        ax.text(val + 1, i, f'{val:.0f}m',
               va='center', ha='left', fontsize=9, fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['valid_merge'], label='Valid Merge', alpha=0.8),
        mpatches.Patch(facecolor=colors['invalid_merge'], label='Invalid Merge', alpha=0.8),
        mpatches.Patch(facecolor=colors['opened'], label='Opened (not merged)', alpha=0.8),
        mpatches.Patch(facecolor=colors['no_mr'], label='No MR', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add statistics text
    avg_duration = np.mean(durations_minutes)
    total_duration = np.sum(durations_minutes)
    valid_count = sum(1 for state in mr_states if state == 'valid_merge')
    invalid_count = sum(1 for state in mr_states if state == 'invalid_merge')
    total_merged = valid_count + invalid_count

    stats_text = (f'Avg: {avg_duration:.0f}min | '
                  f'Total: {total_duration:.0f}min | '
                  f'Merged: {total_merged} ({valid_count}V, {invalid_count}I)')

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_statistics_visualization(stats_df, output_file):
    """
    Create statistics table showing branch duration summary
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for _, row in stats_df.iterrows():
        if row['project'] == 'OVERALL':
            continue

        # Convert days to minutes for display
        avg_min = row['avg_duration_days'] * 24 * 60
        median_min = row['median_duration_days'] * 24 * 60
        min_min = row['min_duration_days'] * 24 * 60
        max_min = row['max_duration_days'] * 24 * 60

        table_data.append([
            row['project'],
            f"{row['total_branches']:.0f}",
            f"{avg_min:.0f}m",
            f"{median_min:.0f}m",
            f"{min_min:.0f}m",
            f"{max_min:.0f}m",
            f"{row['valid_merges']:.0f}",
            f"{row['invalid_merges']:.0f}",
            f"{row['avg_pipelines']:.1f}"
        ])

    # Add overall row
    overall = stats_df[stats_df['project'] == 'OVERALL'].iloc[0]

    # Convert days to minutes for display
    overall_avg_min = overall['avg_duration_days'] * 24 * 60
    overall_median_min = overall['median_duration_days'] * 24 * 60
    overall_min_min = overall['min_duration_days'] * 24 * 60
    overall_max_min = overall['max_duration_days'] * 24 * 60

    table_data.append([
        'OVERALL',
        f"{overall['total_branches']:.0f}",
        f"{overall_avg_min:.0f}m",
        f"{overall_median_min:.0f}m",
        f"{overall_min_min:.0f}m",
        f"{overall_max_min:.0f}m",
        f"{overall['valid_merges']:.0f}",
        f"{overall['invalid_merges']:.0f}",
        f"{overall['avg_pipelines']:.1f}"
    ])

    # Column headers
    columns = ['Project', 'Total\nBranches', 'Avg\nDuration', 'Median\nDuration',
               'Min\nDuration', 'Max\nDuration', 'Valid\nMerges', 'Invalid\nMerges', 'Avg\nPipelines']

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.10, 0.10, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header row
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5A7A9B')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Color data cells
    for i in range(len(table_data)):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]

            # Highlight OVERALL row
            if table_data[i][0] == 'OVERALL':
                cell.set_facecolor((0.9, 0.9, 0.95))
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor((0.95, 0.95, 0.95))

            cell.set_text_props(fontsize=10)

    plt.title(
        f'Branch Lifecycle Duration Statistics - All Projects\n'
        f'Overall: {overall["total_branches"]:.0f} branches, '
        f'Avg Duration: {overall_avg_min:.0f} minutes, '
        f'Merges: {overall["merged_branches"]:.0f} ({overall["valid_merges"]:.0f} valid, {overall["invalid_merges"]:.0f} invalid)',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Statistics visualization saved: {output_file}")


def main():
    """Analyze branch lifecycles across all projects"""

    print("=" * 100)
    print("BRANCH LIFECYCLE DURATION ANALYSIS")
    print("=" * 100)
    print()

    # Load projects from config
    from core import load_project_config
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Analyzing {len(projects)} projects...\n")

    # Determine project type from first project for letter-based structure
    first_project_name = projects[0][1] if projects else None
    if not first_project_name:
        print("[ERROR] No projects found")
        return 1

    project_type = get_project_letter(first_project_name)

    # Create output directories (letter-based structure)
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations' / project_type / 'branch_lifecycle'  # Per-project charts
    summary_dir = base_dir / 'visualizations' / project_type / 'summary' / 'branch_lifecycle'  # Summary files
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect data from all projects
    all_lifecycle_data = []

    for project_id, project_name in projects:
        # Extract letter and number to create label (e.g., "A01", "B05")
        letter = get_project_letter(project_name)
        number = get_project_number(project_name)
        project_label = f"{letter.upper()}{number}"

        print(f"Analyzing {project_label}...", end=' ')

        lifecycle_data = analyze_branch_lifecycle(project_name)

        # Add project label to each record
        for record in lifecycle_data:
            record['project'] = project_label

        all_lifecycle_data.extend(lifecycle_data)

        print(f"[OK] {len(lifecycle_data)} branches")

    # Create DataFrame
    df = pd.DataFrame(all_lifecycle_data)

    # Save detailed CSV
    csv_file = summary_dir / 'branch_lifecycle_durations.csv'
    df.to_csv(csv_file, index=False, float_format='%.2f')
    print(f"\n[OK] Detailed data saved: {csv_file}")

    # Calculate statistics per project
    stats = []
    for project in sorted(df['project'].unique()):
        project_df = df[df['project'] == project]

        valid_merges = len(project_df[project_df['mr_state'] == 'valid_merge'])
        invalid_merges = len(project_df[project_df['mr_state'] == 'invalid_merge'])

        stats.append({
            'project': project,
            'total_branches': len(project_df),
            'avg_duration_days': project_df['development_duration_days'].mean(),
            'median_duration_days': project_df['development_duration_days'].median(),
            'min_duration_days': project_df['development_duration_days'].min(),
            'max_duration_days': project_df['development_duration_days'].max(),
            'valid_merges': valid_merges,
            'invalid_merges': invalid_merges,
            'merged_branches': valid_merges + invalid_merges,
            'avg_pipelines': project_df['total_pipelines'].mean()
        })

    # Add overall statistics
    overall_valid = len(df[df['mr_state'] == 'valid_merge'])
    overall_invalid = len(df[df['mr_state'] == 'invalid_merge'])

    stats.append({
        'project': 'OVERALL',
        'total_branches': len(df),
        'avg_duration_days': df['development_duration_days'].mean(),
        'median_duration_days': df['development_duration_days'].median(),
        'min_duration_days': df['development_duration_days'].min(),
        'max_duration_days': df['development_duration_days'].max(),
        'valid_merges': overall_valid,
        'invalid_merges': overall_invalid,
        'merged_branches': overall_valid + overall_invalid,
        'avg_pipelines': df['total_pipelines'].mean()
    })

    stats_df = pd.DataFrame(stats)

    # Save statistics CSV
    stats_file = summary_dir / 'branch_lifecycle_statistics.csv'
    stats_df.to_csv(stats_file, index=False, float_format='%.2f')
    print(f"[OK] Statistics saved: {stats_file}")

    # Create visualizations
    print()
    print("=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)
    print()

    # Create per-project duration charts
    charts_created = 0
    for project in sorted(df['project'].unique()):
        project_data = df[df['project'] == project].to_dict('records')
        chart_file = output_dir / f'{project.lower()}_branch_durations.png'
        create_project_duration_chart(project, project_data, chart_file)
        charts_created += 1
        print(f"  [OK] {project} duration chart saved")

    print()

    # Statistics table
    stats_viz_file = summary_dir / 'branch_lifecycle_statistics.png'
    create_statistics_visualization(stats_df, stats_viz_file)

    print(f"\n[OK] Created {charts_created} duration charts (one per project)")

    # Print summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Total branches analyzed: {len(df)}")
    print(f"Average branch duration: {df['development_duration_days'].mean():.1f} days")
    print(f"Median branch duration: {df['development_duration_days'].median():.1f} days")
    print(f"Shortest branch: {df['development_duration_days'].min():.2f} days")
    print(f"Longest branch: {df['development_duration_days'].max():.1f} days")
    print()
    print("Merge Quality:")
    valid_merges = len(df[df['mr_state'] == 'valid_merge'])
    invalid_merges = len(df[df['mr_state'] == 'invalid_merge'])
    total_merges = valid_merges + invalid_merges
    print(f"  Valid merges: {valid_merges}")
    print(f"  Invalid merges: {invalid_merges}")
    print(f"  Total merged: {total_merges}")
    if total_merges > 0:
        print(f"  Invalid rate: {invalid_merges/total_merges*100:.1f}%")
    print()

    # Branch duration distribution
    print("Branch duration distribution:")
    print(f"  < 1 day:  {len(df[df['development_duration_days'] < 1])} branches")
    print(f"  1-3 days: {len(df[(df['development_duration_days'] >= 1) & (df['development_duration_days'] < 3)])} branches")
    print(f"  3-7 days: {len(df[(df['development_duration_days'] >= 3) & (df['development_duration_days'] < 7)])} branches")
    print(f"  > 7 days: {len(df[df['development_duration_days'] >= 7])} branches")

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
