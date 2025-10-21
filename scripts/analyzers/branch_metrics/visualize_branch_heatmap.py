"""
Branch-Level Heat Map Visualization

Creates a comprehensive heat map table showing all branches across all projects
with color-coded metrics for quick comparison.

Output:
    - branch_heatmap_all_projects.png: Heat map table with all 60 branches
    - branch_heatmap_statistics.csv: Summary statistics

Usage:
    python visualize_branch_heatmap.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def extract_issue_number(branch_name):
    """Extract issue number from branch name"""
    if 'issue-' in branch_name.lower():
        parts = branch_name.lower().split('issue-')
        if len(parts) > 1:
            issue_num = parts[1].split('-')[0]
            return f"Issue #{issue_num}"
    elif branch_name == 'master':
        return 'Master'
    return branch_name[:20]


def aggregate_branch_data(project_name):
    """Aggregate all metrics for each branch in a project"""

    base_dir = Path(__file__).parent.parent.parent.parent
    project_dir = base_dir / 'data_raw' / project_name

    # Load all data files
    with open(project_dir / 'branches.json') as f:
        branches = json.load(f)
    with open(project_dir / 'pipelines.json') as f:
        pipelines = json.load(f)
    with open(project_dir / 'coverage.json') as f:
        coverage = json.load(f)
    with open(project_dir / 'merge_requests.json') as f:
        merge_requests = json.load(f)

    # Initialize branch data structure
    branch_data = {}

    for branch in branches:
        branch_name = branch['name']
        branch_data[branch_name] = {
            'branch_name': branch_name,
            'issue_label': extract_issue_number(branch_name),
            'pipelines': [],
            'compile_jobs': [],
            'test_jobs': [],
            'coverage_records': [],
            'mr_info': None
        }

    # Map pipelines to branches (sorted by created_at for timeline)
    for pipeline in pipelines:
        ref = pipeline['ref']
        if ref in branch_data:
            branch_data[ref]['pipelines'].append(pipeline)

            for job in pipeline.get('jobs', []):
                if job['stage'] == 'compile':
                    branch_data[ref]['compile_jobs'].append(job)
                elif job['stage'] == 'test':
                    branch_data[ref]['test_jobs'].append(job)

    # Sort pipelines by created_at for each branch
    for data in branch_data.values():
        data['pipelines'].sort(key=lambda p: p['created_at'])

    # Map coverage to branches
    pipeline_to_branch = {p['id']: p['ref'] for p in pipelines}
    for cov in coverage:
        pipeline_id = cov['pipeline_id']
        if pipeline_id in pipeline_to_branch:
            branch_name = pipeline_to_branch[pipeline_id]
            if branch_name in branch_data:
                branch_data[branch_name]['coverage_records'].append(cov)

    # Map MRs to branches
    for mr in merge_requests:
        source_branch = mr['source_branch']
        if source_branch in branch_data:
            branch_data[source_branch]['mr_info'] = mr

    # Calculate metrics for each branch
    results = []

    for branch_name, data in branch_data.items():
        if len(data['pipelines']) == 0:
            continue  # Skip branches with no pipelines

        # Compile metrics
        compile_total = len(data['compile_jobs'])
        compile_success = sum(1 for j in data['compile_jobs'] if j['status'] == 'success')
        compile_rate = (compile_success / compile_total * 100) if compile_total > 0 else 0

        # Test metrics
        test_total = len(data['test_jobs'])
        test_success = sum(1 for j in data['test_jobs'] if j['status'] == 'success')
        test_rate = (test_success / test_total * 100) if test_total > 0 else 0

        # Coverage metrics - chronological order
        successful_cov = [c for c in data['coverage_records']
                         if c.get('parse_status') == 'success']

        # Sort coverage by pipeline_id to get chronological order
        pipeline_order = {p['id']: i for i, p in enumerate(data['pipelines'])}
        successful_cov.sort(key=lambda c: pipeline_order.get(c['pipeline_id'], 999))

        # Final coverage (last measurement)
        final_cov = successful_cov[-1]['coverage_percentage'] if successful_cov else 0

        # Coverage trend (first → last)
        if len(successful_cov) >= 2:
            first_cov = successful_cov[0]['coverage_percentage']
            cov_trend = final_cov - first_cov
        else:
            cov_trend = 0

        # Determine which pipeline to use for validation
        # For merged branches, use last pipeline BEFORE merge
        # For other branches, use the last pipeline overall
        validation_pipeline = data['pipelines'][-1]  # Default to last

        if data['mr_info'] and data['mr_info'].get('merged_at'):
            # Get merge timestamp
            merge_time = datetime.fromisoformat(
                data['mr_info']['merged_at'].replace('Z', '+00:00')
            )

            # Filter to pipelines before or at merge time
            pipelines_before_merge = [
                p for p in data['pipelines']
                if datetime.fromisoformat(p['created_at'].replace('Z', '+00:00')) <= merge_time
            ]

            if pipelines_before_merge:
                validation_pipeline = pipelines_before_merge[-1]

        # Get compile/test status from validation pipeline
        last_compile_status = None
        last_test_status = None

        for job in validation_pipeline.get('jobs', []):
            if job['stage'] == 'compile':
                last_compile_status = job['status']
            elif job['stage'] == 'test':
                last_test_status = job['status']

        # MR status with merge validation
        if data['mr_info']:
            mr_state = data['mr_info']['state']

            # Validate merge if it was merged
            if mr_state == 'merged':
                # Check if last pipeline BEFORE merge had both compile and test success
                is_valid_merge = (last_compile_status == 'success' and
                                 last_test_status == 'success')
                mr_state = 'valid_merge' if is_valid_merge else 'invalid_merge'
        else:
            mr_state = 'no_mr'

        results.append({
            'project': project_name.replace('ba_project_', '').replace('_battleship', '').upper(),
            'issue': data['issue_label'],
            'compile_rate': compile_rate,
            'test_rate': test_rate,
            'final_coverage': final_cov,
            'cov_trend': cov_trend,
            'last_compile_status': last_compile_status,
            'last_test_status': last_test_status,
            'mr_state': mr_state,
            'branch_name': branch_name
        })

    return results


def get_color(value, metric_type='percentage'):
    """
    Get color for a cell based on value

    Args:
        value: Numeric value
        metric_type: 'percentage' or 'trend'

    Returns:
        RGB color tuple
    """
    if metric_type == 'percentage':
        if value >= 70:
            return (0.4, 0.8, 0.4)  # Green
        elif value >= 30:
            return (1.0, 0.9, 0.4)  # Yellow
        else:
            return (1.0, 0.5, 0.5)  # Red
    elif metric_type == 'trend':
        if value > 10:
            return (0.4, 0.8, 0.4)  # Green - improved
        elif value < -10:
            return (1.0, 0.5, 0.5)  # Red - degraded
        else:
            return (0.9, 0.9, 0.9)  # Light gray - stable


def get_trend_indicator(trend_value):
    """
    Get trend indicator symbol

    Args:
        trend_value: Coverage trend (positive = improved, negative = degraded)

    Returns:
        String with trend indicator
    """
    if trend_value > 10:
        return '⬆'  # Improved significantly
    elif trend_value < -10:
        return '⬇'  # Degraded significantly
    else:
        return '→'  # Stable


def get_status_indicator(status):
    """
    Get status indicator symbol

    Args:
        status: Job status (success/failed/etc)

    Returns:
        String with status symbol
    """
    if status == 'success':
        return '✓'
    elif status == 'failed':
        return '✗'
    elif status == 'canceled':
        return '○'
    elif status == 'skipped':
        return '⊘'
    else:
        return '?'


def get_mr_status_display(mr_state):
    """
    Get display text for MR status

    Args:
        mr_state: MR state (valid_merge, invalid_merge, opened, no_mr)

    Returns:
        Formatted string for display
    """
    if mr_state == 'valid_merge':
        return '✓ Valid Merge'
    elif mr_state == 'invalid_merge':
        return '✗ Invalid Merge'
    elif mr_state == 'opened':
        return 'Open MR'
    elif mr_state == 'no_mr':
        return 'No MR'
    else:
        return mr_state.replace('_', ' ').title()


def get_mr_status_color(mr_state):
    """
    Get color for MR status cell

    Args:
        mr_state: MR state (valid_merge, invalid_merge, opened, no_mr)

    Returns:
        RGB color tuple
    """
    if mr_state == 'valid_merge':
        return (0.4, 0.8, 0.4)  # Green - valid merge
    elif mr_state == 'invalid_merge':
        return (1.0, 0.5, 0.5)  # Red - invalid merge
    elif mr_state == 'opened':
        return (1.0, 0.9, 0.4)  # Yellow - open MR
    else:  # no_mr
        return (0.95, 0.95, 0.95)  # Light gray - no MR


def create_project_heatmap(project_label, project_df, output_file):
    """
    Create heat map table for a single project

    Args:
        project_label: Project label (e.g., 'A01')
        project_df: DataFrame with branch metrics for this project
        output_file: Path to save PNG
    """
    # Prepare data for table
    table_data = []
    row_colors = []

    for idx, row in project_df.iterrows():
        table_data.append([
            row['issue'],
            f"{row['compile_rate']:.0f}%",
            f"{row['test_rate']:.0f}%",
            f"{row['final_coverage']:.0f}%",
            get_mr_status_display(row['mr_state'])
        ])

        # Store colors for this row
        row_colors.append([
            (0.95, 0.95, 0.95),  # issue (light gray)
            get_color(row['compile_rate'], 'percentage'),
            get_color(row['test_rate'], 'percentage'),
            get_color(row['final_coverage'], 'percentage'),
            get_mr_status_color(row['mr_state'])
        ])

    # Create figure with appropriate height based on number of rows
    num_rows = len(table_data)
    fig_height = max(4, num_rows * 0.6 + 2)  # Dynamic height
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Column headers
    columns = ['Branch/Issue', 'Build ✓', 'Test ✓', 'Final Cov', 'MR Status']

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.20]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Color header row
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5A7A9B')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Color data cells
    for i in range(len(table_data)):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_colors[i][j])
            cell.set_text_props(fontsize=9)

    # Add title
    plt.title(
        f'Project {project_label}: Branch-Level Metrics Heat Map\n'
        'Metrics: Green (>70%), Yellow (30-70%), Red (<30%) | MR: Green=Valid, Red=Invalid',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=(0.4, 0.8, 0.4), label='Good (>70%) / Valid Merge'),
        mpatches.Patch(facecolor=(1.0, 0.9, 0.4), label='Moderate (30-70%) / Open MR'),
        mpatches.Patch(facecolor=(1.0, 0.5, 0.5), label='Poor (<30%) / Invalid Merge'),
        mpatches.Patch(facecolor=(0.95, 0.95, 0.95), label='No MR')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             framealpha=0.95, bbox_to_anchor=(0.98, 0.98))

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {project_label} heat map saved")


def create_summary_table(df, output_file):
    """
    Create summary table showing merge quality per project

    Args:
        df: DataFrame with all branch data
        output_file: Path to save PNG
    """
    # Calculate per-project statistics (feature branches only for averages)
    summary_data = []

    for project in sorted(df['project'].unique()):
        project_df = df[df['project'] == project]
        feature_df = project_df[project_df['issue'] != 'Master']

        total_branches = len(feature_df)  # Feature branches only
        valid_merges = len(project_df[project_df['mr_state'] == 'valid_merge'])
        invalid_merges = len(project_df[project_df['mr_state'] == 'invalid_merge'])
        open_mrs = len(project_df[project_df['mr_state'] == 'opened'])
        no_mrs = len(project_df[project_df['mr_state'] == 'no_mr'])

        total_merges = valid_merges + invalid_merges
        invalid_rate = (invalid_merges / total_merges * 100) if total_merges > 0 else 0

        # Calculate averages for FEATURE BRANCHES ONLY
        avg_compile = feature_df['compile_rate'].mean()
        avg_test = feature_df['test_rate'].mean()
        avg_coverage = feature_df['final_coverage'].mean()

        summary_data.append([
            project,
            total_branches,
            f"{avg_compile:.0f}%",
            f"{avg_test:.0f}%",
            f"{avg_coverage:.0f}%",
            valid_merges,
            invalid_merges,
            f"{invalid_rate:.0f}%" if total_merges > 0 else "N/A"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Column headers
    columns = ['Project', 'Feature\nBranches', 'Avg Build\n(Features)', 'Avg Test\n(Features)', 'Avg Cov\n(Features)',
               'Valid ✓', 'Invalid ✗', 'Invalid %']

    # Create table
    table = ax.table(
        cellText=summary_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header row
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5A7A9B')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Color data cells
    for i in range(len(summary_data)):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]

            # Color merge quality columns
            if j == 5:  # Valid merges
                value = summary_data[i][5]
                if value > 0:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))
            elif j == 6:  # Invalid merges
                value = summary_data[i][6]
                if value > 0:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))
            elif j == 7:  # Invalid %
                invalid_pct_str = summary_data[i][7]
                if invalid_pct_str != "N/A":
                    invalid_pct = float(invalid_pct_str.replace('%', ''))
                    if invalid_pct > 50:
                        cell.set_facecolor((1.0, 0.5, 0.5))  # Red
                    elif invalid_pct > 0:
                        cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                    else:
                        cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))
            else:
                cell.set_facecolor((0.95, 0.95, 0.95))

            cell.set_text_props(fontsize=10)

    # Add title with feature vs master statistics
    total_valid = sum(row[5] for row in summary_data)
    total_invalid = sum(row[6] for row in summary_data)
    total_all_merges = total_valid + total_invalid
    overall_invalid_rate = (total_invalid / total_all_merges * 100) if total_all_merges > 0 else 0

    # Calculate overall feature vs master averages
    feature_df = df[df['issue'] != 'Master']
    master_df = df[df['issue'] == 'Master']

    plt.title(
        f'Branch-Level Metrics Summary - All Projects\n'
        f'Feature Branches (n={len(feature_df)}): Avg Coverage {feature_df["final_coverage"].mean():.1f}% | '
        f'Master Branches (n={len(master_df)}): Avg Coverage {master_df["final_coverage"].mean():.1f}%\n'
        f'Merges: {total_all_merges} total | Valid: {total_valid} ({total_valid/total_all_merges*100:.1f}%) | '
        f'Invalid: {total_invalid} ({overall_invalid_rate:.1f}%)',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Summary table saved: {output_file}")


def main():
    """Generate branch-level heat map visualization"""

    print("=" * 100)
    print("BRANCH-LEVEL HEAT MAP VISUALIZATION")
    print("=" * 100)
    print()

    # Load projects from config
    from core import load_project_config
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Processing {len(projects)} projects...\n")

    # Aggregate data from all projects
    all_branches = []

    for project_id, project_name in projects:
        print(f"  Loading {project_name}...", end=' ')
        branch_data = aggregate_branch_data(project_name)
        all_branches.extend(branch_data)
        print(f"[OK] ({len(branch_data)} branches)")

    print()
    print(f"Total branches with pipelines: {len(all_branches)}")
    print()

    # Create DataFrame
    df = pd.DataFrame(all_branches)

    # Sort by project, then by issue
    df = df.sort_values(['project', 'issue'])

    # Create output directories
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations/branch_metrics'  # Per-project heatmaps
    summary_dir = base_dir / 'visualizations/summary/branch_metrics'  # Summary files
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...\n")

    # Create per-project heat maps
    heatmaps_created = 0

    for project in df['project'].unique():
        project_df = df[df['project'] == project].copy()
        heatmap_file = output_dir / f'{project.lower()}_branch_heatmap.png'
        create_project_heatmap(project, project_df, heatmap_file)
        heatmaps_created += 1

    print()

    # Create summary table
    summary_file = summary_dir / 'branch_metrics_summary.png'
    create_summary_table(df, summary_file)

    # Save statistics CSV
    stats_file = summary_dir / 'branch_metrics_all.csv'
    df.to_csv(stats_file, index=False, float_format='%.2f')
    print(f"  [OK] CSV data saved: {stats_file}")
    print(f"  [OK] Created {heatmaps_created} heat map tables (one per project)")
    print(f"  [OK] Summary table created")

    # Print summary statistics
    print()
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    # Separate feature and master branches
    feature_df = df[df['issue'] != 'Master']
    master_df = df[df['issue'] == 'Master']

    print(f"Total branches: {len(df)} ({len(feature_df)} feature + {len(master_df)} master)")
    print()

    print("Feature Branches:")
    print(f"  Average compile success rate: {feature_df['compile_rate'].mean():.1f}%")
    print(f"  Average test success rate: {feature_df['test_rate'].mean():.1f}%")
    print(f"  Average final coverage: {feature_df['final_coverage'].mean():.1f}%")
    print()

    print("Master Branches:")
    print(f"  Average compile success rate: {master_df['compile_rate'].mean():.1f}%")
    print(f"  Average test success rate: {master_df['test_rate'].mean():.1f}%")
    print(f"  Average final coverage: {master_df['final_coverage'].mean():.1f}%")
    print()

    print("Overall (All Branches):")
    print(f"  Average compile success rate: {df['compile_rate'].mean():.1f}%")
    print(f"  Average test success rate: {df['test_rate'].mean():.1f}%")
    print(f"  Average final coverage: {df['final_coverage'].mean():.1f}%")
    print()
    print("Coverage Trends:")
    improved = len(df[df['cov_trend'] > 10])
    stable = len(df[df['cov_trend'].abs() <= 10])
    degraded = len(df[df['cov_trend'] < -10])
    print(f"  Improved (>+10%): {improved} branches")
    print(f"  Stable (±10%): {stable} branches")
    print(f"  Degraded (<-10%): {degraded} branches")
    print()
    print(f"Branches with >70% final coverage: {len(df[df['final_coverage'] > 70])}")
    print(f"Branches with >70% test success: {len(df[df['test_rate'] > 70])}")
    print()
    print("Merge Quality:")
    valid_merges = len(df[df['mr_state'] == 'valid_merge'])
    invalid_merges = len(df[df['mr_state'] == 'invalid_merge'])
    total_merges = valid_merges + invalid_merges
    print(f"  Valid merges: {valid_merges}")
    print(f"  Invalid merges: {invalid_merges}")
    if total_merges > 0:
        print(f"  Invalid rate: {invalid_merges/total_merges*100:.1f}%")

    print()
    print("=" * 100)
    print("COMPLETED")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
