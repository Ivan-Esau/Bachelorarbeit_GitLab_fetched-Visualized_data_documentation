"""
Merge Quality Analysis Script

Analyzes all merge requests across all projects to determine:
- How many merges occurred
- How many were valid (pipeline succeeded)
- How many were invalid (pipeline failed)

Validation Criteria:
- Valid merge: Last pipeline had compile=success AND test=success
- Invalid merge: Last pipeline had any failure

Output:
    - merge_quality_statistics.csv: Per-project and overall statistics
    - Console report with detailed breakdown

Usage:
    python analyze_merge_quality.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def parse_datetime(dt_string):
    """Parse GitLab datetime string"""
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        return None


def analyze_project_merges(project_name, data_base_dir=None):
    """
    Analyze merge quality for a single project

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data

    Returns:
        Dict with merge statistics
    """
    if data_base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent
        data_base_dir = base_dir / 'data_raw'

    project_dir = Path(data_base_dir) / project_name

    # Load data
    with open(project_dir / 'branches.json') as f:
        branches = json.load(f)
    with open(project_dir / 'merge_requests.json') as f:
        merge_requests = json.load(f)
    with open(project_dir / 'pipelines.json') as f:
        pipelines = json.load(f)

    # Filter merged MRs
    merged_mrs = [mr for mr in merge_requests if mr.get('state') == 'merged']

    valid_merges = []
    invalid_merges = []

    for mr in merged_mrs:
        source_branch = mr['source_branch']
        merged_at = parse_datetime(mr.get('merged_at'))

        # Find all pipelines for this branch
        branch_pipelines = [p for p in pipelines if p['ref'] == source_branch]

        if not branch_pipelines:
            continue

        # Sort by created_at
        branch_pipelines.sort(key=lambda p: parse_datetime(p['created_at']))

        # Find last pipeline before merge
        pipelines_before_merge = [
            p for p in branch_pipelines
            if parse_datetime(p['created_at']) <= merged_at
        ]

        if not pipelines_before_merge:
            continue

        last_pipeline = pipelines_before_merge[-1]

        # Get job statuses
        compile_status = None
        test_status = None

        for job in last_pipeline.get('jobs', []):
            if job['stage'] == 'compile':
                compile_status = job['status']
            elif job['stage'] == 'test':
                test_status = job['status']

        # Determine validity
        is_valid = (compile_status == 'success' and test_status == 'success')

        merge_info = {
            'mr_iid': mr['iid'],
            'title': mr['title'],
            'source_branch': source_branch,
            'merged_at': merged_at,
            'pipeline_id': last_pipeline['id'],
            'compile_status': compile_status,
            'test_status': test_status,
            'is_valid': is_valid
        }

        if is_valid:
            valid_merges.append(merge_info)
        else:
            invalid_merges.append(merge_info)

    # Count total branches (exclude master/main)
    total_branches = len([b for b in branches if b['name'].lower() not in ['master', 'main']])

    return {
        'total_branches': total_branches,
        'total_merges': len(valid_merges) + len(invalid_merges),
        'valid_merges': len(valid_merges),
        'invalid_merges': len(invalid_merges),
        'valid_merge_details': valid_merges,
        'invalid_merge_details': invalid_merges
    }


def create_merge_quality_visualization(stats_df, output_file):
    """
    Create visualization table for merge quality statistics

    Args:
        stats_df: DataFrame with merge quality statistics
        output_file: Path to save PNG
    """
    # Prepare data for visualization (exclude TOTAL row for the table)
    table_data = []

    for _, row in stats_df.iterrows():
        if row['Project'] == 'TOTAL':
            continue

        table_data.append([
            row['Project'],
            f"{row['Total Branches']:.0f}",
            f"{row['Total Merges']:.0f}",
            f"{row['Valid Merges']:.0f}",
            f"{row['Invalid Merges']:.0f}",
            f"{row['Success Rate (%)']:.1f}%" if row['Total Merges'] > 0 else "N/A",
            f"{row['Invalid Rate (%)']:.1f}%" if row['Total Merges'] > 0 else "N/A",
            f"{row['Solved Issue (%)']:.1f}%"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Column headers
    columns = ['Project', 'Feature\nBranches', 'Total\nMerges', 'Valid\nMerges ✓',
               'Invalid\nMerges ✗', 'Success\nRate', 'Invalid\nRate', 'Solved\nIssue %']

    # Create table
    table = ax.table(
        cellText=table_data,
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
    for i in range(len(table_data)):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]

            # Color Valid Merges column (green if > 0)
            if j == 3:  # Valid Merges
                value = int(table_data[i][3])
                if value > 0:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))

            # Color Invalid Merges column (red if > 0)
            elif j == 4:  # Invalid Merges
                value = int(table_data[i][4])
                if value > 0:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))

            # Color Success Rate column
            elif j == 5:  # Success Rate
                if table_data[i][5] != "N/A":
                    rate = float(table_data[i][5].replace('%', ''))
                    if rate == 100:
                        cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                    elif rate >= 50:
                        cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                    else:
                        cell.set_facecolor((1.0, 0.5, 0.5))  # Red
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))

            # Color Invalid Rate column
            elif j == 6:  # Invalid Rate
                if table_data[i][6] != "N/A":
                    rate = float(table_data[i][6].replace('%', ''))
                    if rate == 0:
                        cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                    elif rate <= 50:
                        cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                    else:
                        cell.set_facecolor((1.0, 0.5, 0.5))  # Red
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))

            # Color Solved Issue % column
            elif j == 7:  # Solved Issue %
                rate = float(table_data[i][7].replace('%', ''))
                if rate >= 50:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                elif rate > 0:
                    cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                else:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red
            else:
                cell.set_facecolor((0.95, 0.95, 0.95))

            cell.set_text_props(fontsize=10)

    # Get TOTAL row data for title
    total_row = stats_df[stats_df['Project'] == 'TOTAL'].iloc[0]

    # Add title
    plt.title(
        f'Merge Quality Analysis - All Projects\n'
        f'Total: {total_row["Total Merges"]:.0f} merges | '
        f'Valid: {total_row["Valid Merges"]:.0f} ({total_row["Success Rate (%)"]:.1f}%) | '
        f'Invalid: {total_row["Invalid Merges"]:.0f} ({total_row["Invalid Rate (%)"]:.1f}%) | '
        f'Overall Solved Issues: {total_row["Solved Issue (%)"]:.1f}%',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Visualization saved: {output_file}")


def main():
    """Analyze merge quality across all projects"""

    print("=" * 100)
    print("MERGE QUALITY ANALYSIS")
    print("=" * 100)
    print()

    # Load projects from config
    from core import load_project_config
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Analyzing {len(projects)} projects...\n")

    # Analyze each project
    project_stats = []
    all_valid_details = []
    all_invalid_details = []

    for project_id, project_name in projects:
        project_label = project_name.replace('ba_project_', '').replace('_battleship', '').upper()

        print(f"Analyzing {project_label}...", end=' ')

        stats = analyze_project_merges(project_name)

        total_branches = stats['total_branches']
        total = stats['total_merges']
        valid = stats['valid_merges']
        invalid = stats['invalid_merges']

        if total > 0:
            success_rate = (valid / total * 100)
            invalid_rate = (invalid / total * 100)
            print(f"[OK] {total} merges ({valid} valid, {invalid} invalid)")
        else:
            success_rate = 0
            invalid_rate = 0
            print(f"[OK] No merges")

        # Calculate solved issue percentage (valid merges / total branches)
        if total_branches > 0:
            solved_issue_rate = (valid / total_branches * 100)
        else:
            solved_issue_rate = 0

        project_stats.append({
            'Project': project_label,
            'Total Branches': total_branches,
            'Total Merges': total,
            'Valid Merges': valid,
            'Invalid Merges': invalid,
            'Success Rate (%)': success_rate,
            'Invalid Rate (%)': invalid_rate,
            'Solved Issue (%)': solved_issue_rate
        })

        # Collect details
        for merge in stats['valid_merge_details']:
            merge['project'] = project_label
            all_valid_details.append(merge)

        for merge in stats['invalid_merge_details']:
            merge['project'] = project_label
            all_invalid_details.append(merge)

    print()

    # Create output directory
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations/summary/quality_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics CSV
    stats_df = pd.DataFrame(project_stats)

    # Add totals row
    totals = {
        'Project': 'TOTAL',
        'Total Branches': stats_df['Total Branches'].sum(),
        'Total Merges': stats_df['Total Merges'].sum(),
        'Valid Merges': stats_df['Valid Merges'].sum(),
        'Invalid Merges': stats_df['Invalid Merges'].sum(),
        'Success Rate (%)': 0,
        'Invalid Rate (%)': 0,
        'Solved Issue (%)': 0
    }

    if totals['Total Merges'] > 0:
        totals['Success Rate (%)'] = (totals['Valid Merges'] / totals['Total Merges'] * 100)
        totals['Invalid Rate (%)'] = (totals['Invalid Merges'] / totals['Total Merges'] * 100)

    if totals['Total Branches'] > 0:
        totals['Solved Issue (%)'] = (totals['Valid Merges'] / totals['Total Branches'] * 100)

    stats_df = pd.concat([stats_df, pd.DataFrame([totals])], ignore_index=True)

    stats_file = output_dir / 'merge_quality_statistics.csv'
    stats_df.to_csv(stats_file, index=False, float_format='%.1f')
    print(f"[OK] Statistics saved: {stats_file}")

    # Create visualization
    viz_file = output_dir / 'merge_quality_table.png'
    create_merge_quality_visualization(stats_df, viz_file)

    # Save detailed merge lists
    if all_valid_details:
        valid_df = pd.DataFrame(all_valid_details)
        valid_file = output_dir / 'valid_merges_details.csv'
        valid_df.to_csv(valid_file, index=False)
        print(f"[OK] Valid merge details saved: {valid_file}")

    if all_invalid_details:
        invalid_df = pd.DataFrame(all_invalid_details)
        invalid_file = output_dir / 'invalid_merges_details.csv'
        invalid_df.to_csv(invalid_file, index=False)
        print(f"[OK] Invalid merge details saved: {invalid_file}")

    print()

    # Print summary report
    print("=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100)
    print()

    # Print table
    print(f"{'Project':<10} {'Branches':>10} {'Merges':>8} {'Valid':>8} {'Invalid':>8} "
          f"{'Success %':>12} {'Invalid %':>12} {'Solved %':>12}")
    print("-" * 110)

    for _, row in stats_df.iterrows():
        if row['Total Merges'] > 0:
            print(f"{row['Project']:<10} {row['Total Branches']:>10.0f} {row['Total Merges']:>8.0f} "
                  f"{row['Valid Merges']:>8.0f} {row['Invalid Merges']:>8.0f} "
                  f"{row['Success Rate (%)']:>11.1f}% {row['Invalid Rate (%)']:>11.1f}% "
                  f"{row['Solved Issue (%)']:>11.1f}%")
        else:
            print(f"{row['Project']:<10} {row['Total Branches']:>10.0f} {row['Total Merges']:>8.0f} "
                  f"{row['Valid Merges']:>8.0f} {row['Invalid Merges']:>8.0f} "
                  f"{'N/A':>12} {'N/A':>12} {row['Solved Issue (%)']:>11.1f}%")

    print()

    # Key findings
    print("KEY FINDINGS:")
    print("-" * 100)

    total_merges = totals['Total Merges']
    valid_merges = totals['Valid Merges']
    invalid_merges = totals['Invalid Merges']

    print(f"Total merges across all projects: {total_merges}")
    print(f"  ✓ Valid merges (pipeline succeeded): {valid_merges} ({valid_merges/total_merges*100:.1f}%)")
    print(f"  ✗ Invalid merges (pipeline failed): {invalid_merges} ({invalid_merges/total_merges*100:.1f}%)")
    print()

    # Projects with perfect discipline
    perfect_projects = stats_df[(stats_df['Invalid Merges'] == 0) & (stats_df['Total Merges'] > 0)]
    if len(perfect_projects) > 0:
        print(f"Projects with 100% valid merges: {len(perfect_projects)}")
        for _, row in perfect_projects.iterrows():
            if row['Project'] != 'TOTAL':
                print(f"  - {row['Project']}: {row['Total Merges']:.0f} merges, all valid")

    print()

    # Projects with invalid merges
    invalid_projects = stats_df[(stats_df['Invalid Merges'] > 0)]
    if len(invalid_projects) > 0:
        print(f"Projects with invalid merges: {len(invalid_projects) - 1}")  # -1 to exclude TOTAL row
        for _, row in invalid_projects.iterrows():
            if row['Project'] != 'TOTAL':
                print(f"  - {row['Project']}: {row['Invalid Merges']:.0f}/{row['Total Merges']:.0f} "
                      f"invalid ({row['Invalid Rate (%)']:.0f}%)")

    print()

    # Invalid merge breakdown
    if all_invalid_details:
        print("INVALID MERGE REASONS:")
        print("-" * 100)

        compile_failed = sum(1 for m in all_invalid_details if m['compile_status'] != 'success')
        test_failed = sum(1 for m in all_invalid_details
                         if m['compile_status'] == 'success' and m['test_status'] != 'success')

        print(f"Compile failures: {compile_failed}")
        print(f"Test failures (compile OK): {test_failed}")
        print()

        if test_failed == len(all_invalid_details):
            print("⚠ NOTE: All invalid merges had successful compile but failed tests")
            print("  This suggests students waited for code to compile before merging,")
            print("  but merged despite test failures.")

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
