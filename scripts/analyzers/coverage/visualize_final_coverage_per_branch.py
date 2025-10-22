"""
Final Coverage per Branch - Per Project Visualization

Creates bar chart showing the final test coverage reached by each feature branch.

Data Source:
- coverage.json: Test coverage metrics from JaCoCo artifacts
- pipelines.json: Pipeline data to link coverage to branches
- branches.json: Branch information

Output:
- visualizations/{type}/{project}/coverage/final_coverage_per_branch.png
- visualizations/{type}/{project}/coverage/final_coverage_statistics.csv

Usage:
    python visualize_final_coverage_per_branch.py --project-type a
    python visualize_final_coverage_per_branch.py --project-type b
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from core import load_project_config
from core.path_helpers import get_data_dir

# Set style
sns.set_style("whitegrid")


def get_final_coverage_per_branch(project_name):
    """
    Get final coverage for each feature branch

    Args:
        project_name: Name of the project directory (e.g., 'a01', 'b05')

    Returns:
        Dict with branch names as keys and final coverage as values
    """
    data_dir = get_data_dir(project_name)

    coverage_file = data_dir / 'coverage.json'
    pipelines_file = data_dir / 'pipelines.json'

    if not coverage_file.exists() or not pipelines_file.exists():
        return None

    # Load data
    with open(coverage_file, 'r', encoding='utf-8') as f:
        coverage_data = json.load(f)

    with open(pipelines_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    # Create pipeline_id to branch mapping
    pipeline_to_branch = {}
    for pipeline in pipelines:
        ref = pipeline.get('ref', '')
        if ref.lower() not in ['master', 'main']:
            pipeline_to_branch[pipeline['id']] = ref

    # Group coverage by branch
    branch_coverage = {}
    for entry in coverage_data:
        # Skip entries with expired or failed parsing
        if entry.get('parse_status') != 'success':
            continue

        # Skip entries without coverage data
        if 'coverage_percentage' not in entry:
            continue

        pipeline_id = entry['pipeline_id']
        if pipeline_id in pipeline_to_branch:
            branch_name = pipeline_to_branch[pipeline_id]
            coverage_pct = entry['coverage_percentage']

            if branch_name not in branch_coverage:
                branch_coverage[branch_name] = []
            branch_coverage[branch_name].append({
                'pipeline_id': pipeline_id,
                'coverage': coverage_pct,
                'branch_rate': entry['branch_rate'] * 100,
                'lines_covered': entry['lines_covered'],
                'lines_total': entry['lines_total'],
                'complexity': entry['complexity']
            })

    # Get final coverage (last pipeline) for each branch
    final_coverage = {}
    for branch, coverages in branch_coverage.items():
        # Sort by pipeline_id to get chronological order
        coverages_sorted = sorted(coverages, key=lambda x: x['pipeline_id'])
        final = coverages_sorted[-1]  # Last pipeline

        # Extract issue number from branch name
        issue_label = extract_issue_label(branch)

        final_coverage[issue_label] = {
            'branch_name': branch,
            'final_coverage': final['coverage'],
            'final_branch_rate': final['branch_rate'],
            'lines_covered': final['lines_covered'],
            'lines_total': final['lines_total'],
            'complexity': final['complexity'],
            'num_pipelines': len(coverages),
            'avg_coverage': sum(c['coverage'] for c in coverages) / len(coverages),
            'first_coverage': coverages_sorted[0]['coverage'],
            'coverage_change': final['coverage'] - coverages_sorted[0]['coverage']
        }

    return final_coverage


def extract_issue_label(branch_name):
    """
    Extract issue label from branch name

    Args:
        branch_name: Full branch name (e.g., 'feature/issue-1-us-01-...')

    Returns:
        Issue label (e.g., 'Issue #1')
    """
    # Try to extract issue number
    if 'issue-' in branch_name.lower():
        try:
            parts = branch_name.lower().split('issue-')
            if len(parts) > 1:
                issue_num = parts[1].split('-')[0]
                return f'Issue #{issue_num}'
        except:
            pass

    # Fallback: use shortened branch name
    return branch_name[:20] + '...' if len(branch_name) > 20 else branch_name


def create_coverage_visualization(coverage_data, project_label, output_file):
    """
    Create bar chart visualization of final coverage per branch

    Args:
        coverage_data: Dict with issue labels as keys and coverage info as values
        project_label: Project label for display (e.g., 'A01')
        output_file: Path to save the PNG file
    """
    if not coverage_data:
        print(f"  [SKIP] No coverage data for {project_label}")
        return

    # Sort by issue number
    sorted_issues = sorted(coverage_data.items(),
                          key=lambda x: int(x[0].split('#')[1]) if '#' in x[0] else 0)

    issue_labels = [issue for issue, _ in sorted_issues]
    coverages = [data['final_coverage'] for _, data in sorted_issues]

    # Create color mapping based on coverage thresholds
    colors = []
    for cov in coverages:
        if cov >= 80:
            colors.append('#27AE60')  # Green - high coverage
        elif cov >= 40:
            colors.append('#F39C12')  # Orange - medium coverage
        else:
            colors.append('#E74C3C')  # Red - low coverage

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    x_pos = np.arange(len(issue_labels))
    bars = ax.bar(x_pos, coverages, color=colors, alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    # Add reference lines
    ax.axhline(y=80, color='#27AE60', linestyle='--', linewidth=1, alpha=0.5, label='High Coverage (80%)')
    ax.axhline(y=40, color='#F39C12', linestyle='--', linewidth=1, alpha=0.5, label='Medium Coverage (40%)')

    # Customize chart
    ax.set_xlabel('Feature Branches', fontsize=11, fontweight='bold')
    ax.set_ylabel('Line Coverage (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Final Test Coverage per Branch - Project {project_label}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(issue_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#34495E')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{cov:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add statistics box
    avg_coverage = sum(coverages) / len(coverages)
    median_coverage = np.median(coverages)
    high_cov_count = sum(1 for c in coverages if c >= 80)
    low_cov_count = sum(1 for c in coverages if c < 40)

    stats_text = (
        f'Statistics:\n'
        f'Mean: {avg_coverage:.1f}%\n'
        f'Median: {median_coverage:.1f}%\n'
        f'High (≥80%): {high_cov_count}/{len(coverages)}\n'
        f'Low (<40%): {low_cov_count}/{len(coverages)}'
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#34495E', linewidth=1.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=props, family='monospace')

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Coverage visualization saved: {output_file}")


def create_statistics_table(coverage_data, project_label, output_file):
    """
    Create detailed statistics CSV table

    Args:
        coverage_data: Dict with issue labels as keys and coverage info as values
        project_label: Project label
        output_file: Path to save the CSV file
    """
    if not coverage_data:
        return

    rows = []
    for issue, data in sorted(coverage_data.items(),
                              key=lambda x: int(x[0].split('#')[1]) if '#' in x[0] else 0):
        rows.append({
            'Project': project_label,
            'Branch': issue,
            'Final Coverage (%)': data['final_coverage'],
            'Branch Coverage (%)': data['final_branch_rate'],
            'Lines Covered': data['lines_covered'],
            'Lines Total': data['lines_total'],
            'Complexity': data['complexity'],
            'Num Pipelines': data['num_pipelines'],
            'Avg Coverage (%)': data['avg_coverage'],
            'First Coverage (%)': data['first_coverage'],
            'Coverage Change (%)': data['coverage_change']
        })

    df = pd.DataFrame(rows)

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"  [OK] Statistics table saved: {output_file}")


def main():
    """Process all projects"""

    import argparse
    parser = argparse.ArgumentParser(description='Visualize final coverage per branch')
    parser.add_argument('--project-type', required=True, choices=['a', 'b'],
                       help='Project type: a or b')
    args = parser.parse_args()

    project_type = args.project_type

    print("=" * 100)
    print(f"FINAL COVERAGE PER BRANCH VISUALIZATION - TYPE {project_type.upper()}")
    print("=" * 100)
    print()

    # Load project types config and manually build project list for the requested type
    from core.config_loader import get_projects_by_letter

    # Get projects for the specified type directly (not limited by active_project_types)
    projects = get_projects_by_letter(project_type)

    print(f"Processing {len(projects)} projects for Type {project_type.upper()}...")
    print()

    # Base output directory
    base_dir = Path(__file__).parent.parent.parent.parent

    # Process each project
    success_count = 0
    for letter, number, gitlab_name in projects:
        # Build project ID and label
        project_id = f"{letter}{number}"  # 'a01', 'b05', etc.
        project_label = f"{letter.upper()}{number}"  # 'A01', 'B05', etc.
        project_name = project_id  # Use project_id as project_name

        print(f"Processing {project_label} ({project_name})...")

        # Get coverage data
        coverage_data = get_final_coverage_per_branch(project_name)

        if not coverage_data:
            print(f"  [SKIP] No coverage data available")
            print()
            continue

        # Create output directory: visualizations/{type}/coverage_per_branch/{project}/
        output_dir = base_dir / 'visualizations' / project_type / 'coverage_per_branch' / project_name

        # Create visualization
        output_file = output_dir / 'final_coverage_per_branch.png'
        create_coverage_visualization(coverage_data, project_label, output_file)

        # Create statistics table
        stats_file = output_dir / 'final_coverage_statistics.csv'
        create_statistics_table(coverage_data, project_label, stats_file)

        success_count += 1
        print()

    print("=" * 100)
    print(f"COMPLETED: {success_count}/{len(projects)} projects processed")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
