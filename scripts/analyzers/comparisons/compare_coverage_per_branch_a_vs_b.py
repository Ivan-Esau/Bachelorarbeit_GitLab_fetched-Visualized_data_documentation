"""
Compare Coverage per Branch: Type A vs Type B

Creates visualization comparing average test coverage per branch/issue number
between Type A and Type B projects.

Data Source:
- visualizations/a/coverage_per_branch/{project}/final_coverage_statistics.csv
- visualizations/b/coverage_per_branch/{project}/final_coverage_statistics.csv

Output:
- visualizations/comparisons/coverage_per_branch_a_vs_b.png
- visualizations/comparisons/coverage_per_branch_a_vs_b_statistics.csv

Usage:
    python compare_coverage_per_branch_a_vs_b.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from core.config_loader import get_projects_by_letter

# Set style
sns.set_style("whitegrid")


def load_all_coverage_data(project_type):
    """
    Load coverage data from all projects of a given type

    Args:
        project_type: 'a' or 'b'

    Returns:
        DataFrame with all coverage data
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    all_data = []

    # Get all projects for this type
    projects = get_projects_by_letter(project_type)

    for letter, number, gitlab_name in projects:
        project_id = f"{letter}{number}"
        project_label = f"{letter.upper()}{number}"

        # Path to statistics CSV
        stats_file = base_dir / 'visualizations' / project_type / 'coverage_per_branch' / project_id / 'final_coverage_statistics.csv'

        if not stats_file.exists():
            continue

        # Load CSV
        df = pd.read_csv(stats_file)
        df['project_id'] = project_id
        df['project_label'] = project_label
        all_data.append(df)

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)


def extract_issue_number(branch_label):
    """
    Extract issue number from branch label

    Args:
        branch_label: e.g., 'Issue #1', 'Issue #2'

    Returns:
        Integer issue number or None
    """
    try:
        if '#' in branch_label:
            return int(branch_label.split('#')[1])
    except:
        pass
    return None


def create_comparison_visualization(df_a, df_b, output_file):
    """
    Create line/bar chart comparing average coverage per issue

    Args:
        df_a: DataFrame with Type A coverage data
        df_b: DataFrame with Type B coverage data
        output_file: Path to save PNG
    """
    # Extract issue numbers
    df_a['issue_num'] = df_a['Branch'].apply(extract_issue_number)
    df_b['issue_num'] = df_b['Branch'].apply(extract_issue_number)

    # Remove entries without valid issue numbers
    df_a = df_a[df_a['issue_num'].notna()]
    df_b = df_b[df_b['issue_num'].notna()]

    # Calculate average coverage per issue (including min/max for whiskers)
    avg_a = df_a.groupby('issue_num')['Final Coverage (%)'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    avg_b = df_b.groupby('issue_num')['Final Coverage (%)'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()

    # Get common issues (issues that exist in both types)
    common_issues = sorted(set(avg_a['issue_num'].tolist()) & set(avg_b['issue_num'].tolist()))

    if not common_issues:
        print("[ERROR] No common issues found between Type A and Type B")
        return None

    # Filter to common issues
    avg_a = avg_a[avg_a['issue_num'].isin(common_issues)].sort_values('issue_num')
    avg_b = avg_b[avg_b['issue_num'].isin(common_issues)].sort_values('issue_num')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # X positions
    x_positions = np.arange(len(common_issues))
    bar_width = 0.35

    # Create grouped bar chart
    bars_a = ax.bar(x_positions - bar_width/2, avg_a['mean'].values, bar_width,
                    label=f'Type A (n={len(df_a["project_id"].unique())} projects)',
                    color='#5A7A9B', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    bars_b = ax.bar(x_positions + bar_width/2, avg_b['mean'].values, bar_width,
                    label=f'Type B (n={len(df_b["project_id"].unique())} projects)',
                    color='#E67E22', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    # Calculate overall statistics for reference lines
    overall_avg_a = df_a['Final Coverage (%)'].mean()
    overall_avg_b = df_b['Final Coverage (%)'].mean()
    overall_median_a = df_a['Final Coverage (%)'].median()
    overall_median_b = df_b['Final Coverage (%)'].median()

    # Add mean and median reference lines for each type
    ax.axhline(y=overall_avg_a, color='#5A7A9B', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Type A Mean ({overall_avg_a:.1f}%)')
    ax.axhline(y=overall_median_a, color='#5A7A9B', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Type A Median ({overall_median_a:.1f}%)')
    ax.axhline(y=overall_avg_b, color='#E67E22', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Type B Mean ({overall_avg_b:.1f}%)')
    ax.axhline(y=overall_median_b, color='#E67E22', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Type B Median ({overall_median_b:.1f}%)')

    # Customize chart
    ax.set_xlabel('Issue / Branch Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Line Coverage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Test Coverage Comparison per Branch: Type A vs Type B',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'Issue #{int(i)}' for i in common_issues], fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#34495E')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bars in [bars_a, bars_b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Comparison visualization saved: {output_file}")

    # Return statistics for CSV
    return {
        'common_issues': common_issues,
        'avg_a': avg_a,
        'avg_b': avg_b,
        'overall_avg_a': overall_avg_a,
        'overall_avg_b': overall_avg_b,
        'overall_median_a': overall_median_a,
        'overall_median_b': overall_median_b
    }


def create_statistics_table(stats, output_file):
    """
    Create detailed statistics CSV

    Args:
        stats: Dict with statistics from visualization
        output_file: Path to save CSV
    """
    rows = []

    # Per-issue comparison
    for i, issue_num in enumerate(stats['common_issues']):
        avg_a_row = stats['avg_a'][stats['avg_a']['issue_num'] == issue_num].iloc[0]
        avg_b_row = stats['avg_b'][stats['avg_b']['issue_num'] == issue_num].iloc[0]

        rows.append({
            'Issue': f'Issue #{int(issue_num)}',
            'Type A Mean (%)': avg_a_row['mean'],
            'Type A Min (%)': avg_a_row['min'],
            'Type A Max (%)': avg_a_row['max'],
            'Type A Std Dev': avg_a_row['std'],
            'Type A Count': int(avg_a_row['count']),
            'Type B Mean (%)': avg_b_row['mean'],
            'Type B Min (%)': avg_b_row['min'],
            'Type B Max (%)': avg_b_row['max'],
            'Type B Std Dev': avg_b_row['std'],
            'Type B Count': int(avg_b_row['count']),
            'Difference (A-B)': avg_a_row['mean'] - avg_b_row['mean']
        })

    # Overall summary
    rows.append({
        'Issue': '',
        'Type A Mean (%)': '',
        'Type A Min (%)': '',
        'Type A Max (%)': '',
        'Type A Std Dev': '',
        'Type A Count': '',
        'Type B Mean (%)': '',
        'Type B Min (%)': '',
        'Type B Max (%)': '',
        'Type B Std Dev': '',
        'Type B Count': '',
        'Difference (A-B)': ''
    })

    rows.append({
        'Issue': 'OVERALL',
        'Type A Mean (%)': stats['overall_avg_a'],
        'Type A Min (%)': '',
        'Type A Max (%)': '',
        'Type A Std Dev': '',
        'Type A Count': '',
        'Type B Mean (%)': stats['overall_avg_b'],
        'Type B Min (%)': '',
        'Type B Max (%)': '',
        'Type B Std Dev': '',
        'Type B Count': '',
        'Difference (A-B)': stats['overall_avg_a'] - stats['overall_avg_b']
    })

    rows.append({
        'Issue': 'MEDIAN',
        'Type A Mean (%)': stats['overall_median_a'],
        'Type A Min (%)': '',
        'Type A Max (%)': '',
        'Type A Std Dev': '',
        'Type A Count': '',
        'Type B Mean (%)': stats['overall_median_b'],
        'Type B Min (%)': '',
        'Type B Max (%)': '',
        'Type B Std Dev': '',
        'Type B Count': '',
        'Difference (A-B)': stats['overall_median_a'] - stats['overall_median_b']
    })

    df = pd.DataFrame(rows)

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"[OK] Statistics table saved: {output_file}")


def main():
    """Compare coverage per branch between Type A and Type B"""

    print("=" * 100)
    print("COVERAGE PER BRANCH COMPARISON: TYPE A vs TYPE B")
    print("=" * 100)
    print()

    # Load data
    print("Loading coverage data...")
    df_a = load_all_coverage_data('a')
    df_b = load_all_coverage_data('b')

    if df_a is None:
        print("[ERROR] No coverage data found for Type A")
        return 1

    if df_b is None:
        print("[ERROR] No coverage data found for Type B")
        return 1

    print(f"Type A: {len(df_a)} branch entries from {len(df_a['project_id'].unique())} projects")
    print(f"Type B: {len(df_b)} branch entries from {len(df_b['project_id'].unique())} projects")
    print()

    # Define output paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations/comparisons'

    # Create visualization
    print("Creating comparison visualization...")
    output_file = output_dir / 'coverage_per_branch_a_vs_b.png'
    stats = create_comparison_visualization(df_a, df_b, output_file)

    if stats is None:
        return 1

    print()

    # Create statistics table
    print("Creating statistics table...")
    stats_file = output_dir / 'coverage_per_branch_a_vs_b_statistics.csv'
    create_statistics_table(stats, stats_file)
    print()

    # Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Overall Average Coverage:")
    print(f"  Type A: {stats['overall_avg_a']:.1f}% (median: {stats['overall_median_a']:.1f}%)")
    print(f"  Type B: {stats['overall_avg_b']:.1f}% (median: {stats['overall_median_b']:.1f}%)")
    print(f"  Difference: {abs(stats['overall_avg_a'] - stats['overall_avg_b']):.1f}%")
    print()

    if stats['overall_avg_a'] > stats['overall_avg_b']:
        print(f"[>] Type A has {stats['overall_avg_a'] - stats['overall_avg_b']:.1f}% higher average coverage")
    else:
        print(f"[>] Type B has {stats['overall_avg_b'] - stats['overall_avg_a']:.1f}% higher average coverage")

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
