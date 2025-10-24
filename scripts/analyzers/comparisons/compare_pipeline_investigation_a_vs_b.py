"""
Compare Pipeline Investigation: Type A vs Type B

Creates visualizations comparing pipeline problems between Type A and Type B projects:
1. Primary Problem Type Distribution
2. Cancellation Rate per Issue

Data Source:
- analysis_results/a/pipeline_investigation/branch_summaries.csv
- analysis_results/b/pipeline_investigation/branch_summaries.csv

Output:
- visualizations/comparisons/pipeline_problems_a_vs_b.png
- visualizations/comparisons/cancellation_rate_per_issue_a_vs_b.png
- visualizations/comparisons/pipeline_investigation_statistics.csv

Usage:
    python compare_pipeline_investigation_a_vs_b.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from core.path_helpers import get_analysis_file

# Set style
sns.set_style("whitegrid")


def load_branch_summaries(project_type):
    """
    Load branch summaries from pipeline investigation

    Args:
        project_type: 'a' or 'b'

    Returns:
        DataFrame with branch summary data
    """
    file_path = get_analysis_file(project_type, 'pipeline_investigation', 'branch_summaries.csv')

    if not Path(file_path).exists():
        print(f"[ERROR] Branch summaries not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    return df


def extract_issue_number(branch_name):
    """
    Extract issue number from branch name

    Args:
        branch_name: e.g., 'feature/issue-1-us-01-...'

    Returns:
        Integer issue number or None
    """
    try:
        if 'issue-' in branch_name.lower():
            parts = branch_name.lower().split('issue-')
            if len(parts) > 1:
                issue_num = parts[1].split('-')[0]
                return int(issue_num)
    except:
        pass
    return None


def create_primary_problem_comparison(df_a, df_b, output_file):
    """
    Create grouped bar chart comparing primary problem types

    Args:
        df_a: DataFrame with Type A branch summaries
        df_b: DataFrame with Type B branch summaries
        output_file: Path to save PNG
    """
    # Count primary problems for each type
    problem_counts_a = df_a['primary_problem'].value_counts()
    problem_counts_b = df_b['primary_problem'].value_counts()

    # Get all unique problem types
    all_problems = sorted(set(problem_counts_a.index.tolist() + problem_counts_b.index.tolist()))

    # Calculate percentages
    total_a = len(df_a)
    total_b = len(df_b)

    percentages_a = [(problem_counts_a.get(p, 0) / total_a * 100) for p in all_problems]
    percentages_b = [(problem_counts_b.get(p, 0) / total_b * 100) for p in all_problems]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # X positions
    x_positions = np.arange(len(all_problems))
    bar_width = 0.35

    # Create grouped bar chart
    bars_a = ax.bar(x_positions - bar_width/2, percentages_a, bar_width,
                    label=f'Type A (n={total_a} branches)',
                    color='#5A7A9B', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    bars_b = ax.bar(x_positions + bar_width/2, percentages_b, bar_width,
                    label=f'Type B (n={total_b} branches)',
                    color='#E67E22', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    # Customize chart
    ax.set_xlabel('Primary Problem Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Branches (%)', fontsize=11, fontweight='bold')
    ax.set_title('Pipeline Problem Distribution: Type A vs Type B',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in all_problems],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, max(percentages_a + percentages_b) * 1.15)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#34495E')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bars in [bars_a, bars_b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Primary problem comparison saved: {output_file}")

    return {
        'all_problems': all_problems,
        'percentages_a': percentages_a,
        'percentages_b': percentages_b,
        'counts_a': [problem_counts_a.get(p, 0) for p in all_problems],
        'counts_b': [problem_counts_b.get(p, 0) for p in all_problems],
    }


def create_cancellation_rate_comparison(df_a, df_b, output_file):
    """
    Create grouped bar chart comparing cancellation rates per issue

    Args:
        df_a: DataFrame with Type A branch summaries
        df_b: DataFrame with Type B branch summaries
        output_file: Path to save PNG
    """
    # Extract issue numbers
    df_a['issue_num'] = df_a['branch'].apply(extract_issue_number)
    df_b['issue_num'] = df_b['branch'].apply(extract_issue_number)

    # Remove entries without valid issue numbers
    df_a_valid = df_a[df_a['issue_num'].notna()].copy()
    df_b_valid = df_b[df_b['issue_num'].notna()].copy()

    # Calculate average cancellation rate per issue
    avg_a = df_a_valid.groupby('issue_num')['cancellation_rate'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    avg_b = df_b_valid.groupby('issue_num')['cancellation_rate'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()

    # Get common issues
    common_issues = sorted(set(avg_a['issue_num'].tolist()) & set(avg_b['issue_num'].tolist()))

    if not common_issues:
        print("[ERROR] No common issues found between Type A and Type B")
        return None

    # Filter to common issues
    avg_a = avg_a[avg_a['issue_num'].isin(common_issues)].sort_values('issue_num')
    avg_b = avg_b[avg_b['issue_num'].isin(common_issues)].sort_values('issue_num')

    # Calculate overall statistics
    overall_avg_a = df_a_valid['cancellation_rate'].mean()
    overall_avg_b = df_b_valid['cancellation_rate'].mean()
    overall_median_a = df_a_valid['cancellation_rate'].median()
    overall_median_b = df_b_valid['cancellation_rate'].median()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # X positions
    x_positions = np.arange(len(common_issues))
    bar_width = 0.35

    # Create grouped bar chart
    bars_a = ax.bar(x_positions - bar_width/2, avg_a['mean'].values, bar_width,
                    label=f'Type A (n={len(df_a_valid["project"].unique())} projects)',
                    color='#5A7A9B', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

    bars_b = ax.bar(x_positions + bar_width/2, avg_b['mean'].values, bar_width,
                    label=f'Type B (n={len(df_b_valid["project"].unique())} projects)',
                    color='#E67E22', alpha=0.8, edgecolor='#34495E', linewidth=1.5)

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
    ax.set_ylabel('Average Cancellation Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Pipeline Cancellation Rate Comparison per Branch: Type A vs Type B',
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

    print(f"[OK] Cancellation rate comparison saved: {output_file}")

    return {
        'common_issues': common_issues,
        'avg_a': avg_a,
        'avg_b': avg_b,
        'overall_avg_a': overall_avg_a,
        'overall_avg_b': overall_avg_b,
        'overall_median_a': overall_median_a,
        'overall_median_b': overall_median_b
    }


def create_statistics_table(problem_stats, cancellation_stats, output_file):
    """
    Create detailed statistics CSV

    Args:
        problem_stats: Dict with problem comparison statistics
        cancellation_stats: Dict with cancellation rate statistics
        output_file: Path to save CSV
    """
    rows = []

    # Section 1: Primary Problem Distribution
    rows.append({
        'Category': 'PRIMARY PROBLEM DISTRIBUTION',
        'Metric': '',
        'Type A': '',
        'Type B': '',
        'Difference': ''
    })

    for i, problem in enumerate(problem_stats['all_problems']):
        rows.append({
            'Category': 'Problem Type',
            'Metric': problem.replace('_', ' ').title(),
            'Type A': f"{problem_stats['percentages_a'][i]:.1f}% ({problem_stats['counts_a'][i]} branches)",
            'Type B': f"{problem_stats['percentages_b'][i]:.1f}% ({problem_stats['counts_b'][i]} branches)",
            'Difference': f"{problem_stats['percentages_a'][i] - problem_stats['percentages_b'][i]:.1f}%"
        })

    # Empty row separator
    rows.append({
        'Category': '',
        'Metric': '',
        'Type A': '',
        'Type B': '',
        'Difference': ''
    })

    # Section 2: Cancellation Rate per Issue
    if cancellation_stats:
        rows.append({
            'Category': 'CANCELLATION RATE PER ISSUE',
            'Metric': '',
            'Type A': '',
            'Type B': '',
            'Difference': ''
        })

        for i, issue_num in enumerate(cancellation_stats['common_issues']):
            avg_a_row = cancellation_stats['avg_a'][cancellation_stats['avg_a']['issue_num'] == issue_num].iloc[0]
            avg_b_row = cancellation_stats['avg_b'][cancellation_stats['avg_b']['issue_num'] == issue_num].iloc[0]

            rows.append({
                'Category': 'Issue Cancellation',
                'Metric': f'Issue #{int(issue_num)}',
                'Type A': f"{avg_a_row['mean']:.1f}% (n={int(avg_a_row['count'])})",
                'Type B': f"{avg_b_row['mean']:.1f}% (n={int(avg_b_row['count'])})",
                'Difference': f"{avg_a_row['mean'] - avg_b_row['mean']:.1f}%"
            })

        # Empty row separator
        rows.append({
            'Category': '',
            'Metric': '',
            'Type A': '',
            'Type B': '',
            'Difference': ''
        })

        # Overall summary
        rows.append({
            'Category': 'OVERALL CANCELLATION RATE',
            'Metric': 'Mean',
            'Type A': f"{cancellation_stats['overall_avg_a']:.1f}%",
            'Type B': f"{cancellation_stats['overall_avg_b']:.1f}%",
            'Difference': f"{cancellation_stats['overall_avg_a'] - cancellation_stats['overall_avg_b']:.1f}%"
        })

        rows.append({
            'Category': 'OVERALL CANCELLATION RATE',
            'Metric': 'Median',
            'Type A': f"{cancellation_stats['overall_median_a']:.1f}%",
            'Type B': f"{cancellation_stats['overall_median_b']:.1f}%",
            'Difference': f"{cancellation_stats['overall_median_a'] - cancellation_stats['overall_median_b']:.1f}%"
        })

    df = pd.DataFrame(rows)

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"[OK] Statistics table saved: {output_file}")


def main():
    """Compare pipeline investigation between Type A and Type B"""

    print("=" * 100)
    print("PIPELINE INVESTIGATION COMPARISON: TYPE A vs TYPE B")
    print("=" * 100)
    print()

    # Load data
    print("Loading pipeline investigation data...")
    df_a = load_branch_summaries('a')
    df_b = load_branch_summaries('b')

    if df_a is None or df_b is None:
        print("[ERROR] Could not load branch summaries")
        return 1

    print(f"Type A: {len(df_a)} branches from {len(df_a['project'].unique())} projects")
    print(f"Type B: {len(df_b)} branches from {len(df_b['project'].unique())} projects")
    print()

    # Define output paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations/comparisons'

    # Create primary problem comparison
    print("Creating primary problem distribution comparison...")
    output_file_1 = output_dir / 'pipeline_problems_a_vs_b.png'
    problem_stats = create_primary_problem_comparison(df_a, df_b, output_file_1)
    print()

    # Create cancellation rate comparison
    print("Creating cancellation rate comparison per issue...")
    output_file_2 = output_dir / 'cancellation_rate_per_issue_a_vs_b.png'
    cancellation_stats = create_cancellation_rate_comparison(df_a, df_b, output_file_2)
    print()

    # Create statistics table
    print("Creating statistics table...")
    stats_file = output_dir / 'pipeline_investigation_statistics.csv'
    create_statistics_table(problem_stats, cancellation_stats, stats_file)
    print()

    # Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    if cancellation_stats:
        print(f"Overall Cancellation Rate:")
        print(f"  Type A: {cancellation_stats['overall_avg_a']:.1f}% (median: {cancellation_stats['overall_median_a']:.1f}%)")
        print(f"  Type B: {cancellation_stats['overall_avg_b']:.1f}% (median: {cancellation_stats['overall_median_b']:.1f}%)")
        print(f"  Difference: {abs(cancellation_stats['overall_avg_a'] - cancellation_stats['overall_avg_b']):.1f}%")
        print()

        if cancellation_stats['overall_avg_a'] > cancellation_stats['overall_avg_b']:
            print(f"[>] Type A has {cancellation_stats['overall_avg_a'] - cancellation_stats['overall_avg_b']:.1f}% higher cancellation rate")
        else:
            print(f"[>] Type B has {cancellation_stats['overall_avg_b'] - cancellation_stats['overall_avg_a']:.1f}% higher cancellation rate")

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
