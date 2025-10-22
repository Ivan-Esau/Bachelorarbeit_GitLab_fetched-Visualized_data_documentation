"""
Compare Merge Success: Type A vs Type B

Creates visualization comparing merge request success rates and quality between Type A and Type B projects.

Data Source:
    - visualizations/a/summary/quality_analysis/merge_quality_statistics.csv (per-project aggregates)
    - visualizations/a/summary/branch_metrics/branch_metrics_all.csv (per-branch details)
    - visualizations/b/summary/quality_analysis/merge_quality_statistics.csv (per-project aggregates)
    - visualizations/b/summary/branch_metrics/branch_metrics_all.csv (per-branch details)

Output:
    - visualizations/comparisons/merge_success_a_vs_b.png: Multi-panel comparison
    - visualizations/comparisons/merge_success_a_vs_b_statistics.csv: Detailed statistics

Usage:
    python compare_merge_success_a_vs_b.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")


def load_merge_data(csv_file):
    """
    Load merge quality data from CSV

    Args:
        csv_file: Path to merge_quality_statistics.csv

    Returns:
        DataFrame with merge quality data (excluding TOTAL row)
    """
    df = pd.read_csv(csv_file)
    # Exclude TOTAL row
    df = df[df['Project'] != 'TOTAL']
    return df


def load_branch_metrics(csv_file):
    """
    Load branch-level metrics to count open MRs

    Args:
        csv_file: Path to branch_metrics_all.csv

    Returns:
        DataFrame with branch-level data including mr_state
    """
    df = pd.read_csv(csv_file)
    return df


def count_open_mrs(df_branches):
    """
    Count open merge requests from branch metrics

    Args:
        df_branches: DataFrame with branch metrics

    Returns:
        Count of branches with mr_state='opened'
    """
    return len(df_branches[df_branches['mr_state'] == 'opened'])


def create_comparison_visualization(df_a, df_b, branches_a, branches_b, output_file):
    """
    Create multi-panel comparison visualization

    Args:
        df_a: DataFrame with Type A merge quality stats
        df_b: DataFrame with Type B merge quality stats
        branches_a: DataFrame with Type A branch metrics (for open MRs)
        branches_b: DataFrame with Type B branch metrics (for open MRs)
        output_file: Path to save the PNG file
    """
    # Create figure with 1 subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # ============================================================================
    # Merge Activity Distribution (Stacked Percentage Bars)
    # ============================================================================

    branch_success_a = df_a['Branch Success Rate (%)'].values
    branch_success_b = df_b['Branch Success Rate (%)'].values

    # Calculate totals
    total_branches_a = df_a['Total Branches'].sum()
    total_branches_b = df_b['Total Branches'].sum()

    valid_merges_a = df_a['Valid Merges'].sum()
    valid_merges_b = df_b['Valid Merges'].sum()

    invalid_merges_a = df_a['Invalid Merges'].sum()
    invalid_merges_b = df_b['Invalid Merges'].sum()

    # Count open MRs from branch metrics
    open_mrs_a = count_open_mrs(branches_a)
    open_mrs_b = count_open_mrs(branches_b)

    total_merges_a = df_a['Total Merges'].sum()
    total_merges_b = df_b['Total Merges'].sum()

    # No MR = branches without merge requests
    no_mr_a = total_branches_a - total_merges_a - open_mrs_a
    no_mr_b = total_branches_b - total_merges_b - open_mrs_b

    # Calculate percentages
    valid_pct_a = valid_merges_a / total_branches_a * 100
    invalid_pct_a = invalid_merges_a / total_branches_a * 100
    open_pct_a = open_mrs_a / total_branches_a * 100
    no_mr_pct_a = no_mr_a / total_branches_a * 100

    valid_pct_b = valid_merges_b / total_branches_b * 100
    invalid_pct_b = invalid_merges_b / total_branches_b * 100
    open_pct_b = open_mrs_b / total_branches_b * 100
    no_mr_pct_b = no_mr_b / total_branches_b * 100

    # Create vertical stacked bars
    categories = ['Type A\n(n=50)', 'Type B\n(n=49)']
    x_pos = np.arange(len(categories))

    # Define colors for each category
    colors = {
        'valid': '#27AE60',      # Green
        'invalid': '#E74C3C',    # Red
        'open': '#F39C12',       # Orange
        'no_mr': '#95A5A6'       # Gray
    }

    # Create stacked bars (bottom to top)
    bar_width = 0.6

    # Valid merges (green) - bottom
    bars_valid = ax.bar(x_pos, [valid_pct_a, valid_pct_b], bar_width,
                         label='Valid Merges', color=colors['valid'], alpha=0.8,
                         edgecolor='#34495E', linewidth=1.5)

    # Invalid merges (red) - stacked on valid
    bars_invalid = ax.bar(x_pos, [invalid_pct_a, invalid_pct_b], bar_width,
                          bottom=[valid_pct_a, valid_pct_b],
                          label='Invalid Merges', color=colors['invalid'], alpha=0.8,
                          edgecolor='#34495E', linewidth=1.5)

    # Open MRs (orange) - stacked on valid + invalid
    bars_open = ax.bar(x_pos, [open_pct_a, open_pct_b], bar_width,
                       bottom=[valid_pct_a + invalid_pct_a, valid_pct_b + invalid_pct_b],
                       label='Open MRs', color=colors['open'], alpha=0.8,
                       edgecolor='#34495E', linewidth=1.5)

    # No MR (gray) - top
    bars_no_mr = ax.bar(x_pos, [no_mr_pct_a, no_mr_pct_b], bar_width,
                        bottom=[valid_pct_a + invalid_pct_a + open_pct_a,
                                valid_pct_b + invalid_pct_b + open_pct_b],
                        label='No MR', color=colors['no_mr'], alpha=0.8,
                        edgecolor='#34495E', linewidth=1.5)

    ax.set_ylabel('Percentage of Branches (%)', fontsize=11, fontweight='bold')
    ax.set_title('Branch Activity Distribution\n(Percentage of all feature branches)',
                  fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#34495E')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add percentage labels on bars (only if segment is large enough)
    def add_percentage_labels(bars, values, bottom_values):
        for bar, value, bottom in zip(bars, values, bottom_values):
            if value > 5:  # Only show label if segment is > 5%
                height = bar.get_height()
                y_pos = bottom + height / 2
                x_pos = bar.get_x() + bar.get_width() / 2
                ax.text(x_pos, y_pos, f'{value:.1f}%',
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white' if value > 15 else '#34495E')

    add_percentage_labels(bars_valid, [valid_pct_a, valid_pct_b], [0, 0])
    add_percentage_labels(bars_invalid, [invalid_pct_a, invalid_pct_b],
                         [valid_pct_a, valid_pct_b])
    add_percentage_labels(bars_open, [open_pct_a, open_pct_b],
                         [valid_pct_a + invalid_pct_a, valid_pct_b + invalid_pct_b])
    add_percentage_labels(bars_no_mr, [no_mr_pct_a, no_mr_pct_b],
                         [valid_pct_a + invalid_pct_a + open_pct_a,
                          valid_pct_b + invalid_pct_b + open_pct_b])

    # Overall title
    fig.suptitle('Merge Request Success Comparison: Type A vs Type B',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Comparison visualization saved: {output_file}")

    return {
        'type_a': {
            'total_branches': total_branches_a,
            'total_merges': total_merges_a,
            'valid_merges': valid_merges_a,
            'invalid_merges': invalid_merges_a,
            'open_mrs': open_mrs_a,
            'no_mr': no_mr_a,
            'branch_success_mean': np.mean(branch_success_a),
            'branch_success_median': np.median(branch_success_a)
        },
        'type_b': {
            'total_branches': total_branches_b,
            'total_merges': total_merges_b,
            'valid_merges': valid_merges_b,
            'invalid_merges': invalid_merges_b,
            'open_mrs': open_mrs_b,
            'no_mr': no_mr_b,
            'branch_success_mean': np.mean(branch_success_b),
            'branch_success_median': np.median(branch_success_b)
        }
    }


def create_statistics_table(df_a, df_b, summary_stats, output_file):
    """
    Create detailed statistics CSV table

    Args:
        df_a: DataFrame with Type A merge data
        df_b: DataFrame with Type B merge data
        summary_stats: Dict with summary statistics
        output_file: Path to save the CSV file
    """
    rows = []

    # Add per-project data for Type A
    for _, row in df_a.iterrows():
        rows.append({
            'Type': 'A',
            'Project': row['Project'],
            'Total Branches': row['Total Branches'],
            'Total Merges': row['Total Merges'],
            'Valid Merges': row['Valid Merges'],
            'Invalid Merges': row['Invalid Merges'],
            'No Merge': row['Total Branches'] - row['Total Merges'],
            'Success Rate (%)': row['Success Rate (%)'],
            'Branch Success Rate (%)': row['Branch Success Rate (%)']
        })

    # Add per-project data for Type B
    for _, row in df_b.iterrows():
        rows.append({
            'Type': 'B',
            'Project': row['Project'],
            'Total Branches': row['Total Branches'],
            'Total Merges': row['Total Merges'],
            'Valid Merges': row['Valid Merges'],
            'Invalid Merges': row['Invalid Merges'],
            'No Merge': row['Total Branches'] - row['Total Merges'],
            'Success Rate (%)': row['Success Rate (%)'],
            'Branch Success Rate (%)': row['Branch Success Rate (%)']
        })

    df = pd.DataFrame(rows)

    # Add summary rows
    summary_rows = [
        {'Type': '', 'Project': '', 'Total Branches': '', 'Total Merges': '',
         'Valid Merges': '', 'Invalid Merges': '', 'Open MRs': '', 'No MR': '',
         'Success Rate (%)': '', 'Branch Success Rate (%)': ''},
        {'Type': 'A', 'Project': 'TOTAL',
         'Total Branches': summary_stats['type_a']['total_branches'],
         'Total Merges': summary_stats['type_a']['total_merges'],
         'Valid Merges': summary_stats['type_a']['valid_merges'],
         'Invalid Merges': summary_stats['type_a']['invalid_merges'],
         'Open MRs': summary_stats['type_a']['open_mrs'],
         'No MR': summary_stats['type_a']['no_mr'],
         'Success Rate (%)': summary_stats['type_a']['valid_merges'] / summary_stats['type_a']['total_merges'] * 100 if summary_stats['type_a']['total_merges'] > 0 else 0,
         'Branch Success Rate (%)': summary_stats['type_a']['branch_success_mean']},
        {'Type': 'B', 'Project': 'TOTAL',
         'Total Branches': summary_stats['type_b']['total_branches'],
         'Total Merges': summary_stats['type_b']['total_merges'],
         'Valid Merges': summary_stats['type_b']['valid_merges'],
         'Invalid Merges': summary_stats['type_b']['invalid_merges'],
         'Open MRs': summary_stats['type_b']['open_mrs'],
         'No MR': summary_stats['type_b']['no_mr'],
         'Success Rate (%)': summary_stats['type_b']['valid_merges'] / summary_stats['type_b']['total_merges'] * 100 if summary_stats['type_b']['total_merges'] > 0 else 0,
         'Branch Success Rate (%)': summary_stats['type_b']['branch_success_mean']}
    ]

    df_summary = pd.DataFrame(summary_rows)
    df = pd.concat([df, df_summary], ignore_index=True)

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"[OK] Statistics table saved: {output_file}")


def main():
    """Compare merge success between Type A and Type B"""

    print("=" * 100)
    print("MERGE SUCCESS COMPARISON: TYPE A vs TYPE B")
    print("=" * 100)
    print()

    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_a_file = base_dir / 'visualizations/a/summary/quality_analysis/merge_quality_statistics.csv'
    data_b_file = base_dir / 'visualizations/b/summary/quality_analysis/merge_quality_statistics.csv'
    branches_a_file = base_dir / 'visualizations/a/summary/branch_metrics/branch_metrics_all.csv'
    branches_b_file = base_dir / 'visualizations/b/summary/branch_metrics/branch_metrics_all.csv'

    # Check if files exist
    if not data_a_file.exists():
        print(f"[ERROR] Type A data not found: {data_a_file}")
        return 1
    if not data_b_file.exists():
        print(f"[ERROR] Type B data not found: {data_b_file}")
        return 1
    if not branches_a_file.exists():
        print(f"[ERROR] Type A branch metrics not found: {branches_a_file}")
        return 1
    if not branches_b_file.exists():
        print(f"[ERROR] Type B branch metrics not found: {branches_b_file}")
        return 1

    print("Loading data...")
    print(f"  Type A merge stats: {data_a_file}")
    print(f"  Type A branch metrics: {branches_a_file}")
    print(f"  Type B merge stats: {data_b_file}")
    print(f"  Type B branch metrics: {branches_b_file}")
    print()

    # Load data
    df_a = load_merge_data(data_a_file)
    df_b = load_merge_data(data_b_file)
    branches_a = load_branch_metrics(branches_a_file)
    branches_b = load_branch_metrics(branches_b_file)

    print(f"Type A: {len(df_a)} projects")
    print(f"Type B: {len(df_b)} projects")
    print()

    # Create output directory
    output_dir = base_dir / 'visualizations/comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    print("Creating comparison visualization...")
    output_file = output_dir / 'merge_success_a_vs_b.png'
    summary_stats = create_comparison_visualization(df_a, df_b, branches_a, branches_b, output_file)
    print()

    # Create statistics table
    print("Creating statistics table...")
    stats_file = output_dir / 'merge_success_a_vs_b_statistics.csv'
    create_statistics_table(df_a, df_b, summary_stats, stats_file)
    print()

    # Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()

    stats_a = summary_stats['type_a']
    stats_b = summary_stats['type_b']

    print(f"Type A:")
    print(f"  Total Branches: {stats_a['total_branches']}")
    print(f"  Total Merges: {stats_a['total_merges']} ({stats_a['total_merges']/stats_a['total_branches']*100:.1f}%)")
    print(f"    - Valid Merges: {stats_a['valid_merges']} ({stats_a['valid_merges']/stats_a['total_merges']*100 if stats_a['total_merges'] > 0 else 0:.1f}% of merges)")
    print(f"    - Invalid Merges: {stats_a['invalid_merges']}")
    print(f"  Open MRs: {stats_a['open_mrs']} ({stats_a['open_mrs']/stats_a['total_branches']*100:.1f}%)")
    print(f"  No MR: {stats_a['no_mr']} ({stats_a['no_mr']/stats_a['total_branches']*100:.1f}%)")
    print(f"  Branch Success Rate: {stats_a['branch_success_mean']:.1f}% (mean), {stats_a['branch_success_median']:.1f}% (median)")
    print()

    print(f"Type B:")
    print(f"  Total Branches: {stats_b['total_branches']}")
    print(f"  Total Merges: {stats_b['total_merges']} ({stats_b['total_merges']/stats_b['total_branches']*100:.1f}%)")
    print(f"    - Valid Merges: {stats_b['valid_merges']} ({stats_b['valid_merges']/stats_b['total_merges']*100 if stats_b['total_merges'] > 0 else 0:.1f}% of merges)")
    print(f"    - Invalid Merges: {stats_b['invalid_merges']}")
    print(f"  Open MRs: {stats_b['open_mrs']} ({stats_b['open_mrs']/stats_b['total_branches']*100:.1f}%)")
    print(f"  No MR: {stats_b['no_mr']} ({stats_b['no_mr']/stats_b['total_branches']*100:.1f}%)")
    print(f"  Branch Success Rate: {stats_b['branch_success_mean']:.1f}% (mean), {stats_b['branch_success_median']:.1f}% (median)")
    print()

    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
