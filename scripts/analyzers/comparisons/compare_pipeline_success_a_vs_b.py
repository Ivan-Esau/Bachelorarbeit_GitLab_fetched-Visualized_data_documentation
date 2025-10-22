"""
Compare Pipeline Success: Type A vs Type B

Creates visualization comparing pipeline success rates between Type A and Type B projects.

Data Source:
    - visualizations/a/summary/pipelines/pipeline_success_summary.csv (per-project aggregates)
    - visualizations/b/summary/pipelines/pipeline_success_summary.csv (per-project aggregates)

Output:
    - visualizations/comparisons/pipeline_success_a_vs_b.png: Comparison visualization
    - visualizations/comparisons/pipeline_success_a_vs_b_statistics.csv: Detailed statistics

Usage:
    python compare_pipeline_success_a_vs_b.py
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


def load_pipeline_data(csv_file):
    """
    Load pipeline success data from CSV

    Args:
        csv_file: Path to pipeline_success_summary.csv

    Returns:
        DataFrame with pipeline success data
    """
    df = pd.read_csv(csv_file)
    return df


def create_comparison_visualization(df_a, df_b, output_file):
    """
    Create comparison visualization

    Args:
        df_a: DataFrame with Type A pipeline stats
        df_b: DataFrame with Type B pipeline stats
        output_file: Path to save the PNG file
    """
    # Create figure with 1 subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # ============================================================================
    # Pipeline Success Distribution (Stacked Percentage Bars)
    # ============================================================================

    # Calculate totals by summing pipelines across all projects
    total_pipelines_a = df_a['total_pipelines'].sum()
    total_pipelines_b = df_b['total_pipelines'].sum()

    # Calculate weighted counts
    both_success_a = (df_a['both_success_pct'] * df_a['total_pipelines'] / 100).sum()
    build_only_a = (df_a['build_only_pct'] * df_a['total_pipelines'] / 100).sum()
    build_failed_a = (df_a['build_failed_pct'] * df_a['total_pipelines'] / 100).sum()
    canceled_a = (df_a['canceled_pct'] * df_a['total_pipelines'] / 100).sum()

    both_success_b = (df_b['both_success_pct'] * df_b['total_pipelines'] / 100).sum()
    build_only_b = (df_b['build_only_pct'] * df_b['total_pipelines'] / 100).sum()
    build_failed_b = (df_b['build_failed_pct'] * df_b['total_pipelines'] / 100).sum()
    canceled_b = (df_b['canceled_pct'] * df_b['total_pipelines'] / 100).sum()

    # Calculate percentages
    both_success_pct_a = both_success_a / total_pipelines_a * 100
    build_only_pct_a = build_only_a / total_pipelines_a * 100
    build_failed_pct_a = build_failed_a / total_pipelines_a * 100
    canceled_pct_a = canceled_a / total_pipelines_a * 100

    both_success_pct_b = both_success_b / total_pipelines_b * 100
    build_only_pct_b = build_only_b / total_pipelines_b * 100
    build_failed_pct_b = build_failed_b / total_pipelines_b * 100
    canceled_pct_b = canceled_b / total_pipelines_b * 100

    # Create vertical stacked bars
    categories = [f'Type A\n(n={int(total_pipelines_a)})', f'Type B\n(n={int(total_pipelines_b)})']
    x_pos = np.arange(len(categories))

    # Define colors for each category
    colors = {
        'both_success': '#27AE60',      # Green - full success
        'build_only': '#F39C12',        # Orange - partial success
        'build_failed': '#E74C3C',      # Red - failure
        'canceled': '#95A5A6'           # Gray - canceled/skipped
    }

    # Create stacked bars (bottom to top)
    bar_width = 0.6

    # Both success (green) - bottom
    bars_both = ax.bar(x_pos, [both_success_pct_a, both_success_pct_b], bar_width,
                       label='Both Success (Compile + Test)', color=colors['both_success'], alpha=0.8,
                       edgecolor='#34495E', linewidth=1.5)

    # Build only (orange) - stacked on both success
    bars_build = ax.bar(x_pos, [build_only_pct_a, build_only_pct_b], bar_width,
                        bottom=[both_success_pct_a, both_success_pct_b],
                        label='Build Only (Test Failed)', color=colors['build_only'], alpha=0.8,
                        edgecolor='#34495E', linewidth=1.5)

    # Build failed (red) - stacked on both + build_only
    bars_failed = ax.bar(x_pos, [build_failed_pct_a, build_failed_pct_b], bar_width,
                         bottom=[both_success_pct_a + build_only_pct_a,
                                 both_success_pct_b + build_only_pct_b],
                         label='Build Failed', color=colors['build_failed'], alpha=0.8,
                         edgecolor='#34495E', linewidth=1.5)

    # Canceled (gray) - top
    bars_canceled = ax.bar(x_pos, [canceled_pct_a, canceled_pct_b], bar_width,
                           bottom=[both_success_pct_a + build_only_pct_a + build_failed_pct_a,
                                   both_success_pct_b + build_only_pct_b + build_failed_pct_b],
                           label='Canceled/Skipped', color=colors['canceled'], alpha=0.8,
                           edgecolor='#34495E', linewidth=1.5)

    ax.set_ylabel('Percentage of Pipelines (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=1, fontsize=9, framealpha=0.9, edgecolor='#34495E')
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

    add_percentage_labels(bars_both, [both_success_pct_a, both_success_pct_b], [0, 0])
    add_percentage_labels(bars_build, [build_only_pct_a, build_only_pct_b],
                         [both_success_pct_a, both_success_pct_b])
    add_percentage_labels(bars_failed, [build_failed_pct_a, build_failed_pct_b],
                         [both_success_pct_a + build_only_pct_a,
                          both_success_pct_b + build_only_pct_b])
    add_percentage_labels(bars_canceled, [canceled_pct_a, canceled_pct_b],
                         [both_success_pct_a + build_only_pct_a + build_failed_pct_a,
                          both_success_pct_b + build_only_pct_b + build_failed_pct_b])

    # Overall title
    fig.suptitle('Pipeline Success Comparison: Type A vs Type B',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Comparison visualization saved: {output_file}")

    return {
        'type_a': {
            'total_pipelines': int(total_pipelines_a),
            'both_success': int(both_success_a),
            'build_only': int(build_only_a),
            'build_failed': int(build_failed_a),
            'canceled': int(canceled_a),
            'both_success_pct': both_success_pct_a,
            'build_only_pct': build_only_pct_a,
            'build_failed_pct': build_failed_pct_a,
            'canceled_pct': canceled_pct_a,
            'test_success_mean': df_a['test_success_rate'].mean(),
            'test_success_median': df_a['test_success_rate'].median()
        },
        'type_b': {
            'total_pipelines': int(total_pipelines_b),
            'both_success': int(both_success_b),
            'build_only': int(build_only_b),
            'build_failed': int(build_failed_b),
            'canceled': int(canceled_b),
            'both_success_pct': both_success_pct_b,
            'build_only_pct': build_only_pct_b,
            'build_failed_pct': build_failed_pct_b,
            'canceled_pct': canceled_pct_b,
            'test_success_mean': df_b['test_success_rate'].mean(),
            'test_success_median': df_b['test_success_rate'].median()
        }
    }


def create_statistics_table(df_a, df_b, summary_stats, output_file):
    """
    Create detailed statistics CSV table

    Args:
        df_a: DataFrame with Type A pipeline data
        df_b: DataFrame with Type B pipeline data
        summary_stats: Dict with summary statistics
        output_file: Path to save the CSV file
    """
    rows = []

    # Add per-project data for Type A
    for _, row in df_a.iterrows():
        rows.append({
            'Type': 'A',
            'Project': row['project'],
            'Total Pipelines': row['total_pipelines'],
            'Both Success (%)': row['both_success_pct'],
            'Build Only (%)': row['build_only_pct'],
            'Build Failed (%)': row['build_failed_pct'],
            'Canceled (%)': row['canceled_pct'],
            'Test Success Rate (%)': row['test_success_rate']
        })

    # Add per-project data for Type B
    for _, row in df_b.iterrows():
        rows.append({
            'Type': 'B',
            'Project': row['project'],
            'Total Pipelines': row['total_pipelines'],
            'Both Success (%)': row['both_success_pct'],
            'Build Only (%)': row['build_only_pct'],
            'Build Failed (%)': row['build_failed_pct'],
            'Canceled (%)': row['canceled_pct'],
            'Test Success Rate (%)': row['test_success_rate']
        })

    df = pd.DataFrame(rows)

    # Add summary rows
    summary_rows = [
        {'Type': '', 'Project': '', 'Total Pipelines': '', 'Both Success (%)': '',
         'Build Only (%)': '', 'Build Failed (%)': '', 'Canceled (%)': '',
         'Test Success Rate (%)': ''},
        {'Type': 'A', 'Project': 'TOTAL',
         'Total Pipelines': summary_stats['type_a']['total_pipelines'],
         'Both Success (%)': summary_stats['type_a']['both_success_pct'],
         'Build Only (%)': summary_stats['type_a']['build_only_pct'],
         'Build Failed (%)': summary_stats['type_a']['build_failed_pct'],
         'Canceled (%)': summary_stats['type_a']['canceled_pct'],
         'Test Success Rate (%)': summary_stats['type_a']['test_success_mean']},
        {'Type': 'B', 'Project': 'TOTAL',
         'Total Pipelines': summary_stats['type_b']['total_pipelines'],
         'Both Success (%)': summary_stats['type_b']['both_success_pct'],
         'Build Only (%)': summary_stats['type_b']['build_only_pct'],
         'Build Failed (%)': summary_stats['type_b']['build_failed_pct'],
         'Canceled (%)': summary_stats['type_b']['canceled_pct'],
         'Test Success Rate (%)': summary_stats['type_b']['test_success_mean']}
    ]

    df_summary = pd.DataFrame(summary_rows)
    df = pd.concat([df, df_summary], ignore_index=True)

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"[OK] Statistics table saved: {output_file}")


def main():
    """Compare pipeline success between Type A and Type B"""

    print("=" * 100)
    print("PIPELINE SUCCESS COMPARISON: TYPE A vs TYPE B")
    print("=" * 100)
    print()

    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_a_file = base_dir / 'visualizations/a/summary/pipelines/pipeline_success_summary.csv'
    data_b_file = base_dir / 'visualizations/b/summary/pipelines/pipeline_success_summary.csv'

    # Check if files exist
    if not data_a_file.exists():
        print(f"[ERROR] Type A data not found: {data_a_file}")
        return 1
    if not data_b_file.exists():
        print(f"[ERROR] Type B data not found: {data_b_file}")
        return 1

    print("Loading data...")
    print(f"  Type A pipeline stats: {data_a_file}")
    print(f"  Type B pipeline stats: {data_b_file}")
    print()

    # Load data
    df_a = load_pipeline_data(data_a_file)
    df_b = load_pipeline_data(data_b_file)

    print(f"Type A: {len(df_a)} projects, {df_a['total_pipelines'].sum()} pipelines")
    print(f"Type B: {len(df_b)} projects, {df_b['total_pipelines'].sum()} pipelines")
    print()

    # Create output directory
    output_dir = base_dir / 'visualizations/comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    print("Creating comparison visualization...")
    output_file = output_dir / 'pipeline_success_a_vs_b.png'
    summary_stats = create_comparison_visualization(df_a, df_b, output_file)
    print()

    # Create statistics table
    print("Creating statistics table...")
    stats_file = output_dir / 'pipeline_success_a_vs_b_statistics.csv'
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
    print(f"  Total Pipelines: {stats_a['total_pipelines']}")
    print(f"  Both Success (Compile + Test): {stats_a['both_success']} ({stats_a['both_success_pct']:.1f}%)")
    print(f"  Build Only (Test Failed): {stats_a['build_only']} ({stats_a['build_only_pct']:.1f}%)")
    print(f"  Build Failed: {stats_a['build_failed']} ({stats_a['build_failed_pct']:.1f}%)")
    print(f"  Canceled/Skipped: {stats_a['canceled']} ({stats_a['canceled_pct']:.1f}%)")
    print(f"  Test Success Rate: {stats_a['test_success_mean']:.1f}% (mean), {stats_a['test_success_median']:.1f}% (median)")
    print()

    print(f"Type B:")
    print(f"  Total Pipelines: {stats_b['total_pipelines']}")
    print(f"  Both Success (Compile + Test): {stats_b['both_success']} ({stats_b['both_success_pct']:.1f}%)")
    print(f"  Build Only (Test Failed): {stats_b['build_only']} ({stats_b['build_only_pct']:.1f}%)")
    print(f"  Build Failed: {stats_b['build_failed']} ({stats_b['build_failed_pct']:.1f}%)")
    print(f"  Canceled/Skipped: {stats_b['canceled']} ({stats_b['canceled_pct']:.1f}%)")
    print(f"  Test Success Rate: {stats_b['test_success_mean']:.1f}% (mean), {stats_b['test_success_median']:.1f}% (median)")
    print()

    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
