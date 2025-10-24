"""
Compare Infrastructure Correlation: Type A vs Type B

Creates overlay visualization comparing infrastructure correlation between Type A and Type B projects.
Shows both project types on a single plot with neutral colors.

Data Source:
    - visualizations/a/summary/quality_analysis/runner_correlation_statistics.csv
    - visualizations/b/summary/quality_analysis/runner_correlation_statistics.csv

Output:
    - visualizations/comparisons/duration_correlation_a_vs_b.png: Overlay scatter plot
    - visualizations/comparisons/duration_correlation_statistics.csv: Comparison statistics

Usage:
    python compare_infrastructure_correlation_a_vs_b.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")


def load_correlation_data(csv_file):
    """
    Load infrastructure correlation data from CSV

    Args:
        csv_file: Path to runner_correlation_statistics.csv

    Returns:
        DataFrame with correlation data
    """
    df = pd.read_csv(csv_file)
    return df


def create_overlay_correlation_plot(df_a, df_b, output_file):
    """
    Create overlay scatter plot with both Type A and Type B

    Args:
        df_a: DataFrame with Type A correlation stats
        df_b: DataFrame with Type B correlation stats
        output_file: Path to save the PNG file
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Set title
    fig.suptitle('Infrastructure Duration vs Project Success - Type A vs Type B Comparison\n'
                 'Sample Size: n=10 Type A + n=10 Type B projects',
                 fontsize=14, fontweight='bold', y=0.96)

    # Define neutral colors for each type
    color_a = '#5A7A9B'  # Blue for Type A
    color_b = '#E67E22'  # Orange for Type B

    # Plot Type A projects (circles)
    ax.scatter(df_a['avg_duration'], df_a['Branch Success Rate (%)'],
               c=color_a, s=250, alpha=0.7, edgecolors='black', linewidth=1.5,
               marker='o', label='Type A Projects', zorder=3)

    # Plot Type B projects (triangles)
    ax.scatter(df_b['avg_duration'], df_b['Branch Success Rate (%)'],
               c=color_b, s=250, alpha=0.7, edgecolors='black', linewidth=1.5,
               marker='^', label='Type B Projects', zorder=3)

    # Calculate trendlines for each type
    # Type A trendline
    z_a = np.polyfit(df_a['avg_duration'], df_a['Branch Success Rate (%)'], 1)
    p_a = np.poly1d(z_a)
    x_trend_a = np.linspace(df_a['avg_duration'].min(), df_a['avg_duration'].max(), 100)
    ax.plot(x_trend_a, p_a(x_trend_a), color=color_a, linestyle='--',
            linewidth=2.5, alpha=0.8, zorder=2, label=f'Type A Trendline')

    # Type B trendline
    z_b = np.polyfit(df_b['avg_duration'], df_b['Branch Success Rate (%)'], 1)
    p_b = np.poly1d(z_b)
    x_trend_b = np.linspace(df_b['avg_duration'].min(), df_b['avg_duration'].max(), 100)
    ax.plot(x_trend_b, p_b(x_trend_b), color=color_b, linestyle='--',
            linewidth=2.5, alpha=0.8, zorder=2, label=f'Type B Trendline')

    # Calculate correlations
    corr_a, _ = stats.pearsonr(df_a['avg_duration'], df_a['Branch Success Rate (%)'])
    r_squared_a = corr_a ** 2

    corr_b, _ = stats.pearsonr(df_b['avg_duration'], df_b['Branch Success Rate (%)'])
    r_squared_b = corr_b ** 2

    # Add correlation info to legend
    correlation_text = (f'Type A: r={corr_a:.3f}, R²={r_squared_a:.3f}\n'
                       f'Type B: r={corr_b:.3f}, R²={r_squared_b:.3f}')

    # Labels and title
    ax.set_xlabel('Average Pipeline Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Branch Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Duration vs Success\n{correlation_text}',
                  fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Annotate Type A projects
    for _, row in df_a.iterrows():
        ax.annotate(row['Project'], (row['avg_duration'], row['Branch Success Rate (%)']),
                    xytext=(6, 6), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.8, color=color_a)

    # Annotate Type B projects
    for _, row in df_b.iterrows():
        ax.annotate(row['Project'], (row['avg_duration'], row['Branch Success Rate (%)']),
                    xytext=(6, 6), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.8, color=color_b)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Overlay scatter plot saved: {output_file}")

    return {
        'corr_a': corr_a,
        'r_squared_a': r_squared_a,
        'corr_b': corr_b,
        'r_squared_b': r_squared_b
    }


def create_comparison_statistics(df_a, df_b, output_file, corr_stats):
    """
    Create comparison statistics CSV

    Args:
        df_a: DataFrame with Type A data
        df_b: DataFrame with Type B data
        output_file: Path to save the CSV file
        corr_stats: Dictionary with correlation statistics
    """
    stats_data = {
        'Metric': [
            'Number of Projects',
            'Avg Duration (seconds)',
            'Median Duration (seconds)',
            'Avg Cancellation Rate (%)',
            'Median Cancellation Rate (%)',
            'Avg Branch Success Rate (%)',
            'Median Branch Success Rate (%)',
            'Correlation (Duration vs Success)',
            'R² (Duration vs Success)',
        ],
        'Type A': [
            len(df_a),
            df_a['avg_duration'].mean(),
            df_a['avg_duration'].median(),
            df_a['canceled_rate'].mean(),
            df_a['canceled_rate'].median(),
            df_a['Branch Success Rate (%)'].mean(),
            df_a['Branch Success Rate (%)'].median(),
            corr_stats['corr_a'],
            corr_stats['r_squared_a'],
        ],
        'Type B': [
            len(df_b),
            df_b['avg_duration'].mean(),
            df_b['avg_duration'].median(),
            df_b['canceled_rate'].mean(),
            df_b['canceled_rate'].median(),
            df_b['Branch Success Rate (%)'].mean(),
            df_b['Branch Success Rate (%)'].median(),
            corr_stats['corr_b'],
            corr_stats['r_squared_b'],
        ]
    }

    stats_df = pd.DataFrame(stats_data)

    # Calculate difference (A - B for consistency with other comparisons)
    stats_df['Difference (A - B)'] = stats_df['Type A'] - stats_df['Type B']

    # Save to CSV
    stats_df.to_csv(output_file, index=False, float_format='%.3f')

    print(f"[OK] Comparison statistics saved: {output_file}")

    # Print summary
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"\nType A - Infrastructure Duration vs Success:")
    print(f"  Correlation: {corr_stats['corr_a']:.3f}")
    print(f"  R²: {corr_stats['r_squared_a']:.3f}")
    print(f"  Avg Duration: {df_a['avg_duration'].mean():.1f}s")
    print(f"  Avg Cancellation Rate: {df_a['canceled_rate'].mean():.1f}%")
    print(f"  Avg Branch Success Rate: {df_a['Branch Success Rate (%)'].mean():.1f}%")

    print(f"\nType B - Infrastructure Duration vs Success:")
    print(f"  Correlation: {corr_stats['corr_b']:.3f}")
    print(f"  R²: {corr_stats['r_squared_b']:.3f}")
    print(f"  Avg Duration: {df_b['avg_duration'].mean():.1f}s")
    print(f"  Avg Cancellation Rate: {df_b['canceled_rate'].mean():.1f}%")
    print(f"  Avg Branch Success Rate: {df_b['Branch Success Rate (%)'].mean():.1f}%")

    print(f"\nKey Findings:")

    # Compare correlations
    if abs(corr_stats['corr_a']) > abs(corr_stats['corr_b']):
        print(f"  - Type A shows stronger correlation ({corr_stats['corr_a']:.3f} vs {corr_stats['corr_b']:.3f})")
    elif abs(corr_stats['corr_b']) > abs(corr_stats['corr_a']):
        print(f"  - Type B shows stronger correlation ({corr_stats['corr_b']:.3f} vs {corr_stats['corr_a']:.3f})")
    else:
        print(f"  - Both types show similar correlation strength")

    # Compare infrastructure
    duration_diff = df_b['avg_duration'].mean() - df_a['avg_duration'].mean()
    if duration_diff > 0:
        print(f"  - Type B has {duration_diff:.1f}s higher average duration")
    else:
        print(f"  - Type A has {-duration_diff:.1f}s higher average duration")

    cancel_diff = df_b['canceled_rate'].mean() - df_a['canceled_rate'].mean()
    if cancel_diff > 0:
        print(f"  - Type B has {cancel_diff:.1f}% higher cancellation rate")
    else:
        print(f"  - Type A has {-cancel_diff:.1f}% higher cancellation rate")

    success_diff = df_b['Branch Success Rate (%)'].mean() - df_a['Branch Success Rate (%)'].mean()
    if success_diff > 0:
        print(f"  - Type B has {success_diff:.1f}% higher branch success rate")
    else:
        print(f"  - Type A has {-success_diff:.1f}% higher branch success rate")


def main():
    """Main execution function"""

    print("="*100)
    print("INFRASTRUCTURE CORRELATION COMPARISON: TYPE A vs TYPE B")
    print("="*100)

    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent

    # Input files
    csv_a = base_dir / 'visualizations' / 'a' / 'summary' / 'quality_analysis' / 'runner_correlation_statistics.csv'
    csv_b = base_dir / 'visualizations' / 'b' / 'summary' / 'quality_analysis' / 'runner_correlation_statistics.csv'

    # Check if input files exist
    if not csv_a.exists():
        print(f"ERROR: Type A file not found: {csv_a}")
        print("Please run analyze_runner_correlation.py for Type A projects first.")
        return

    if not csv_b.exists():
        print(f"ERROR: Type B file not found: {csv_b}")
        print("Please run analyze_runner_correlation.py for Type B projects first.")
        return

    # Output directory
    output_dir = base_dir / 'visualizations' / 'comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    plot_file = output_dir / 'duration_correlation_a_vs_b.png'
    stats_file = output_dir / 'duration_correlation_statistics.csv'

    # Load data
    print("\nLoading data...")
    df_a = load_correlation_data(csv_a)
    df_b = load_correlation_data(csv_b)

    print(f"  Type A: {len(df_a)} projects")
    print(f"  Type B: {len(df_b)} projects")

    # Create visualization
    print("\nCreating overlay correlation plot...")
    corr_stats = create_overlay_correlation_plot(df_a, df_b, plot_file)

    # Create statistics
    print("\nGenerating comparison statistics...")
    create_comparison_statistics(df_a, df_b, stats_file, corr_stats)

    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)
    print(f"\nOutput files:")
    print(f"  - Visualization: {plot_file}")
    print(f"  - Statistics: {stats_file}")
    print()


if __name__ == '__main__':
    main()
