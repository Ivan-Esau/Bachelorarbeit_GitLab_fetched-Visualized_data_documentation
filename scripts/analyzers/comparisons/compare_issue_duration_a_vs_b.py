"""
Compare Issue Duration Between Type A and Type B Projects

This script compares the development duration of each issue/branch across all projects.
Creates box plots showing the distribution of durations for each issue, side by side for Type A and Type B.

Data Sources:
- visualizations/a/summary/branch_lifecycle/branch_lifecycle_durations.csv
- visualizations/b/summary/branch_lifecycle/branch_lifecycle_durations.csv

Output:
- issue_duration_a_vs_b.png: Box plot comparison of issue durations
- issue_duration_a_vs_b_statistics.csv: Comprehensive statistics table
- issue_duration_a_vs_b_comprehensive_statistics.csv: Detailed min/max/avg/median/std

Metrics:
- Development duration in minutes per issue
- Box plots show: min, Q1, median, Q3, max, outliers
- Statistics: count, min, max, mean, median, std dev
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style to match other comparison visualizations
sns.set_style("whitegrid")

def load_duration_data(project_type):
    """Load branch lifecycle duration data for a project type."""
    csv_path = Path(f'visualizations/{project_type}/summary/branch_lifecycle/branch_lifecycle_durations.csv')

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df['type'] = project_type.upper()
    return df


def extract_issue_number(issue_str):
    """Extract issue number from issue string like 'Issue #1'."""
    if pd.isna(issue_str):
        return None
    try:
        return int(issue_str.replace('Issue #', '').strip())
    except:
        return None


def create_issue_duration_comparison():
    """Create comprehensive issue duration comparison between Type A and Type B."""

    print("\n" + "="*80)
    print("COMPARING ISSUE DURATION: TYPE A vs TYPE B")
    print("="*80 + "\n")

    # Load data for both types
    print("Loading data...")
    df_a = load_duration_data('a')
    df_b = load_duration_data('b')

    if df_a.empty or df_b.empty:
        print("Error: Missing data files")
        return

    # Combine data
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Extract issue numbers
    df['issue_number'] = df['issue'].apply(extract_issue_number)

    # Filter out any rows without issue numbers
    df = df[df['issue_number'].notna()].copy()

    # Convert duration from seconds to minutes
    df['development_duration_minutes'] = df['development_duration_seconds'] / 60

    print(f"Loaded {len(df_a)} Type A branches and {len(df_b)} Type B branches")
    print(f"Issues found: {sorted(df['issue_number'].unique())}")

    # Create output directory
    output_dir = Path('visualizations/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. CREATE BOX PLOT VISUALIZATION
    # ========================================
    print("\nCreating box plot visualization...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plot
    issues = sorted(df['issue_number'].unique())
    positions_a = np.arange(len(issues)) * 2 - 0.35
    positions_b = np.arange(len(issues)) * 2 + 0.35

    # Collect data for each issue
    data_a = []
    data_b = []

    for issue in issues:
        durations_a = df[(df['type'] == 'A') & (df['issue_number'] == issue)]['development_duration_minutes'].values
        durations_b = df[(df['type'] == 'B') & (df['issue_number'] == issue)]['development_duration_minutes'].values
        data_a.append(durations_a)
        data_b.append(durations_b)

    # Create box plots with consistent style
    bp_a = ax.boxplot(data_a, positions=positions_a, widths=0.6,
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
                      boxprops=dict(facecolor='#5A7A9B', alpha=0.7, linewidth=1.5),
                      medianprops=dict(color='#34495E', linewidth=2.5),
                      whiskerprops=dict(color='#34495E', linewidth=1.5),
                      capprops=dict(color='#34495E', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='none', markersize=5,
                                    markeredgecolor='black', markeredgewidth=1))

    bp_b = ax.boxplot(data_b, positions=positions_b, widths=0.6,
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
                      boxprops=dict(facecolor='#E67E22', alpha=0.7, linewidth=1.5),
                      medianprops=dict(color='#34495E', linewidth=2.5),
                      whiskerprops=dict(color='#34495E', linewidth=1.5),
                      capprops=dict(color='#34495E', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='none', markersize=5,
                                    markeredgecolor='black', markeredgewidth=1))

    # Customize plot
    ax.set_xlabel('Issue Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Development Duration (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Issue Development Duration Comparison: Type A vs Type B\n' +
                 'Box Plot Distribution per Issue',
                 fontsize=13, fontweight='bold', pad=15)

    # Set x-axis labels
    ax.set_xticks(np.arange(len(issues)) * 2)
    ax.set_xticklabels([f'Issue #{i}' for i in issues], fontsize=11, fontweight='bold')

    # Set y-axis ticks at 50-minute intervals for more detailed scale
    max_duration = max([max(data) for data in data_a + data_b if len(data) > 0])
    y_max = int(np.ceil(max_duration / 50) * 50) + 50  # Round up to next 50 and add buffer
    y_ticks = np.arange(0, y_max + 1, 50)
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, y_max)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add statistical summary text box
    overall_a_median = df[df['type'] == 'A']['development_duration_minutes'].median()
    overall_b_median = df[df['type'] == 'B']['development_duration_minutes'].median()
    overall_a_mean = df[df['type'] == 'A']['development_duration_minutes'].mean()
    overall_b_mean = df[df['type'] == 'B']['development_duration_minutes'].mean()
    overall_a_std = df[df['type'] == 'A']['development_duration_minutes'].std()
    overall_b_std = df[df['type'] == 'B']['development_duration_minutes'].std()
    count_a = len(df[df['type'] == 'A'])
    count_b = len(df[df['type'] == 'B'])

    stats_text = (
        f'Type A (n={count_a}):\n'
        f'  Mean: {overall_a_mean:.0f}m | Median: {overall_a_median:.0f}m\n'
        f'  Range: {df[df["type"] == "A"]["development_duration_minutes"].min():.0f}-{df[df["type"] == "A"]["development_duration_minutes"].max():.0f}m\n'
        f'\n'
        f'Type B (n={count_b}):\n'
        f'  Mean: {overall_b_mean:.0f}m | Median: {overall_b_median:.0f}m\n'
        f'  Range: {df[df["type"] == "B"]["development_duration_minutes"].min():.0f}-{df[df["type"] == "B"]["development_duration_minutes"].max():.0f}m'
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#34495E', linewidth=1.5)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=props)

    # Add legend in top left corner
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#5A7A9B', alpha=0.7, edgecolor='#34495E', linewidth=1.5, label='Type A'),
        Patch(facecolor='#E67E22', alpha=0.7, edgecolor='#34495E', linewidth=1.5, label='Type B'),
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markersize=6, markeredgecolor='red', linestyle='None', label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=5, markeredgecolor='black', markeredgewidth=1, linestyle='None', label='Outliers')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.9, edgecolor='#34495E', title='Legend', title_fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'issue_duration_a_vs_b.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Box plot saved: {output_path}")

    # ========================================
    # 2. CREATE STATISTICS CSV
    # ========================================
    print("\nGenerating statistics...")

    stats_rows = []

    for issue in issues:
        df_a_issue = df[(df['type'] == 'A') & (df['issue_number'] == issue)]
        df_b_issue = df[(df['type'] == 'B') & (df['issue_number'] == issue)]

        durations_a = df_a_issue['development_duration_minutes'].values
        durations_b = df_b_issue['development_duration_minutes'].values

        stats_rows.append({
            'Issue': f'Issue #{issue}',
            'Type A Count': len(durations_a),
            'Type A Mean (minutes)': np.mean(durations_a) if len(durations_a) > 0 else np.nan,
            'Type A Median (minutes)': np.median(durations_a) if len(durations_a) > 0 else np.nan,
            'Type A Min (minutes)': np.min(durations_a) if len(durations_a) > 0 else np.nan,
            'Type A Max (minutes)': np.max(durations_a) if len(durations_a) > 0 else np.nan,
            'Type A Std (minutes)': np.std(durations_a, ddof=1) if len(durations_a) > 1 else np.nan,
            'Type B Count': len(durations_b),
            'Type B Mean (minutes)': np.mean(durations_b) if len(durations_b) > 0 else np.nan,
            'Type B Median (minutes)': np.median(durations_b) if len(durations_b) > 0 else np.nan,
            'Type B Min (minutes)': np.min(durations_b) if len(durations_b) > 0 else np.nan,
            'Type B Max (minutes)': np.max(durations_b) if len(durations_b) > 0 else np.nan,
            'Type B Std (minutes)': np.std(durations_b, ddof=1) if len(durations_b) > 1 else np.nan,
            'Difference Mean (A-B)': np.mean(durations_a) - np.mean(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Difference Median (A-B)': np.median(durations_a) - np.median(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan
        })

    # Add overall statistics
    durations_a_all = df[df['type'] == 'A']['development_duration_minutes'].values
    durations_b_all = df[df['type'] == 'B']['development_duration_minutes'].values

    stats_rows.append({
        'Issue': 'OVERALL',
        'Type A Count': len(durations_a_all),
        'Type A Mean (minutes)': np.mean(durations_a_all),
        'Type A Median (minutes)': np.median(durations_a_all),
        'Type A Min (minutes)': np.min(durations_a_all),
        'Type A Max (minutes)': np.max(durations_a_all),
        'Type A Std (minutes)': np.std(durations_a_all, ddof=1),
        'Type B Count': len(durations_b_all),
        'Type B Mean (minutes)': np.mean(durations_b_all),
        'Type B Median (minutes)': np.median(durations_b_all),
        'Type B Min (minutes)': np.min(durations_b_all),
        'Type B Max (minutes)': np.max(durations_b_all),
        'Type B Std (minutes)': np.std(durations_b_all, ddof=1),
        'Difference Mean (A-B)': np.mean(durations_a_all) - np.mean(durations_b_all),
        'Difference Median (A-B)': np.median(durations_a_all) - np.median(durations_b_all)
    })

    stats_df = pd.DataFrame(stats_rows)

    # Save statistics
    stats_path = output_dir / 'issue_duration_a_vs_b_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] Statistics saved: {stats_path}")

    # ========================================
    # 3. CREATE COMPREHENSIVE STATISTICS TABLE
    # ========================================
    print("\nGenerating comprehensive statistics table...")

    comprehensive_rows = []

    # Per-issue statistics
    for issue in issues:
        durations_a = df[(df['type'] == 'A') & (df['issue_number'] == issue)]['development_duration_minutes'].values
        durations_b = df[(df['type'] == 'B') & (df['issue_number'] == issue)]['development_duration_minutes'].values

        # Type A
        comprehensive_rows.append({
            'Metric': f'Issue #{issue}',
            'Type': 'A',
            'Count': len(durations_a),
            'Min (minutes)': np.min(durations_a) if len(durations_a) > 0 else np.nan,
            'Max (minutes)': np.max(durations_a) if len(durations_a) > 0 else np.nan,
            'Mean (minutes)': np.mean(durations_a) if len(durations_a) > 0 else np.nan,
            'Median (minutes)': np.median(durations_a) if len(durations_a) > 0 else np.nan,
            'Std (minutes)': np.std(durations_a, ddof=1) if len(durations_a) > 1 else np.nan,
            'Q1 (minutes)': np.percentile(durations_a, 25) if len(durations_a) > 0 else np.nan,
            'Q3 (minutes)': np.percentile(durations_a, 75) if len(durations_a) > 0 else np.nan
        })

        # Type B
        comprehensive_rows.append({
            'Metric': f'Issue #{issue}',
            'Type': 'B',
            'Count': len(durations_b),
            'Min (minutes)': np.min(durations_b) if len(durations_b) > 0 else np.nan,
            'Max (minutes)': np.max(durations_b) if len(durations_b) > 0 else np.nan,
            'Mean (minutes)': np.mean(durations_b) if len(durations_b) > 0 else np.nan,
            'Median (minutes)': np.median(durations_b) if len(durations_b) > 0 else np.nan,
            'Std (minutes)': np.std(durations_b, ddof=1) if len(durations_b) > 1 else np.nan,
            'Q1 (minutes)': np.percentile(durations_b, 25) if len(durations_b) > 0 else np.nan,
            'Q3 (minutes)': np.percentile(durations_b, 75) if len(durations_b) > 0 else np.nan
        })

        # Difference
        comprehensive_rows.append({
            'Metric': f'Issue #{issue}',
            'Type': 'Difference (A - B)',
            'Count': '',
            'Min (minutes)': np.min(durations_a) - np.min(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Max (minutes)': np.max(durations_a) - np.max(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Mean (minutes)': np.mean(durations_a) - np.mean(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Median (minutes)': np.median(durations_a) - np.median(durations_b) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Std (minutes)': '',
            'Q1 (minutes)': np.percentile(durations_a, 25) - np.percentile(durations_b, 25) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan,
            'Q3 (minutes)': np.percentile(durations_a, 75) - np.percentile(durations_b, 75) if len(durations_a) > 0 and len(durations_b) > 0 else np.nan
        })

        # Add separator
        comprehensive_rows.append({
            'Metric': '', 'Type': '', 'Count': '', 'Min (minutes)': '', 'Max (minutes)': '',
            'Mean (minutes)': '', 'Median (minutes)': '', 'Std (minutes)': '', 'Q1 (minutes)': '', 'Q3 (minutes)': ''
        })

    # Overall statistics
    comprehensive_rows.append({
        'Metric': 'ALL ISSUES',
        'Type': 'A',
        'Count': len(durations_a_all),
        'Min (minutes)': np.min(durations_a_all),
        'Max (minutes)': np.max(durations_a_all),
        'Mean (minutes)': np.mean(durations_a_all),
        'Median (minutes)': np.median(durations_a_all),
        'Std (minutes)': np.std(durations_a_all, ddof=1),
        'Q1 (minutes)': np.percentile(durations_a_all, 25),
        'Q3 (minutes)': np.percentile(durations_a_all, 75)
    })

    comprehensive_rows.append({
        'Metric': 'ALL ISSUES',
        'Type': 'B',
        'Count': len(durations_b_all),
        'Min (minutes)': np.min(durations_b_all),
        'Max (minutes)': np.max(durations_b_all),
        'Mean (minutes)': np.mean(durations_b_all),
        'Median (minutes)': np.median(durations_b_all),
        'Std (minutes)': np.std(durations_b_all, ddof=1),
        'Q1 (minutes)': np.percentile(durations_b_all, 25),
        'Q3 (minutes)': np.percentile(durations_b_all, 75)
    })

    comprehensive_rows.append({
        'Metric': 'ALL ISSUES',
        'Type': 'Difference (A - B)',
        'Count': '',
        'Min (minutes)': np.min(durations_a_all) - np.min(durations_b_all),
        'Max (minutes)': np.max(durations_a_all) - np.max(durations_b_all),
        'Mean (minutes)': np.mean(durations_a_all) - np.mean(durations_b_all),
        'Median (minutes)': np.median(durations_a_all) - np.median(durations_b_all),
        'Std (minutes)': '',
        'Q1 (minutes)': np.percentile(durations_a_all, 25) - np.percentile(durations_b_all, 25),
        'Q3 (minutes)': np.percentile(durations_a_all, 75) - np.percentile(durations_b_all, 75)
    })

    comprehensive_df = pd.DataFrame(comprehensive_rows)

    # Save comprehensive statistics
    comprehensive_path = output_dir / 'issue_duration_a_vs_b_comprehensive_statistics.csv'
    comprehensive_df.to_csv(comprehensive_path, index=False)
    print(f"[OK] Comprehensive statistics saved: {comprehensive_path}")

    # ========================================
    # 4. PRINT SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal branches analyzed:")
    print(f"  Type A: {len(durations_a_all)} branches")
    print(f"  Type B: {len(durations_b_all)} branches")

    print(f"\nOverall Duration Statistics (minutes):")
    print(f"  Type A: Mean={overall_a_mean:.0f}, Median={overall_a_median:.0f}, Std={np.std(durations_a_all, ddof=1):.0f}")
    print(f"  Type B: Mean={overall_b_mean:.0f}, Median={overall_b_median:.0f}, Std={np.std(durations_b_all, ddof=1):.0f}")
    print(f"  Difference (A-B): Mean={overall_a_mean - overall_b_mean:.0f}, Median={overall_a_median - overall_b_median:.0f}")

    print(f"\nPer-Issue Summary:")
    for issue in issues:
        durations_a = df[(df['type'] == 'A') & (df['issue_number'] == issue)]['development_duration_minutes'].values
        durations_b = df[(df['type'] == 'B') & (df['issue_number'] == issue)]['development_duration_minutes'].values
        print(f"  Issue #{issue}:")
        print(f"    Type A: Mean={np.mean(durations_a):.0f}m, Median={np.median(durations_a):.0f}m (n={len(durations_a)})")
        print(f"    Type B: Mean={np.mean(durations_b):.0f}m, Median={np.median(durations_b):.0f}m (n={len(durations_b)})")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    create_issue_duration_comparison()
