"""
Create comprehensive statistics tables for all comparison diagrams.
Generates detailed min, max, avg, median, std tables for each comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_merge_success_statistics_table():
    """Create comprehensive statistics table for merge success comparison."""
    print("Creating comprehensive statistics for merge_success_a_vs_b...")

    # Read the existing merge success data
    df = pd.read_csv('visualizations/comparisons/merge_success_a_vs_b_statistics.csv')

    # Filter out the TOTAL row for statistical calculations
    df_no_total = df[df['Project'] != 'TOTAL'].copy()

    # Separate Type A and Type B
    df_a = df_no_total[df_no_total['Type'] == 'A']
    df_b = df_no_total[df_no_total['Type'] == 'B']

    # Metrics to analyze
    metrics = ['Total Branches', 'Total Merges', 'Valid Merges', 'Invalid Merges',
               'Success Rate (%)', 'Branch Success Rate (%)', 'Open MRs']

    # Create statistics table
    stats_data = []

    for metric in metrics:
        if metric not in df_a.columns:
            continue

        # Type A statistics
        a_values = df_a[metric].dropna()
        a_stats = {
            'Metric': metric,
            'Type': 'A',
            'Count': len(a_values),
            'Min': a_values.min() if len(a_values) > 0 else np.nan,
            'Max': a_values.max() if len(a_values) > 0 else np.nan,
            'Mean': a_values.mean() if len(a_values) > 0 else np.nan,
            'Median': a_values.median() if len(a_values) > 0 else np.nan,
            'Std': a_values.std() if len(a_values) > 0 else np.nan,
            'Total': a_values.sum() if len(a_values) > 0 else np.nan
        }
        stats_data.append(a_stats)

        # Type B statistics
        b_values = df_b[metric].dropna()
        b_stats = {
            'Metric': metric,
            'Type': 'B',
            'Count': len(b_values),
            'Min': b_values.min() if len(b_values) > 0 else np.nan,
            'Max': b_values.max() if len(b_values) > 0 else np.nan,
            'Mean': b_values.mean() if len(b_values) > 0 else np.nan,
            'Median': b_values.median() if len(b_values) > 0 else np.nan,
            'Std': b_values.std() if len(b_values) > 0 else np.nan,
            'Total': b_values.sum() if len(b_values) > 0 else np.nan
        }
        stats_data.append(b_stats)

        # Difference
        diff_stats = {
            'Metric': metric,
            'Type': 'Difference (A - B)',
            'Count': '',
            'Min': a_values.min() - b_values.min() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Max': a_values.max() - b_values.max() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Mean': a_values.mean() - b_values.mean() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Median': a_values.median() - b_values.median() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Std': '',
            'Total': a_values.sum() - b_values.sum() if len(a_values) > 0 and len(b_values) > 0 else np.nan
        }
        stats_data.append(diff_stats)

        # Add separator
        stats_data.append({'Metric': '', 'Type': '', 'Count': '', 'Min': '', 'Max': '', 'Mean': '', 'Median': '', 'Std': '', 'Total': ''})

    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    output_path = 'visualizations/comparisons/merge_success_a_vs_b_comprehensive_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return stats_df


def create_pipeline_success_statistics_table():
    """Create comprehensive statistics table for pipeline success comparison."""
    print("Creating comprehensive statistics for pipeline_success_a_vs_b...")

    # Read the existing pipeline success data
    df = pd.read_csv('visualizations/comparisons/pipeline_success_a_vs_b_statistics.csv')

    # Filter out the TOTAL row for statistical calculations
    df_no_total = df[df['Project'] != 'TOTAL'].copy()

    # Separate Type A and Type B
    df_a = df_no_total[df_no_total['Type'] == 'A']
    df_b = df_no_total[df_no_total['Type'] == 'B']

    # Metrics to analyze
    metrics = ['Total Pipelines', 'Both Success (%)', 'Build Only (%)',
               'Build Failed (%)', 'Canceled (%)', 'Test Success Rate (%)']

    # Create statistics table
    stats_data = []

    for metric in metrics:
        if metric not in df_a.columns:
            continue

        # Type A statistics
        a_values = df_a[metric].dropna()
        a_stats = {
            'Metric': metric,
            'Type': 'A',
            'Count': len(a_values),
            'Min': a_values.min() if len(a_values) > 0 else np.nan,
            'Max': a_values.max() if len(a_values) > 0 else np.nan,
            'Mean': a_values.mean() if len(a_values) > 0 else np.nan,
            'Median': a_values.median() if len(a_values) > 0 else np.nan,
            'Std': a_values.std() if len(a_values) > 0 else np.nan,
            'Total': a_values.sum() if len(a_values) > 0 else np.nan
        }
        stats_data.append(a_stats)

        # Type B statistics
        b_values = df_b[metric].dropna()
        b_stats = {
            'Metric': metric,
            'Type': 'B',
            'Count': len(b_values),
            'Min': b_values.min() if len(b_values) > 0 else np.nan,
            'Max': b_values.max() if len(b_values) > 0 else np.nan,
            'Mean': b_values.mean() if len(b_values) > 0 else np.nan,
            'Median': b_values.median() if len(b_values) > 0 else np.nan,
            'Std': b_values.std() if len(b_values) > 0 else np.nan,
            'Total': b_values.sum() if len(b_values) > 0 else np.nan
        }
        stats_data.append(b_stats)

        # Difference
        diff_stats = {
            'Metric': metric,
            'Type': 'Difference (A - B)',
            'Count': '',
            'Min': a_values.min() - b_values.min() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Max': a_values.max() - b_values.max() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Mean': a_values.mean() - b_values.mean() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Median': a_values.median() - b_values.median() if len(a_values) > 0 and len(b_values) > 0 else np.nan,
            'Std': '',
            'Total': a_values.sum() - b_values.sum() if len(a_values) > 0 and len(b_values) > 0 else np.nan
        }
        stats_data.append(diff_stats)

        # Add separator
        stats_data.append({'Metric': '', 'Type': '', 'Count': '', 'Min': '', 'Max': '', 'Mean': '', 'Median': '', 'Std': '', 'Total': ''})

    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    output_path = 'visualizations/comparisons/pipeline_success_a_vs_b_comprehensive_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return stats_df


def create_pipeline_investigation_statistics_table():
    """Create comprehensive statistics table for pipeline investigation (cancellation rates)."""
    print("Creating comprehensive statistics for pipeline_investigation (cancellation rates)...")

    # Read the existing pipeline investigation data
    df = pd.read_csv('visualizations/comparisons/pipeline_investigation_statistics.csv')

    # Extract cancellation rate data (rows 12-17 from the original file)
    cancellation_data = df[df['Category'] == 'Issue Cancellation'].copy()

    # Parse the cancellation rate values
    # Format: "14.6% (n=10)"
    def parse_rate(rate_str):
        if pd.isna(rate_str) or rate_str == '':
            return np.nan
        try:
            return float(rate_str.split('%')[0])
        except:
            return np.nan

    # Extract Type A and Type B values
    a_rates = []
    b_rates = []
    issues = []

    for _, row in cancellation_data.iterrows():
        issue = row['Metric']
        issues.append(issue)
        a_rate = parse_rate(row['Type A'])
        b_rate = parse_rate(row['Type B'])
        a_rates.append(a_rate)
        b_rates.append(b_rate)

    # Create comprehensive statistics table
    stats_data = []

    # Type A statistics
    a_valid = [r for r in a_rates if not np.isnan(r)]
    if a_valid:
        stats_data.append({
            'Metric': 'Cancellation Rate per Issue',
            'Type': 'A',
            'Count': len(a_valid),
            'Min (%)': min(a_valid),
            'Max (%)': max(a_valid),
            'Mean (%)': np.mean(a_valid),
            'Median (%)': np.median(a_valid),
            'Std (%)': np.std(a_valid, ddof=1) if len(a_valid) > 1 else 0
        })

    # Type B statistics
    b_valid = [r for r in b_rates if not np.isnan(r)]
    if b_valid:
        stats_data.append({
            'Metric': 'Cancellation Rate per Issue',
            'Type': 'B',
            'Count': len(b_valid),
            'Min (%)': min(b_valid),
            'Max (%)': max(b_valid),
            'Mean (%)': np.mean(b_valid),
            'Median (%)': np.median(b_valid),
            'Std (%)': np.std(b_valid, ddof=1) if len(b_valid) > 1 else 0
        })

    # Difference
    if a_valid and b_valid:
        stats_data.append({
            'Metric': 'Cancellation Rate per Issue',
            'Type': 'Difference (A - B)',
            'Count': '',
            'Min (%)': min(a_valid) - min(b_valid),
            'Max (%)': max(a_valid) - max(b_valid),
            'Mean (%)': np.mean(a_valid) - np.mean(b_valid),
            'Median (%)': np.median(a_valid) - np.median(b_valid),
            'Std (%)': ''
        })

    # Add separator
    stats_data.append({'Metric': '', 'Type': '', 'Count': '', 'Min (%)': '', 'Max (%)': '', 'Mean (%)': '', 'Median (%)': '', 'Std (%)': ''})

    # Add per-issue breakdown
    for i, issue in enumerate(issues):
        if not np.isnan(a_rates[i]) and not np.isnan(b_rates[i]):
            stats_data.append({
                'Metric': issue,
                'Type': 'A',
                'Count': '',
                'Min (%)': '',
                'Max (%)': '',
                'Mean (%)': a_rates[i],
                'Median (%)': '',
                'Std (%)': ''
            })
            stats_data.append({
                'Metric': issue,
                'Type': 'B',
                'Count': '',
                'Min (%)': '',
                'Max (%)': '',
                'Mean (%)': b_rates[i],
                'Median (%)': '',
                'Std (%)': ''
            })
            stats_data.append({
                'Metric': issue,
                'Type': 'Difference (A - B)',
                'Count': '',
                'Min (%)': '',
                'Max (%)': '',
                'Mean (%)': a_rates[i] - b_rates[i],
                'Median (%)': '',
                'Std (%)': ''
            })

    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    output_path = 'visualizations/comparisons/pipeline_investigation_comprehensive_statistics.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return stats_df


def create_all_comprehensive_tables():
    """Create all comprehensive statistics tables for comparison diagrams."""
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE STATISTICS TABLES FOR ALL COMPARISON DIAGRAMS")
    print("="*80 + "\n")

    # Create output directory if it doesn't exist
    output_dir = Path('visualizations/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive tables
    tables = []

    print("\n1. MERGE SUCCESS COMPARISON")
    print("-" * 80)
    merge_table = create_merge_success_statistics_table()
    tables.append(('Merge Success', merge_table))

    print("\n2. PIPELINE SUCCESS COMPARISON")
    print("-" * 80)
    pipeline_table = create_pipeline_success_statistics_table()
    tables.append(('Pipeline Success', pipeline_table))

    print("\n3. PIPELINE INVESTIGATION (CANCELLATION RATES)")
    print("-" * 80)
    investigation_table = create_pipeline_investigation_statistics_table()
    tables.append(('Pipeline Investigation', investigation_table))

    print("\n" + "="*80)
    print("SUMMARY OF COMPREHENSIVE STATISTICS TABLES CREATED")
    print("="*80)
    print("\nThe following comprehensive statistics tables have been created:")
    print("1. merge_success_a_vs_b_comprehensive_statistics.csv")
    print("2. pipeline_success_a_vs_b_comprehensive_statistics.csv")
    print("3. pipeline_investigation_comprehensive_statistics.csv")

    print("\nNote: The following existing files already have comprehensive statistics:")
    print("- project_duration_a_vs_b_statistics.csv (has min, max, mean, median, std)")
    print("- coverage_per_branch_a_vs_b_statistics.csv (has min, max, mean, median, std)")
    print("- duration_correlation_statistics.csv (has comprehensive correlation metrics)")

    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS GENERATION COMPLETE")
    print("="*80 + "\n")

    return tables


if __name__ == '__main__':
    create_all_comprehensive_tables()
