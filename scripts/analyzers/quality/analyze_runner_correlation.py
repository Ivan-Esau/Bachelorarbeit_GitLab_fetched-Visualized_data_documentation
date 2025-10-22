"""
Pipeline Infrastructure and Success Correlation Analysis

Investigates correlations between:
- Project success (merge quality, solved issue rate)
- Pipeline durations (infrastructure performance)
- Canceled/skipped pipelines (runner problems)

Research Questions:
1. Do longer pipeline durations correlate with lower success?
2. Do canceled/skipped pipelines indicate runner problems?
3. Did some projects suffer more from infrastructure issues?
4. How did infrastructure problems impact project outcomes?

Output:
    - runner_correlation_statistics.csv: Per-project metrics
    - Console report with correlation analysis

Usage:
    python analyze_runner_correlation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from core.path_helpers import get_data_dir
from core.config_loader import get_project_letter, get_project_number


def parse_datetime(dt_string):
    """Parse GitLab datetime string"""
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
    except:
        return None


def analyze_pipeline_infrastructure(project_name, data_base_dir=None):
    """
    Analyze pipeline infrastructure metrics for a project

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data (deprecated, uses path_helpers)

    Returns:
        Dict with infrastructure metrics
    """
    # Use path helper for letter-based structure
    project_dir = get_data_dir(project_name)

    # Load data
    with open(project_dir / 'pipelines.json', encoding='utf-8') as f:
        pipelines = json.load(f)

    # Initialize counters
    total_pipelines = len(pipelines)
    durations = []

    status_counts = defaultdict(int)
    job_status_counts = defaultdict(int)

    canceled_pipelines = 0
    skipped_pipelines = 0
    failed_pipelines = 0
    success_pipelines = 0

    compile_durations = []
    test_durations = []

    for pipeline in pipelines:
        # Pipeline status
        status = pipeline.get('status', 'unknown')
        status_counts[status] += 1

        if status == 'canceled':
            canceled_pipelines += 1
        elif status == 'skipped':
            skipped_pipelines += 1
        elif status == 'failed':
            failed_pipelines += 1
        elif status == 'success':
            success_pipelines += 1

        # Pipeline duration
        duration = pipeline.get('duration')
        if duration is not None and duration > 0:
            durations.append(duration)

        # Job-level analysis
        for job in pipeline.get('jobs', []):
            job_status = job.get('status', 'unknown')
            job_status_counts[job_status] += 1

            # Job durations by stage
            job_duration = job.get('duration')
            if job_duration is not None and job_duration > 0:
                if job['stage'] == 'compile':
                    compile_durations.append(job_duration)
                elif job['stage'] == 'test':
                    test_durations.append(job_duration)

    # Calculate statistics
    avg_duration = np.mean(durations) if durations else 0
    median_duration = np.median(durations) if durations else 0
    max_duration = np.max(durations) if durations else 0

    avg_compile_duration = np.mean(compile_durations) if compile_durations else 0
    avg_test_duration = np.mean(test_durations) if test_durations else 0

    # Calculate rates
    canceled_rate = (canceled_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
    skipped_rate = (skipped_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
    failed_rate = (failed_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
    success_rate = (success_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0

    # Job-level cancellation/skip rates
    total_jobs = sum(job_status_counts.values())
    canceled_jobs = job_status_counts.get('canceled', 0)
    skipped_jobs = job_status_counts.get('skipped', 0)

    job_canceled_rate = (canceled_jobs / total_jobs * 100) if total_jobs > 0 else 0
    job_skipped_rate = (skipped_jobs / total_jobs * 100) if total_jobs > 0 else 0

    return {
        'total_pipelines': total_pipelines,
        'success_pipelines': success_pipelines,
        'failed_pipelines': failed_pipelines,
        'canceled_pipelines': canceled_pipelines,
        'skipped_pipelines': skipped_pipelines,
        'avg_duration': avg_duration,
        'median_duration': median_duration,
        'max_duration': max_duration,
        'avg_compile_duration': avg_compile_duration,
        'avg_test_duration': avg_test_duration,
        'canceled_rate': canceled_rate,
        'skipped_rate': skipped_rate,
        'failed_rate': failed_rate,
        'success_rate': success_rate,
        'total_jobs': total_jobs,
        'canceled_jobs': canceled_jobs,
        'skipped_jobs': skipped_jobs,
        'job_canceled_rate': job_canceled_rate,
        'job_skipped_rate': job_skipped_rate,
        'pipeline_status_counts': dict(status_counts),
        'job_status_counts': dict(job_status_counts)
    }


def load_merge_quality_stats(merge_quality_file):
    """Load merge quality statistics from CSV"""
    if not merge_quality_file.exists():
        print(f"[WARNING] Merge quality file not found: {merge_quality_file}")
        return None

    df = pd.read_csv(merge_quality_file)
    # Exclude TOTAL row
    df = df[df['Project'] != 'TOTAL']
    return df


def calculate_correlations(combined_df):
    """Calculate correlations between infrastructure and success metrics"""

    # Select numeric columns for correlation
    correlation_columns = [
        'avg_duration',
        'median_duration',
        'canceled_rate',
        'skipped_rate',
        'job_canceled_rate',
        'Success Rate (%)',
        'Branch Success Rate (%)'
    ]

    # Filter to only include projects with merges
    df_with_merges = combined_df[combined_df['Total Merges'] > 0].copy()

    if len(df_with_merges) < 2:
        return None

    # Calculate correlation matrix
    corr_matrix = df_with_merges[correlation_columns].corr()

    return corr_matrix


def create_scatter_plots(df, output_file):
    """
    Option 1: Scatter plots showing correlation
    Two scatter plots side-by-side with trendlines
    """
    # Define project tiers for color coding
    def get_tier_color(row):
        if row['canceled_rate'] == 0 and row['avg_duration'] < 100:
            return '#27AE60', 'Good'  # Green
        elif row['canceled_rate'] < 30 and row['avg_duration'] < 300:
            return '#F39C12', 'Moderate'  # Orange
        else:
            return '#E74C3C', 'Poor'  # Red

    df['color'], df['tier'] = zip(*df.apply(get_tier_color, axis=1))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    n_projects = len(df)
    fig.suptitle(f'Infrastructure Problems vs Project Success - Correlation Analysis\n'
                 f'Sample Size: n={n_projects} projects | Note: Small sample size limits statistical power',
                 fontsize=14, fontweight='bold', y=0.96)

    # Plot 1: Average Duration vs Solved Issue %
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax1.scatter(tier_data['avg_duration'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    # Add trendline
    z = np.polyfit(df['avg_duration'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['avg_duration'].min(), df['avg_duration'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    # Calculate correlation
    corr1, _ = stats.pearsonr(df['avg_duration'], df['Branch Success Rate (%)'])
    r_squared1 = corr1 ** 2

    ax1.set_xlabel('Average Pipeline Duration (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Duration vs Success\nCorrelation: {corr1:.3f} | R²: {r_squared1:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Add project labels
    for _, row in df.iterrows():
        ax1.annotate(row['Project'], (row['avg_duration'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    # Plot 2: Cancellation Rate vs Solved Issue %
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax2.scatter(tier_data['canceled_rate'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    # Add trendline
    z = np.polyfit(df['canceled_rate'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['canceled_rate'].min(), df['canceled_rate'].max(), 100)
    ax2.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    # Calculate correlation
    corr2, _ = stats.pearsonr(df['canceled_rate'], df['Branch Success Rate (%)'])
    r_squared2 = corr2 ** 2

    ax2.set_xlabel('Pipeline Cancellation Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cancellation vs Success\nCorrelation: {corr2:.3f} | R²: {r_squared2:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)

    # Add project labels
    for _, row in df.iterrows():
        ax2.annotate(row['Project'], (row['canceled_rate'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Option 1 - Scatter plots saved: {output_file}")


def create_multipanel_dashboard(df, output_file):
    """
    Option 2: Multi-panel dashboard
    Four panels showing infrastructure metrics, success metrics, and correlations
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Pipeline Infrastructure vs Project Success - Comprehensive Dashboard',
                 fontsize=15, fontweight='bold', y=0.98)

    # Panel 1: Infrastructure Metrics (Duration + Cancellation)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df))
    width = 0.35

    # Normalize durations for visualization (scale to percentage)
    max_duration = df['avg_duration'].max()
    duration_pct = (df['avg_duration'] / max_duration * 100)

    bars1 = ax1.bar(x - width/2, duration_pct, width, label='Avg Duration (scaled %)',
                   color='#3498DB', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, df['canceled_rate'], width, label='Cancellation Rate (%)',
                   color='#E74C3C', alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Project', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Percentage / Scaled Value', fontsize=11, fontweight='bold')
    ax1.set_title('Infrastructure Problems by Project', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Project'], fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Success Metrics
    ax2 = fig.add_subplot(gs[1, :])
    bars3 = ax2.bar(df['Project'], df['Branch Success Rate (%)'], color='#27AE60',
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Color bars by value
    for i, (bar, val) in enumerate(zip(bars3, df['Branch Success Rate (%)'])):
        if val >= 50:
            bar.set_color('#27AE60')  # Green
        elif val > 0:
            bar.set_color('#F39C12')  # Orange
        else:
            bar.set_color('#E74C3C')  # Red

    ax2.set_xlabel('Project', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Branch Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Project Success Rate', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 3: Duration correlation scatter
    ax3 = fig.add_subplot(gs[2, 0])
    colors = ['#27AE60' if c == 0 and d < 100 else '#E74C3C'
              for c, d in zip(df['canceled_rate'], df['avg_duration'])]
    ax3.scatter(df['avg_duration'], df['Branch Success Rate (%)'], c=colors, s=150, alpha=0.7, edgecolors='black')

    # Trendline
    z = np.polyfit(df['avg_duration'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['avg_duration'].min(), df['avg_duration'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2)

    corr, _ = stats.pearsonr(df['avg_duration'], df['Branch Success Rate (%)'])
    ax3.set_xlabel('Avg Duration (s)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Branch Success Rate (%)', fontsize=10, fontweight='bold')
    ax3.set_title(f'Duration Correlation: {corr:.3f}', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Cancellation correlation scatter
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(df['canceled_rate'], df['Branch Success Rate (%)'], c=colors, s=150, alpha=0.7, edgecolors='black')

    # Trendline
    z = np.polyfit(df['canceled_rate'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['canceled_rate'].min(), df['canceled_rate'].max(), 100)
    ax4.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2)

    corr, _ = stats.pearsonr(df['canceled_rate'], df['Branch Success Rate (%)'])
    ax4.set_xlabel('Cancellation Rate (%)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Branch Success Rate (%)', fontsize=10, fontweight='bold')
    ax4.set_title(f'Cancellation Correlation: {corr:.3f}', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Option 2 - Multi-panel dashboard saved: {output_file}")


def create_colored_table(df, output_file):
    """
    Option 3: Color-coded table
    Shows all metrics with color coding for good/bad infrastructure
    """
    # Prepare data for table
    table_data = []

    for _, row in df.iterrows():
        table_data.append([
            row['Project'],
            f"{row['avg_duration']:.0f}s",
            f"{row['canceled_rate']:.0f}%",
            f"{row['job_canceled_rate']:.0f}%",
            f"{row['Total Merges']:.0f}",
            f"{row['Valid Merges']:.0f}",
            f"{row['Branch Success Rate (%)']:.0f}%"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Column headers
    columns = ['Project', 'Avg\nDuration', 'Pipeline\nCancel %', 'Job\nCancel %',
               'Total\nMerges', 'Valid\nMerges', 'Solved\nIssue %']

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.15, 0.15, 0.15, 0.12, 0.12, 0.15]
    )

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

            # Avg Duration column
            if j == 1:
                duration = df.iloc[i]['avg_duration']
                if duration < 100:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                elif duration < 500:
                    cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                else:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red

            # Cancellation columns
            elif j == 2 or j == 3:
                rate = df.iloc[i]['canceled_rate'] if j == 2 else df.iloc[i]['job_canceled_rate']
                if rate == 0:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                elif rate < 30:
                    cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                else:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red

            # Valid Merges column
            elif j == 5:
                valid = df.iloc[i]['Valid Merges']
                if valid > 0:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor((0.95, 0.95, 0.95))

            # Solved Issue % column
            elif j == 6:
                solved = df.iloc[i]['Branch Success Rate (%)']
                if solved >= 50:
                    cell.set_facecolor((0.4, 0.8, 0.4))  # Green
                elif solved > 0:
                    cell.set_facecolor((1.0, 0.9, 0.4))  # Yellow
                else:
                    cell.set_facecolor((1.0, 0.5, 0.5))  # Red

            else:
                cell.set_facecolor((0.95, 0.95, 0.95))

            cell.set_text_props(fontsize=10)

    # Calculate correlations for title
    corr_duration = df['avg_duration'].corr(df['Branch Success Rate (%)'])
    corr_cancel = df['canceled_rate'].corr(df['Branch Success Rate (%)'])

    plt.title(
        f'Infrastructure vs Success - Color-Coded Metrics Table\n'
        f'Correlations: Duration={corr_duration:.3f} | Cancellation={corr_cancel:.3f}',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Option 3 - Color-coded table saved: {output_file}")


def create_scatter_plots_only(df, output_file):
    """
    Scatter plots only from Option 4
    Two correlation scatter plots without the table
    """
    # Define colors
    def get_tier_color(row):
        if row['canceled_rate'] == 0 and row['avg_duration'] < 100:
            return '#27AE60', 'Good'
        elif row['canceled_rate'] < 30 and row['avg_duration'] < 300:
            return '#F39C12', 'Moderate'
        else:
            return '#E74C3C', 'Poor'

    df['color'], df['tier'] = zip(*df.apply(get_tier_color, axis=1))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    n_projects = len(df)
    fig.suptitle(f'Infrastructure Problems vs Project Success - Correlation Analysis\n'
                 f'Sample Size: n={n_projects} projects | Note: Small sample size limits statistical power',
                 fontsize=14, fontweight='bold', y=0.96)

    # Left: Duration scatter
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax1.scatter(tier_data['avg_duration'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    # Trendline
    z = np.polyfit(df['avg_duration'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['avg_duration'].min(), df['avg_duration'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    corr1, _ = stats.pearsonr(df['avg_duration'], df['Branch Success Rate (%)'])
    r_squared1 = corr1 ** 2

    ax1.set_xlabel('Average Pipeline Duration (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Duration vs Success\nCorrelation: {corr1:.3f} | R²: {r_squared1:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    for _, row in df.iterrows():
        ax1.annotate(row['Project'], (row['avg_duration'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    # Right: Cancellation scatter
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax2.scatter(tier_data['canceled_rate'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    z = np.polyfit(df['canceled_rate'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['canceled_rate'].min(), df['canceled_rate'].max(), 100)
    ax2.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    corr2, _ = stats.pearsonr(df['canceled_rate'], df['Branch Success Rate (%)'])
    r_squared2 = corr2 ** 2

    ax2.set_xlabel('Pipeline Cancellation Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cancellation vs Success\nCorrelation: {corr2:.3f} | R²: {r_squared2:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)

    for _, row in df.iterrows():
        ax2.annotate(row['Project'], (row['canceled_rate'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Scatter plots only saved: {output_file}")


def create_summary_table_only(df, output_file):
    """
    Summary table only from Option 4
    Table with infrastructure tier color coding
    """
    # Define colors
    def get_tier_color(row):
        if row['canceled_rate'] == 0 and row['avg_duration'] < 100:
            return '#27AE60', 'Good'
        elif row['canceled_rate'] < 30 and row['avg_duration'] < 300:
            return '#F39C12', 'Moderate'
        else:
            return '#E74C3C', 'Poor'

    df['color'], df['tier'] = zip(*df.apply(get_tier_color, axis=1))

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Project'],
            f"{row['avg_duration']:.0f}s",
            f"{row['canceled_rate']:.0f}%",
            f"{row['Valid Merges']:.0f}/{row['Total Merges']:.0f}",
            f"{row['Branch Success Rate (%)']:.0f}%",
            row['tier']
        ])

    columns = ['Project', 'Avg Duration', 'Cancellation\nRate', 'Valid/Total\nMerges',
               'Solved\nIssue %', 'Infrastructure\nTier']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.18, 0.18, 0.18, 0.15, 0.19]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Color header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5A7A9B')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Color data cells by tier
    for i in range(len(table_data)):
        tier = df.iloc[i]['tier']
        tier_color = {'Good': (0.4, 0.8, 0.4, 0.3),
                      'Moderate': (1.0, 0.9, 0.4, 0.3),
                      'Poor': (1.0, 0.5, 0.5, 0.3)}[tier]

        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(tier_color)
            cell.set_text_props(fontsize=10)

    # Calculate correlations for title
    corr_duration = df['avg_duration'].corr(df['Branch Success Rate (%)'])
    corr_cancel = df['canceled_rate'].corr(df['Branch Success Rate (%)'])

    plt.title(
        f'Infrastructure Metrics and Project Success Summary\n'
        f'Correlations: Duration={corr_duration:.3f} | Cancellation={corr_cancel:.3f}',
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Summary table only saved: {output_file}")


def create_combined_visualization(df, output_file):
    """
    Option 4: Combined scatter + table
    Top: Two scatter plots showing correlations
    Bottom: Summary table with key metrics
    """
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    fig.suptitle('Infrastructure Problems vs Project Success - Comprehensive Analysis',
                 fontsize=15, fontweight='bold', y=0.97)

    # Define colors
    def get_tier_color(row):
        if row['canceled_rate'] == 0 and row['avg_duration'] < 100:
            return '#27AE60', 'Good'
        elif row['canceled_rate'] < 30 and row['avg_duration'] < 300:
            return '#F39C12', 'Moderate'
        else:
            return '#E74C3C', 'Poor'

    df['color'], df['tier'] = zip(*df.apply(get_tier_color, axis=1))

    # Top-left: Duration scatter
    ax1 = fig.add_subplot(gs[0, 0])
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax1.scatter(tier_data['avg_duration'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    # Trendline
    z = np.polyfit(df['avg_duration'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['avg_duration'].min(), df['avg_duration'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    corr1, _ = stats.pearsonr(df['avg_duration'], df['Branch Success Rate (%)'])
    r_squared1 = corr1 ** 2

    ax1.set_xlabel('Average Pipeline Duration (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Duration vs Success\nCorrelation: {corr1:.3f} | R²: {r_squared1:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    for _, row in df.iterrows():
        ax1.annotate(row['Project'], (row['avg_duration'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    # Top-right: Cancellation scatter
    ax2 = fig.add_subplot(gs[0, 1])
    for tier, color in [('Good', '#27AE60'), ('Moderate', '#F39C12'), ('Poor', '#E74C3C')]:
        tier_data = df[df['tier'] == tier]
        ax2.scatter(tier_data['canceled_rate'], tier_data['Branch Success Rate (%)'],
                   c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                   label=f'{tier} Infrastructure', zorder=3)

    z = np.polyfit(df['canceled_rate'], df['Branch Success Rate (%)'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['canceled_rate'].min(), df['canceled_rate'].max(), 100)
    ax2.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, zorder=2)

    corr2, _ = stats.pearsonr(df['canceled_rate'], df['Branch Success Rate (%)'])
    r_squared2 = corr2 ** 2

    ax2.set_xlabel('Pipeline Cancellation Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Branch Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cancellation vs Success\nCorrelation: {corr2:.3f} | R²: {r_squared2:.3f}',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)

    for _, row in df.iterrows():
        ax2.annotate(row['Project'], (row['canceled_rate'], row['Branch Success Rate (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    fontweight='bold', alpha=0.7)

    # Bottom: Summary table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')

    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Project'],
            f"{row['avg_duration']:.0f}s",
            f"{row['canceled_rate']:.0f}%",
            f"{row['Valid Merges']:.0f}/{row['Total Merges']:.0f}",
            f"{row['Branch Success Rate (%)']:.0f}%",
            row['tier']
        ])

    columns = ['Project', 'Avg Duration', 'Cancellation\nRate', 'Valid/Total\nMerges',
               'Solved\nIssue %', 'Infrastructure\nTier']

    table = ax3.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.18, 0.18, 0.18, 0.15, 0.19]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # Color header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#5A7A9B')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # Color data cells by tier
    for i in range(len(table_data)):
        tier = df.iloc[i]['tier']
        tier_color = {'Good': (0.4, 0.8, 0.4, 0.3),
                      'Moderate': (1.0, 0.9, 0.4, 0.3),
                      'Poor': (1.0, 0.5, 0.5, 0.3)}[tier]

        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(tier_color)
            cell.set_text_props(fontsize=10)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Option 4 - Combined visualization saved: {output_file}")


def main():
    """Analyze pipeline infrastructure and correlate with project success"""

    print("=" * 100)
    print("PIPELINE INFRASTRUCTURE AND SUCCESS CORRELATION ANALYSIS")
    print("=" * 100)
    print()

    # Load projects from config
    from core import load_project_config
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"Analyzing {len(projects)} projects...\n")

    # Determine project type from first project for letter-based structure
    first_project_name = projects[0][1] if projects else None
    if not first_project_name:
        print("[ERROR] No projects found")
        return 1

    project_type = get_project_letter(first_project_name)

    # Analyze infrastructure for each project
    infrastructure_stats = []

    for project_id, project_name in projects:
        # Extract project label for display (e.g., "A01", "B05")
        letter = get_project_letter(project_name)
        number = get_project_number(project_name)
        project_label = f"{letter.upper()}{number}"

        print(f"Analyzing infrastructure for {project_label}...", end=' ')

        stats = analyze_pipeline_infrastructure(project_name)

        print(f"[OK] {stats['total_pipelines']} pipelines, "
              f"{stats['canceled_rate']:.1f}% canceled, "
              f"avg duration: {stats['avg_duration']:.1f}s")

        infrastructure_stats.append({
            'Project': project_label,
            'total_pipelines': stats['total_pipelines'],
            'success_pipelines': stats['success_pipelines'],
            'failed_pipelines': stats['failed_pipelines'],
            'canceled_pipelines': stats['canceled_pipelines'],
            'skipped_pipelines': stats['skipped_pipelines'],
            'avg_duration': stats['avg_duration'],
            'median_duration': stats['median_duration'],
            'max_duration': stats['max_duration'],
            'avg_compile_duration': stats['avg_compile_duration'],
            'avg_test_duration': stats['avg_test_duration'],
            'canceled_rate': stats['canceled_rate'],
            'skipped_rate': stats['skipped_rate'],
            'failed_rate': stats['failed_rate'],
            'pipeline_success_rate': stats['success_rate'],
            'total_jobs': stats['total_jobs'],
            'canceled_jobs': stats['canceled_jobs'],
            'skipped_jobs': stats['skipped_jobs'],
            'job_canceled_rate': stats['job_canceled_rate'],
            'job_skipped_rate': stats['job_skipped_rate']
        })

    print()

    # Create DataFrame
    infra_df = pd.DataFrame(infrastructure_stats)

    # Load merge quality statistics (letter-based structure)
    base_dir = Path(__file__).parent.parent.parent.parent
    merge_quality_file = base_dir / 'visualizations' / project_type / 'summary' / 'quality_analysis' / 'merge_quality_statistics.csv'
    merge_df = load_merge_quality_stats(merge_quality_file)

    # Combine datasets
    if merge_df is not None:
        combined_df = pd.merge(infra_df, merge_df, on='Project', how='left')
        print("[OK] Combined infrastructure and merge quality data")
    else:
        combined_df = infra_df
        print("[WARNING] Merge quality data not available, analysis will be limited")

    print()

    # Create output directory (letter-based structure)
    output_dir = base_dir / 'visualizations' / project_type / 'summary' / 'quality_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined statistics
    stats_file = output_dir / 'runner_correlation_statistics.csv'
    combined_df.to_csv(stats_file, index=False, float_format='%.2f')
    print(f"[OK] Statistics saved: {stats_file}")
    print()

    # Create visualizations
    print("=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)
    print()

    # Correlation scatter plots
    scatter_file = output_dir / 'correlation_scatter_plots.png'
    create_scatter_plots_only(combined_df.copy(), scatter_file)

    # Summary table
    table_file = output_dir / 'correlation_summary_table.png'
    create_summary_table_only(combined_df.copy(), table_file)

    print()
    print("Visualizations created!")
    print()

    # Print summary report
    print("=" * 100)
    print("INFRASTRUCTURE METRICS SUMMARY")
    print("=" * 100)
    print()

    # Print infrastructure table
    print(f"{'Project':<10} {'Pipelines':>10} {'Canceled':>10} {'Skipped':>9} "
          f"{'Cancel %':>10} {'Avg Dur(s)':>12} {'Med Dur(s)':>12}")
    print("-" * 100)

    for _, row in infra_df.iterrows():
        print(f"{row['Project']:<10} {row['total_pipelines']:>10.0f} "
              f"{row['canceled_pipelines']:>10.0f} {row['skipped_pipelines']:>9.0f} "
              f"{row['canceled_rate']:>9.1f}% {row['avg_duration']:>11.1f}s "
              f"{row['median_duration']:>11.1f}s")

    print()

    # Infrastructure problem ranking
    print("=" * 100)
    print("INFRASTRUCTURE PROBLEM RANKING")
    print("=" * 100)
    print()

    print("Projects ranked by cancellation rate:")
    sorted_by_cancel = infra_df.sort_values('canceled_rate', ascending=False)
    for idx, (_, row) in enumerate(sorted_by_cancel.iterrows(), 1):
        print(f"  {idx:2d}. {row['Project']}: {row['canceled_rate']:5.1f}% "
              f"({row['canceled_pipelines']:.0f}/{row['total_pipelines']:.0f} pipelines)")

    print()
    print("Projects ranked by average duration:")
    sorted_by_duration = infra_df.sort_values('avg_duration', ascending=False)
    for idx, (_, row) in enumerate(sorted_by_duration.iterrows(), 1):
        print(f"  {idx:2d}. {row['Project']}: {row['avg_duration']:6.1f}s avg "
              f"(median: {row['median_duration']:.1f}s)")

    print()

    # Correlation analysis
    if merge_df is not None:
        print("=" * 100)
        print("CORRELATION ANALYSIS: Infrastructure vs. Success")
        print("=" * 100)
        print()

        corr_matrix = calculate_correlations(combined_df)

        if corr_matrix is not None:
            print("Correlation with Merge Success Rate:")
            success_corr = corr_matrix['Success Rate (%)'].drop('Success Rate (%)')
            for metric, corr_value in success_corr.items():
                direction = "↑" if corr_value > 0 else "↓"
                strength = "STRONG" if abs(corr_value) > 0.7 else "MODERATE" if abs(corr_value) > 0.4 else "WEAK"
                print(f"  {metric:25s}: {corr_value:+.3f} {direction} ({strength})")

            print()
            print("Correlation with Solved Issue Rate:")
            solved_corr = corr_matrix['Branch Success Rate (%)'].drop('Branch Success Rate (%)')
            for metric, corr_value in solved_corr.items():
                direction = "↑" if corr_value > 0 else "↓"
                strength = "STRONG" if abs(corr_value) > 0.7 else "MODERATE" if abs(corr_value) > 0.4 else "WEAK"
                print(f"  {metric:25s}: {corr_value:+.3f} {direction} ({strength})")

            print()

            # Key findings
            print("KEY FINDINGS:")
            print("-" * 100)

            # Find strongest negative correlations
            success_negative = success_corr[success_corr < -0.3].sort_values()
            if len(success_negative) > 0:
                print("\nFactors negatively correlated with merge success:")
                for metric, corr in success_negative.items():
                    print(f"  - {metric}: {corr:.3f} (higher {metric} → lower success)")

            # Find strongest positive correlations
            success_positive = success_corr[success_corr > 0.3].sort_values(ascending=False)
            if len(success_positive) > 0:
                print("\nFactors positively correlated with merge success:")
                for metric, corr in success_positive.items():
                    print(f"  - {metric}: {corr:.3f} (higher {metric} → higher success)")

    print()

    # Job-level analysis
    print("=" * 100)
    print("JOB-LEVEL CANCELLATION ANALYSIS")
    print("=" * 100)
    print()

    total_jobs = infra_df['total_jobs'].sum()
    total_canceled_jobs = infra_df['canceled_jobs'].sum()
    total_skipped_jobs = infra_df['skipped_jobs'].sum()

    print(f"Total jobs across all projects: {total_jobs:.0f}")
    print(f"  Canceled jobs: {total_canceled_jobs:.0f} ({total_canceled_jobs/total_jobs*100:.1f}%)")
    print(f"  Skipped jobs: {total_skipped_jobs:.0f} ({total_skipped_jobs/total_jobs*100:.1f}%)")
    print()

    print("Projects ranked by job cancellation rate:")
    sorted_by_job_cancel = infra_df.sort_values('job_canceled_rate', ascending=False)
    for idx, (_, row) in enumerate(sorted_by_job_cancel.iterrows(), 1):
        print(f"  {idx:2d}. {row['Project']}: {row['job_canceled_rate']:5.1f}% "
              f"({row['canceled_jobs']:.0f}/{row['total_jobs']:.0f} jobs)")

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
