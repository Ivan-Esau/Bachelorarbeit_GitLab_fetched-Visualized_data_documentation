"""
Pipeline Job Success Summary - All Projects Comparison

Creates a summary visualization comparing pipeline job success rates across all projects.
Uses equal-weight aggregation: each pipeline weighted equally (scientifically correct).

Data Source:
- pipelines.json from each project

Output:
- visualizations/summary/pipelines/pipeline_success_summary.png
- visualizations/summary/pipelines/pipeline_success_summary.csv

Metrics:
- Build + Test Success Rate
- Build-Only Success Rate
- Build Failed Rate
- Canceled/Skipped Rate

Aggregation Method:
- Equal-weight: sums all pipelines, then calculates percentages
- Each pipeline run has equal weight (not averaged by branch)

Usage:
    python visualize_pipeline_success_summary.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
from core import load_project_config
from core.path_helpers import get_data_dir
from core.config_loader import get_project_letter, get_project_number

# Set style
sns.set_style("whitegrid")


def get_project_pipeline_success(project_name, data_base_dir=None):
    """
    Calculate pipeline success rates using equal-weight aggregation (scientifically correct)

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data (deprecated, uses path_helpers)

    Returns:
        Dict with aggregated pipeline statistics (each pipeline weighted equally)
    """
    # Use path helper for letter-based structure
    data_dir = get_data_dir(project_name)
    pipelines_file = data_dir / 'pipelines.json'

    if not os.path.exists(pipelines_file):
        return None

    with open(pipelines_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    # Group pipelines by branch and calculate per-branch statistics
    branch_stats = defaultdict(lambda: {
        'total_pipelines': 0,
        'both_success': 0,
        'build_only': 0,
        'build_failed': 0,
        'canceled': 0
    })

    total_pipelines_count = 0

    for pipeline in pipelines:
        ref = pipeline.get('ref', 'unknown')

        # Skip master/main branch
        if ref.lower() in ['master', 'main']:
            continue

        total_pipelines_count += 1
        jobs = pipeline.get('jobs', [])
        stats = branch_stats[ref]
        stats['total_pipelines'] += 1

        # Check if pipeline has no jobs
        if not jobs:
            stats['canceled'] += 1
            continue

        # Find build and test jobs by STAGE (most reliable)
        compile_job = None
        test_job = None

        for job in jobs:
            stage = job.get('stage', '')
            if stage == 'compile':
                compile_job = job
            elif stage == 'test':
                test_job = job

        # Get job statuses
        compile_status = compile_job.get('status') if compile_job else None
        test_status = test_job.get('status') if test_job else None

        # Check if compile job was canceled/skipped or doesn't exist
        if compile_status in ['canceled', 'skipped', 'manual'] or compile_status is None:
            stats['canceled'] += 1
            continue

        # Categorize based on actual job outcomes
        if compile_status == 'success' and test_status == 'success':
            stats['both_success'] += 1
        elif compile_status == 'success':
            stats['build_only'] += 1  # Test failed, skipped, or canceled
        else:
            stats['build_failed'] += 1

    # Sum all counts across branches (equal-weight aggregation)
    if not branch_stats:
        return None

    total_branches = len(branch_stats)

    # Sum all counts
    total_both_success = sum(stats['both_success'] for stats in branch_stats.values())
    total_build_only = sum(stats['build_only'] for stats in branch_stats.values())
    total_build_failed = sum(stats['build_failed'] for stats in branch_stats.values())
    total_canceled = sum(stats['canceled'] for stats in branch_stats.values())

    # Calculate percentages from totals (each pipeline weighted equally)
    if total_pipelines_count > 0:
        both_success_pct = (total_both_success / total_pipelines_count) * 100
        build_only_pct = (total_build_only / total_pipelines_count) * 100
        build_failed_pct = (total_build_failed / total_pipelines_count) * 100
        canceled_pct = (total_canceled / total_pipelines_count) * 100
    else:
        both_success_pct = build_only_pct = build_failed_pct = canceled_pct = 0.0

    return {
        'total_pipelines': total_pipelines_count,
        'total_branches': total_branches,
        'both_success_pct': both_success_pct,
        'build_only_pct': build_only_pct,
        'build_failed_pct': build_failed_pct,
        'canceled_pct': canceled_pct,
        'total_success_rate': both_success_pct + build_only_pct,
        'test_success_rate': both_success_pct
    }


def create_summary_visualization(project_stats, output_file):
    """
    Create grouped bar chart comparing all projects

    Args:
        project_stats: DataFrame with project statistics
        output_file: Path to save the PNG file
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                     gridspec_kw={'height_ratios': [2, 1]})

    # ===== Top Chart: 100% Stacked Bar Chart =====
    projects = project_stats['project'].tolist()
    x = np.arange(len(projects))
    width = 0.7

    # Extract percentages
    both_success = project_stats['both_success_pct'].tolist()
    build_only = project_stats['build_only_pct'].tolist()
    build_failed = project_stats['build_failed_pct'].tolist()
    canceled = project_stats['canceled_pct'].tolist()

    # Create stacked bars
    bars1 = ax1.bar(x, both_success, width,
                    label='Build + Test erfolgreich',
                    color='#27AE60', alpha=0.85, edgecolor='#1E8449', linewidth=1.5)

    bars2 = ax1.bar(x, build_only, width, bottom=both_success,
                    label='Nur Build erfolgreich',
                    color='#F39C12', alpha=0.85, edgecolor='#D68910', linewidth=1.5)

    bottom_failed = [b + o for b, o in zip(both_success, build_only)]
    bars3 = ax1.bar(x, build_failed, width, bottom=bottom_failed,
                    label='Build fehlgeschlagen',
                    color='#95A5A6', alpha=0.6, edgecolor='#7F8C8D', linewidth=1.5)

    bottom_canceled = [b + o + f for b, o, f in zip(both_success, build_only, build_failed)]
    bars4 = ax1.bar(x, canceled, width, bottom=bottom_canceled,
                    label='Abgebrochen/übersprungen',
                    color='#566573', alpha=0.5, edgecolor='#34495E', linewidth=1.5)

    # Add percentage labels on segments
    for i in range(len(projects)):
        # Both success
        if both_success[i] > 5:
            ax1.text(i, both_success[i]/2, f'{both_success[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')

        # Build only
        if build_only[i] > 5:
            ax1.text(i, both_success[i] + build_only[i]/2, f'{build_only[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')

        # Failed
        if build_failed[i] > 5:
            ax1.text(i, bottom_failed[i] + build_failed[i]/2, f'{build_failed[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', fontsize=9)

        # Canceled
        if canceled[i] > 5:
            ax1.text(i, bottom_canceled[i] + canceled[i]/2, f'{canceled[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', fontsize=9, color='white')

    ax1.set_xlabel('Projekt', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Anteil (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Pipeline-Erfolgsraten im Projektvergleich\n' +
                  'Aggregiert über alle Pipelines pro Projekt (ohne Master)\n' +
                  'Aggregation Method: Equal-weight (each pipeline weighted equally)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(projects, fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Bottom Chart: Pipeline Counts =====
    pipeline_counts = project_stats['total_pipelines'].tolist()

    bars = ax2.bar(x, pipeline_counts, width,
                   color='#3498DB', alpha=0.8, edgecolor='#2980B9', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, pipeline_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_xlabel('Projekt', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Anzahl Pipelines', fontsize=13, fontweight='bold')
    ax2.set_title('Gesamtanzahl der Pipelines pro Projekt',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(projects, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Summary visualization saved: {output_file}")


def main():
    """Main function to create pipeline success summary"""

    print("=" * 100)
    print("PIPELINE SUCCESS SUMMARY - ALL PROJECTS")
    print("=" * 100)
    print()

    # Load projects
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

    # Collect statistics from all projects
    all_stats = []

    for project_id, project_name in projects:
        # Extract project label (e.g., "A01", "B05")
        letter = get_project_letter(project_name)
        number = get_project_number(project_name)
        project_label = f"{letter.upper()}{number}"

        print(f"Processing {project_label}...", end=' ')

        stats = get_project_pipeline_success(project_name)

        if stats:
            stats['project'] = project_label
            all_stats.append(stats)
            print(f"[OK] {stats['total_pipelines']} pipelines")
        else:
            print("[SKIP] No pipeline data")

    if not all_stats:
        print("\n[ERROR] No pipeline data found")
        return 1

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Sort by project name
    df = df.sort_values('project')

    print()
    print("=" * 100)
    print("CREATING SUMMARY VISUALIZATION")
    print("=" * 100)
    print()

    # Create output directory (letter-based structure)
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / 'visualizations' / project_type / 'summary' / 'pipelines'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_file = output_dir / 'pipeline_success_summary.csv'
    df.to_csv(csv_file, index=False, float_format='%.2f')
    print(f"[OK] Statistics CSV saved: {csv_file}")

    # Create visualization
    viz_file = output_dir / 'pipeline_success_summary.png'
    create_summary_visualization(df, viz_file)

    # Print summary statistics
    print()
    print("=" * 100)
    print("SUMMARY STATISTICS (EQUAL-WEIGHT AGGREGATION)")
    print("=" * 100)
    print()

    total_pipelines = df['total_pipelines'].sum()
    total_branches = df['total_branches'].sum()
    avg_success_rate = df['total_success_rate'].mean()
    avg_test_success_rate = df['test_success_rate'].mean()
    avg_canceled_rate = df['canceled_pct'].mean()

    print(f"Total pipelines analyzed: {total_pipelines}")
    print(f"Total branches analyzed: {total_branches}")
    print(f"Average build success rate (equal-weight): {avg_success_rate:.1f}%")
    print(f"Average full success rate (Build+Test): {avg_test_success_rate:.1f}%")
    print(f"Average canceled/skipped rate: {avg_canceled_rate:.1f}%")
    print()

    print("Per-project rates (each pipeline weighted equally):")
    for _, row in df.iterrows():
        print(f"  {row['project']}: {row['total_success_rate']:.1f}% build, "
              f"{row['test_success_rate']:.1f}% full, "
              f"{row['canceled_pct']:.1f}% canceled "
              f"({row['total_branches']:.0f} branches, {row['total_pipelines']:.0f} pipelines)")

    print()
    print("=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
