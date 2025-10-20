"""
Visualize Pipeline Job Success Rates per Project (ALL Pipelines)

Creates one stacked bar chart per project showing job success rates for each issue/branch.
INCLUDES all pipelines (also canceled/skipped ones).

This follows scientific research standards with German labels.

Input Data Requirements:
    - pipelines.json: Pipeline execution data with jobs

Output:
    - One PNG per project: visualizations/pipeline_job_success_all/{project}_job_success_all.png
    - Statistics CSV: pipeline_job_success_all_statistics.csv

Usage:
    python visualize_pipeline_job_success_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import numpy as np
from core import load_project_config

# Set style
sns.set_style("whitegrid")


def get_job_success_rates(project_name, data_base_dir='../../data_raw'):
    """
    Calculate pipeline success rates grouped by branch/issue

    Categorizes each pipeline into:
    - Both: Build AND Test successful
    - Build only: Build successful, Test failed
    - Failed: Build failed

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data

    Returns:
        Dict mapping issue labels to pipeline statistics
    """
    data_dir = os.path.join(data_base_dir, project_name)
    pipelines_file = os.path.join(data_dir, 'pipelines.json')

    if not os.path.exists(pipelines_file):
        return {}

    with open(pipelines_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    # Group pipelines by branch
    branch_stats = defaultdict(lambda: {
        'total_pipelines': 0,
        'both_success': 0,      # Build + Test erfolgreich
        'build_only': 0,        # Nur Build erfolgreich
        'build_failed': 0,      # Build fehlgeschlagen
        'canceled': 0           # Abgebrochen/übersprungen (keine Jobs)
    })

    for pipeline in pipelines:
        ref = pipeline.get('ref', 'unknown')
        jobs = pipeline.get('jobs', [])

        # Skip master/main branch
        if ref.lower() in ['master', 'main']:
            continue

        # Extract issue label from branch name
        issue_label = ref
        if 'issue-' in ref.lower():
            parts = ref.lower().split('issue-')
            if len(parts) > 1:
                issue_num = parts[1].split('-')[0]
                issue_label = f'Issue #{issue_num}'
        elif ref == 'master' or ref == 'main':
            issue_label = 'Master'
        else:
            issue_label = ref.split('/')[-1][:20]

        stats = branch_stats[issue_label]
        stats['total_pipelines'] += 1

        # Check if pipeline has jobs
        if not jobs:
            # Pipeline ohne Jobs = canceled/skipped
            stats['canceled'] += 1
            continue

        # Analyze pipeline: check build and test status
        build_status = None
        test_status = None

        for job in jobs:
            stage = job.get('stage', '')
            status = job.get('status', '')

            if stage == 'compile':
                build_status = status
            elif stage == 'test':
                test_status = status

        # Categorize pipeline
        # Check if pipeline was canceled/skipped
        if build_status in ['canceled', 'skipped', 'manual'] or (build_status is None):
            stats['canceled'] += 1
        elif build_status == 'success' and test_status == 'success':
            stats['both_success'] += 1
        elif build_status == 'success':  # Test nicht success (failed, skipped, etc.)
            stats['build_only'] += 1
        else:  # Build failed
            stats['build_failed'] += 1

    # Calculate percentages based on total pipelines (including canceled)
    result = {}
    for label, stats in branch_stats.items():
        total = stats['total_pipelines']
        if total > 0:
            result[label] = {
                'total_pipelines': total,
                'both_success': stats['both_success'],
                'build_only': stats['build_only'],
                'build_failed': stats['build_failed'],
                'canceled': stats['canceled'],
                'both_success_pct': stats['both_success'] / total * 100,
                'build_only_pct': stats['build_only'] / total * 100,
                'build_failed_pct': stats['build_failed'] / total * 100,
                'canceled_pct': stats['canceled'] / total * 100,
                # Für Statistik-Tabelle
                'build_success_rate': (stats['both_success'] + stats['build_only']) / total * 100,
                'test_success_rate': stats['both_success'] / total * 100
            }

    return result


def create_job_success_chart(project_label, job_stats, output_file):
    """
    Create 100% stacked bar chart for job success rates

    Each bar represents 100% of all jobs (Build + Test)
    Stacked segments show: Build success, Test success, Build failed, Test failed

    Args:
        project_label: Short label for the project (e.g., 'A01')
        job_stats: Dict mapping issue labels to job statistics
        output_file: Path to save the PNG file
    """
    if not job_stats:
        print(f"  [SKIP] Keine Job-Daten für {project_label}")
        return

    # Sort by issue number
    def sort_key(label):
        if label.startswith('Issue #'):
            try:
                return (0, int(label.split('#')[1]))
            except:
                return (1, label)
        elif label == 'Master':
            return (2, label)
        else:
            return (3, label)

    sorted_labels = sorted(job_stats.keys(), key=sort_key)

    # Prepare data for 100% stacked bar chart (Pipeline-based, ALL pipelines)
    issues = []
    both_success_pct = []  # Build + Test erfolgreich
    build_only_pct = []    # Nur Build erfolgreich
    build_failed_pct = []  # Build fehlgeschlagen
    canceled_pct = []      # Abgebrochen/übersprungen
    pipeline_counts = []

    for label in sorted_labels:
        stats = job_stats[label]

        issues.append(label)
        both_success_pct.append(stats['both_success_pct'])
        build_only_pct.append(stats['build_only_pct'])
        build_failed_pct.append(stats['build_failed_pct'])
        canceled_pct.append(stats['canceled_pct'])
        pipeline_counts.append(stats['total_pipelines'])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(issues))
    width = 0.7

    # Create stacked bars (from bottom to top) - 100% = alle Pipelines (inkl. canceled)
    # 1. Build + Test erfolgreich (grün)
    bars1 = ax.bar(x, both_success_pct, width,
                   label='Build + Test erfolgreich',
                   color='#27AE60', alpha=0.85, edgecolor='#1E8449', linewidth=1.5)

    # 2. Nur Build erfolgreich (orange)
    bars2 = ax.bar(x, build_only_pct, width, bottom=both_success_pct,
                   label='Nur Build erfolgreich',
                   color='#F39C12', alpha=0.85, edgecolor='#D68910', linewidth=1.5)

    # 3. Build fehlgeschlagen (hellgrau)
    bottom_failed = [b + o for b, o in zip(both_success_pct, build_only_pct)]
    bars3 = ax.bar(x, build_failed_pct, width, bottom=bottom_failed,
                   label='Build fehlgeschlagen',
                   color='#95A5A6', alpha=0.6, edgecolor='#7F8C8D', linewidth=1.5)

    # 4. Abgebrochen/übersprungen (dunkelgrau)
    bottom_canceled = [b + o + f for b, o, f in zip(both_success_pct, build_only_pct, build_failed_pct)]
    bars4 = ax.bar(x, canceled_pct, width, bottom=bottom_canceled,
                   label='Abgebrochen/übersprungen',
                   color='#566573', alpha=0.5, edgecolor='#34495E', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Branch/Issue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Anteil aller Pipelines (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Projekt {project_label}: Pipeline-Erfolgsraten pro Issue\n(100% = ALLE Pipelines inkl. abgebrochene)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(issues, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 105)  # Slightly above 100%

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add annotations below bars
    for i, (label, pipelines) in enumerate(zip(issues, pipeline_counts)):
        ax.text(i, -6, f'n={pipelines}',
               ha='center', va='top', fontsize=8, color='#34495E')

    # Add percentage labels in segments (only if segment is large enough)
    for i in range(len(issues)):
        percentages = [both_success_pct[i], build_only_pct[i], build_failed_pct[i], canceled_pct[i]]
        bottoms = [0,
                  both_success_pct[i],
                  both_success_pct[i] + build_only_pct[i],
                  both_success_pct[i] + build_only_pct[i] + build_failed_pct[i]]
        colors_text = ['white', 'white', '#34495E', 'white']  # White on green/orange/darkgray, dark on lightgray

        for j, (pct, bottom, color_text) in enumerate(zip(percentages, bottoms, colors_text)):
            if pct > 8:  # Only show if segment is large enough
                y_pos = bottom + pct/2
                ax.text(i, y_pos, f'{pct:.0f}%',
                       ha='center', va='center', fontsize=9,
                       fontweight='bold', color=color_text)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='#5A7A9B')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Gespeichert: {output_file}")


def create_statistics_table(all_project_data, output_file):
    """
    Create summary statistics table for pipeline success rates

    Args:
        all_project_data: Dict mapping project labels to pipeline statistics
        output_file: Path to save the CSV file
    """
    stats_list = []

    for project_label in sorted(all_project_data.keys()):
        pipeline_stats = all_project_data[project_label]

        for issue_label in sorted(pipeline_stats.keys()):
            stats = pipeline_stats[issue_label]

            row = {
                'Projekt': project_label,
                'Branch/Issue': issue_label,
                'Pipelines (n)': stats['total_pipelines'],
                'Build+Test erfolgreich': stats['both_success'],
                'Build+Test erfolgreich (%)': stats['both_success_pct'],
                'Nur Build erfolgreich': stats['build_only'],
                'Nur Build erfolgreich (%)': stats['build_only_pct'],
                'Build fehlgeschlagen': stats['build_failed'],
                'Build fehlgeschlagen (%)': stats['build_failed_pct'],
                'Abgebrochen/übersprungen': stats['canceled'],
                'Abgebrochen/übersprungen (%)': stats['canceled_pct'],
                'Build Erfolgsrate gesamt (%)': stats['build_success_rate'],
                'Test Erfolgsrate gesamt (%)': stats['test_success_rate']
            }
            stats_list.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(stats_list)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"\n  [OK] Statistik-Tabelle gespeichert: {output_file}")


def main():
    """Generate job success rate visualizations for all projects (ALL pipelines)"""

    print("="*80)
    print("PIPELINE-JOB ERFOLGSRATEN - ALLE PIPELINES (INKL. ABGEBROCHENE)")
    print("="*80)
    print()

    # Load projects from config
    projects = load_project_config()

    if not projects:
        print("[ERROR] Keine Projekte in config gefunden")
        return 1

    print(f"Verarbeite {len(projects)} Projekte...\n")

    # Create output directory
    output_dir = os.path.join('..', '..', 'visualizations', 'pipeline_job_success_all')
    os.makedirs(output_dir, exist_ok=True)

    # Collect data for all projects
    all_project_data = {}
    total_created = 0

    for project_id, project_name in projects:
        # Extract clean project label
        project_label = project_name.replace('ba_project_', '').replace('_battleship', '').upper()

        print(f"Lade {project_label}...", end=' ')

        # Get job success rate data
        job_stats = get_job_success_rates(project_name)

        if job_stats:
            total_pipelines = sum(s['total_pipelines'] for s in job_stats.values())
            avg_build_rate = np.mean([s['build_success_rate'] for s in job_stats.values()])
            avg_test_rate = np.mean([s['test_success_rate'] for s in job_stats.values()])

            print(f"[OK] ({len(job_stats)} Branches, {total_pipelines} Pipelines)")
            print(f"       Durchschn. Build: {avg_build_rate:.1f}%, Test: {avg_test_rate:.1f}%")

            all_project_data[project_label] = job_stats

            # Create visualization
            output_file = os.path.join(output_dir, f'{project_label.lower()}_job_success_all.png')
            create_job_success_chart(project_label, job_stats, output_file)
            total_created += 1
        else:
            print(f"[SKIP] (keine Daten)")

    if not all_project_data:
        print("\n[ERROR] Keine Daten für Projekte gefunden")
        return 1

    print()

    # Generate statistics table
    stats_file = os.path.join(output_dir, 'pipeline_job_success_all_statistics.csv')
    create_statistics_table(all_project_data, stats_file)

    print()
    print("="*80)
    print(f"ABGESCHLOSSEN: {total_created} Visualisierungen erstellt")
    print(f"Speicherort: {output_dir}")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
