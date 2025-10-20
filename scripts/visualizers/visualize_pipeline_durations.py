"""
Visualize Pipeline Durations per Project

Creates one boxplot per project showing the distribution of pipeline execution times
for each branch/issue. Each box represents all pipeline runs for that branch.

This follows scientific research standards with German labels.

Input Data Requirements:
    - pipelines.json: Pipeline execution data with duration and status

Output:
    - One PNG per project: visualizations/pipeline_durations/{project}_pipeline_durations.png
    - Statistics CSV: pipeline_durations_statistics.csv

Usage:
    python visualize_pipeline_durations.py
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scripts.core import load_project_config

# Set style
sns.set_style("whitegrid")


def get_pipeline_durations(project_name, data_base_dir='../../data_raw'):
    """
    Extract pipeline durations grouped by branch/issue

    Args:
        project_name: Name of the project directory
        data_base_dir: Base directory containing project data

    Returns:
        Dict mapping issue labels to list of pipeline durations (in seconds)
    """
    data_dir = os.path.join(data_base_dir, project_name)
    pipelines_file = os.path.join(data_dir, 'pipelines.json')

    if not os.path.exists(pipelines_file):
        return {}

    with open(pipelines_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    # Group pipelines by branch
    branch_durations = defaultdict(lambda: {'durations': [], 'success': 0, 'failed': 0})

    for pipeline in pipelines:
        ref = pipeline.get('ref', 'unknown')
        duration = pipeline.get('duration')
        status = pipeline.get('status')

        if duration is not None and duration > 0:
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
                # Truncate long branch names
                issue_label = ref.split('/')[-1][:20]

            branch_durations[issue_label]['durations'].append(duration)
            if status == 'success':
                branch_durations[issue_label]['success'] += 1
            elif status == 'failed':
                branch_durations[issue_label]['failed'] += 1

    # Convert to simple dict with only durations for plotting
    result = {}
    for label, data in branch_durations.items():
        if len(data['durations']) > 0:
            result[label] = data['durations']

    return result


def create_pipeline_boxplot(project_label, pipeline_data, output_file):
    """
    Create boxplot for pipeline durations per branch/issue

    Args:
        project_label: Short label for the project (e.g., 'A01')
        pipeline_data: Dict mapping issue labels to pipeline durations
        output_file: Path to save the PNG file
    """
    if not pipeline_data:
        print(f"  [SKIP] Keine Pipeline-Daten für {project_label}")
        return

    # Sort by issue number if possible
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

    sorted_labels = sorted(pipeline_data.keys(), key=sort_key)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for boxplot
    positions = range(1, len(sorted_labels) + 1)
    box_data = [pipeline_data[label] for label in sorted_labels]

    # Create boxplot
    bp = ax.boxplot(box_data,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='#5A7A9B', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='#34495E', linewidth=2.5),
                    whiskerprops=dict(color='#34495E', linewidth=1.5),
                    capprops=dict(color='#34495E', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#E74C3C',
                                  markersize=6, alpha=0.7, markeredgecolor='#C0392B'))

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_labels, fontsize=10, rotation=45, ha='right')
    ax.set_xlabel('Branch/Issue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pipeline-Dauer (Sekunden)', fontsize=12, fontweight='bold')
    ax.set_title(f'Projekt {project_label}: Pipeline-Ausführungszeiten pro Branch/Issue',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add sample size annotations (n = number of pipeline runs)
    for i, label in enumerate(sorted_labels, 1):
        n = len(pipeline_data[label])
        ax.text(i, ax.get_ylim()[0], f'n={n}',
               ha='center', va='top', fontsize=8, color='#34495E')

    # Add legend
    legend_elements = [
        Patch(facecolor='#5A7A9B', alpha=0.7, edgecolor='#34495E', linewidth=1.5,
              label='Box (IQR: Q1-Q3)'),
        Line2D([0], [0], color='#34495E', linewidth=2.5, label='Median'),
        Line2D([0], [0], color='#34495E', linewidth=1.5, label='Whisker (1,5×IQR)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=6, alpha=0.7, markeredgecolor='#C0392B',
               label='Ausreißer', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             framealpha=0.95, edgecolor='#5A7A9B')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Gespeichert: {output_file}")


def create_statistics_table(all_project_data, output_file):
    """
    Create summary statistics table for pipeline durations

    Args:
        all_project_data: Dict mapping project labels to pipeline data
        output_file: Path to save the CSV file
    """
    stats_list = []

    for project_label in sorted(all_project_data.keys()):
        pipeline_data = all_project_data[project_label]

        for issue_label in sorted(pipeline_data.keys()):
            durations = pipeline_data[issue_label]

            stats = {
                'Projekt': project_label,
                'Branch/Issue': issue_label,
                'n (Pipelines)': len(durations),
                'Median (s)': np.median(durations),
                'Mittelwert (s)': np.mean(durations),
                'Stdabw (s)': np.std(durations, ddof=1) if len(durations) > 1 else 0,
                'Min (s)': min(durations),
                'Max (s)': max(durations)
            }
            stats_list.append(stats)

    # Create DataFrame and save
    df = pd.DataFrame(stats_list)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"\n  [OK] Statistik-Tabelle gespeichert: {output_file}")


def main():
    """Generate pipeline duration boxplots for all projects"""

    print("="*80)
    print("PIPELINE-DAUER VISUALISIERUNG - BOXPLOTS PRO PROJEKT")
    print("="*80)
    print()

    # Load projects from config
    projects = load_project_config()

    if not projects:
        print("[ERROR] Keine Projekte in config gefunden")
        return 1

    print(f"Verarbeite {len(projects)} Projekte...\n")

    # Create output directory
    output_dir = os.path.join('..', '..', 'visualizations', 'pipeline_durations')
    os.makedirs(output_dir, exist_ok=True)

    # Collect data for all projects
    all_project_data = {}
    total_created = 0

    for project_id, project_name in projects:
        # Extract clean project label
        project_label = project_name.replace('ba_project_', '').replace('_battleship', '').upper()

        print(f"Lade {project_label}...", end=' ')

        # Get pipeline duration data
        pipeline_data = get_pipeline_durations(project_name)

        if pipeline_data:
            total_pipelines = sum(len(durations) for durations in pipeline_data.values())
            print(f"[OK] ({len(pipeline_data)} Branches, {total_pipelines} Pipelines)")

            all_project_data[project_label] = pipeline_data

            # Create visualization
            output_file = os.path.join(output_dir, f'{project_label.lower()}_pipeline_durations.png')
            create_pipeline_boxplot(project_label, pipeline_data, output_file)
            total_created += 1
        else:
            print(f"[SKIP] (keine Daten)")

    if not all_project_data:
        print("\n[ERROR] Keine Daten für Projekte gefunden")
        return 1

    print()

    # Generate statistics table
    stats_file = os.path.join(output_dir, 'pipeline_durations_statistics.csv')
    create_statistics_table(all_project_data, stats_file)

    print()
    print("="*80)
    print(f"ABGESCHLOSSEN: {total_created} Visualisierungen erstellt")
    print(f"Speicherort: {output_dir}")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
