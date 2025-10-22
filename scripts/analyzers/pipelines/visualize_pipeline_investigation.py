"""
Pipeline Investigation Visualization Script

Creates focused visualization to analyze pipeline failures:
- Failure Type Stacked Bars (Why they fail per project)

Supports letter-based project structure (a, b, c, etc.)

Author: Data Analysis System
Date: 2025-10-21
Updated: 2025-10-22 (Letter-based structure support)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from scripts.core.path_helpers import get_analysis_file, get_viz_file, ensure_dir
except ModuleNotFoundError:
    from core.path_helpers import get_analysis_file, get_viz_file, ensure_dir

# Set style for scientific visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Color scheme
COLORS = {
    'success': '#2ecc71',      # Green
    'test_failure': '#e74c3c', # Red
    'test_timeout': '#e67e22', # Orange
    'compile_failure': '#f39c12', # Yellow-orange
    'test_skipped': '#95a5a6', # Grey
    'critical': '#c0392b',     # Dark red
    'high': '#e67e22',         # Orange
    'medium': '#f39c12',       # Yellow
    'low': '#27ae60',          # Green
}


def load_data(project_type: str):
    """Load all analysis data using letter-based structure."""
    print("Loading data...")

    branch_summary = pd.read_csv(get_analysis_file(project_type, 'pipeline_investigation', 'branch_summaries.csv'))
    detailed = pd.read_csv(get_analysis_file(project_type, 'pipeline_investigation', 'detailed_pipeline_analysis.csv'))
    stuck = pd.read_csv(get_analysis_file(project_type, 'pipeline_investigation', 'stuck_pipelines_detailed.csv'))
    critical = pd.read_csv(get_analysis_file(project_type, 'pipeline_investigation', 'critical_branches.csv'))

    print(f"  Loaded {len(branch_summary)} branches")
    print(f"  Loaded {len(detailed)} pipelines")
    print(f"  Loaded {len(stuck)} stuck pipelines")
    print(f"  Loaded {len(critical)} critical branches")

    return branch_summary, detailed, stuck, critical


def viz4_failure_type_stacked_bars(branch_summary, project_type: str):
    """
    Visualization 4: Failure Type Stacked Bars
    Shows why pipelines fail per project (excluding master branch)

    Args:
        branch_summary: DataFrame with branch summary data
        project_type: Project type letter for output path
    """
    print("\nCreating Failure Type Stacked Bars...")

    # Filter out master branch
    branch_summary_no_master = branch_summary[branch_summary['issue_number'] != 'master'].copy()
    print(f"  Filtered to {len(branch_summary_no_master)} branches (excluding master)")

    # Group by project
    project_data = branch_summary_no_master.groupby('project').agg({
        'success_count': 'sum',
        'test_failures': 'sum',
        'test_cancellations': 'sum',
        'compile_failures': 'sum',
        'total_pipelines': 'sum'
    }).reset_index()

    # Calculate test skipped
    project_data['test_skipped'] = (project_data['total_pipelines'] -
                                     project_data['success_count'] -
                                     project_data['test_failures'] -
                                     project_data['test_cancellations'] -
                                     project_data['compile_failures'])

    # Calculate percentages
    for col in ['success_count', 'test_failures', 'test_cancellations',
                'compile_failures', 'test_skipped']:
        project_data[f'{col}_pct'] = (project_data[col] / project_data['total_pipelines'] * 100)

    # Sort by success rate
    project_data = project_data.sort_values('success_count_pct')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    projects = project_data['project']
    x = np.arange(len(projects))
    width = 0.7

    # Create stacked bars
    bottom = np.zeros(len(projects))

    categories = [
        ('success_count_pct', 'Success', COLORS['success']),
        ('test_failures_pct', 'Test Failures', COLORS['test_failure']),
        ('test_cancellations_pct', 'Test Timeouts', COLORS['test_timeout']),
        ('compile_failures_pct', 'Compile Failures', COLORS['compile_failure']),
        ('test_skipped_pct', 'Test Skipped', COLORS['test_skipped']),
    ]

    for col, label, color in categories:
        values = project_data[col].values
        ax.bar(x, values, width, label=label, bottom=bottom, color=color, edgecolor='black', linewidth=0.5)

        # Add percentage labels for significant segments
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:  # Only label if >5%
                ax.text(i, bot + val/2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=8, weight='bold')

        bottom += values

    ax.set_xlabel('Project (sorted by success rate)', fontsize=12, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Pipeline Outcome Distribution by Project\nWhy Do Pipelines Fail? (Excluding Master Branch)',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(projects)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add sample size labels
    for i, (idx, row) in enumerate(project_data.iterrows()):
        ax.text(i, -5, f"n={row['total_pipelines']:.0f}",
                ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    output_path = get_viz_file(project_type, 'pipeline_investigation', '04_failure_type_stacked_bars.png')
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize pipeline investigation data')
    parser.add_argument(
        '--project-type',
        type=str,
        default='a',
        help='Project type letter (a, b, c, etc.). Default: a'
    )
    args = parser.parse_args()

    project_type = args.project_type

    print("=" * 80)
    print("PIPELINE INVESTIGATION VISUALIZATION")
    print("Creating Failure Type Analysis")
    print("=" * 80)
    print(f"Project Type: {project_type}")
    print()

    # Load data
    branch_summary, detailed, stuck, critical = load_data(project_type)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    # Create only the failure type stacked bars visualization
    viz4_failure_type_stacked_bars(branch_summary, project_type)

    # Get output directory for display
    output_dir = Path(get_viz_file(project_type, 'pipeline_investigation', 'dummy')).parent

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nVisualization saved to:")
    print(f"  {output_dir}")
    print("\nGenerated file:")
    print("  - 04_failure_type_stacked_bars.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
