"""
Path Helpers - Consistent path management for letter-based structure

Provides utility functions for getting data, analysis, and visualization paths
for different project types (a, b, c, etc.)

Author: Data Analysis System
Date: 2025-10-22
"""

from pathlib import Path
from typing import Optional

try:
    from scripts.core.config_loader import get_project_letter, get_project_number
except ModuleNotFoundError:
    from .config_loader import get_project_letter, get_project_number


def get_base_dir() -> Path:
    """Get the base directory of the project."""
    return Path(__file__).resolve().parents[2]


def get_data_dir(project_name: str, project_type: Optional[str] = None) -> Path:
    """
    Get data directory for a project.

    Args:
        project_name: GitLab project name (e.g., 'ba_project_a01_battleship') or short name (e.g., 'a01')
        project_type: Optional project type override (e.g., 'a', 'b')

    Returns:
        Path to project data directory

    Examples:
        get_data_dir('ba_project_a01_battleship') -> data_raw/a/a01/
        get_data_dir('a01') -> data_raw/a/a01/
        get_data_dir('a01', 'a') -> data_raw/a/a01/
    """
    if project_type is None:
        project_type = get_project_letter(project_name)

    project_number = get_project_number(project_name)
    project_id = f"{project_type}{project_number}"

    return get_base_dir() / "data_raw" / project_type / project_id


def get_analysis_dir(project_type: str, analysis_name: str) -> Path:
    """
    Get analysis directory for a project type.

    Args:
        project_type: Project type letter (e.g., 'a', 'b', 'c')
        analysis_name: Analysis subdirectory name (e.g., 'pipeline_investigation')

    Returns:
        Path to analysis directory

    Examples:
        get_analysis_dir('a', 'pipeline_investigation') -> analysis_results/a/pipeline_investigation/
        get_analysis_dir('b', 'quality_analysis') -> analysis_results/b/quality_analysis/
    """
    return get_base_dir() / "analysis_results" / project_type / analysis_name


def get_viz_dir(project_type: str, viz_category: str) -> Path:
    """
    Get visualization directory for a project type.

    Args:
        project_type: Project type letter (e.g., 'a', 'b', 'c')
        viz_category: Visualization category (e.g., 'pipeline_investigation', 'branch_lifecycle')

    Returns:
        Path to visualization directory

    Examples:
        get_viz_dir('a', 'pipeline_investigation') -> visualizations/a/pipeline_investigation/
        get_viz_dir('a', 'summary') -> visualizations/a/summary/
    """
    return get_base_dir() / "visualizations" / project_type / viz_category


def get_data_file(project_name: str, filename: str, project_type: Optional[str] = None) -> Path:
    """
    Get full path to a data file for a project.

    Args:
        project_name: GitLab project name or short name
        filename: Name of the file (e.g., 'pipelines.json', 'issues.json')
        project_type: Optional project type override

    Returns:
        Full path to the data file

    Examples:
        get_data_file('a01', 'pipelines.json') -> data_raw/a/a01/pipelines.json
        get_data_file('ba_project_a01_battleship', 'issues.json') -> data_raw/a/a01/issues.json
    """
    return get_data_dir(project_name, project_type) / filename


def get_analysis_file(project_type: str, analysis_name: str, filename: str) -> Path:
    """
    Get full path to an analysis output file.

    Args:
        project_type: Project type letter
        analysis_name: Analysis subdirectory name
        filename: Output filename (e.g., 'branch_summaries.csv')

    Returns:
        Full path to the analysis file

    Examples:
        get_analysis_file('a', 'pipeline_investigation', 'branch_summaries.csv')
        -> analysis_results/a/pipeline_investigation/branch_summaries.csv
    """
    return get_analysis_dir(project_type, analysis_name) / filename


def get_viz_file(project_type: str, viz_category: str, filename: str) -> Path:
    """
    Get full path to a visualization file.

    Args:
        project_type: Project type letter
        viz_category: Visualization category
        filename: Visualization filename (e.g., '04_failure_type_stacked_bars.png')

    Returns:
        Full path to the visualization file

    Examples:
        get_viz_file('a', 'pipeline_investigation', '04_failure_type_stacked_bars.png')
        -> visualizations/a/pipeline_investigation/04_failure_type_stacked_bars.png
    """
    return get_viz_dir(project_type, viz_category) / filename


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# Example usage
if __name__ == "__main__":
    print("Testing path_helpers...")
    print()

    # Test data paths
    print("Data paths:")
    print(f"  get_data_dir('a01'): {get_data_dir('a01')}")
    print(f"  get_data_dir('ba_project_a01_battleship'): {get_data_dir('ba_project_a01_battleship')}")
    print(f"  get_data_file('a01', 'pipelines.json'): {get_data_file('a01', 'pipelines.json')}")
    print()

    # Test analysis paths
    print("Analysis paths:")
    print(f"  get_analysis_dir('a', 'pipeline_investigation'): {get_analysis_dir('a', 'pipeline_investigation')}")
    print(f"  get_analysis_file('a', 'pipeline_investigation', 'branch_summaries.csv'):")
    print(f"    {get_analysis_file('a', 'pipeline_investigation', 'branch_summaries.csv')}")
    print()

    # Test visualization paths
    print("Visualization paths:")
    print(f"  get_viz_dir('a', 'pipeline_investigation'): {get_viz_dir('a', 'pipeline_investigation')}")
    print(f"  get_viz_file('a', 'pipeline_investigation', '04_failure_type_stacked_bars.png'):")
    print(f"    {get_viz_file('a', 'pipeline_investigation', '04_failure_type_stacked_bars.png')}")
