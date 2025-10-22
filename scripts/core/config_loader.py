"""
Configuration Loader - Load project configuration from JSON

Supports letter-based project structure (a, b, c, etc.)
Handles loading projects from config/projects.json and config/project_types.json
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


def get_base_dir() -> Path:
    """Get the base directory of the project."""
    return Path(__file__).resolve().parents[2]


def load_project_types(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load project types configuration

    Args:
        config_file: Path to project_types.json (optional)

    Returns:
        Dict with project type configurations
    """
    if config_file is None:
        config_file = get_base_dir() / 'config' / 'project_types.json'
    else:
        config_file = Path(config_file)

    if not config_file.exists():
        # Return default configuration
        return {
            'project_types': {
                'a': {
                    'name': 'Series A',
                    'letter': 'a',
                    'enabled': True,
                    'projects': [f'{i:02d}' for i in range(1, 11)]
                }
            },
            'active_project_types': ['a'],
            'default_project_type': 'a'
        }

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_projects_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load projects configuration

    Args:
        config_file: Path to projects.json (optional)

    Returns:
        Dict with project configurations
    """
    if config_file is None:
        # Try new location first
        config_file = get_base_dir() / 'config' / 'projects.json'
        if not config_file.exists():
            # Fallback to old location
            config_file = get_base_dir() / 'config_projects.json'
    else:
        config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_project_letter(project_name: str) -> str:
    """
    Extract project letter from project name.

    Examples:
        'ba_project_a01_battleship' -> 'a'
        'ba_project_b05_game' -> 'b'

    Args:
        project_name: GitLab project name

    Returns:
        Project letter (a, b, c, etc.)
    """
    # Handle both old and new naming
    if '_' in project_name:
        # Old format: ba_project_a01_battleship
        parts = project_name.split('_')
        for part in parts:
            if part.startswith('a') or part.startswith('b') or part.startswith('c'):
                if len(part) >= 2 and part[1:].isdigit():
                    return part[0]

    # New format: a01, b01, etc.
    if project_name[0].isalpha():
        return project_name[0]

    # Default to 'a'
    return 'a'


def get_project_number(project_name: str) -> str:
    """
    Extract project number from project name.

    Examples:
        'ba_project_a01_battleship' -> '01'
        'a01' -> '01'

    Args:
        project_name: GitLab project name or short name

    Returns:
        Project number as string (e.g., '01', '02')
    """
    # Handle both old and new naming
    if '_' in project_name:
        # Old format: ba_project_a01_battleship
        parts = project_name.split('_')
        for part in parts:
            if part[0].isalpha() and len(part) >= 2 and part[1:].isdigit():
                return part[1:]

    # New format: a01, b01, etc.
    if project_name[0].isalpha() and len(project_name) >= 2:
        return project_name[1:]

    return '01'


def get_projects_by_letter(letter: str) -> List[Tuple[str, str, str]]:
    """
    Get all projects for a specific letter.

    Args:
        letter: Project letter (a, b, c, etc.)

    Returns:
        List of (letter, number, gitlab_full_path) tuples
        Example: [('a', '01', 'simulationsprojects_bachelorarbeit_ivan_esau/ba_project_a01_battleship'), ...]
    """
    types_config = load_project_types()
    projects = []

    if letter in types_config['project_types']:
        type_config = types_config['project_types'][letter]

        if type_config.get('enabled', False):
            gitlab_group = type_config.get('gitlab_group', '')
            gitlab_prefix = type_config.get('gitlab_prefix', f'ba_project_{letter}')
            gitlab_suffix = type_config.get('gitlab_suffix', '_battleship')

            for number in type_config.get('projects', []):
                project_name = f"{gitlab_prefix}{number}{gitlab_suffix}"
                # Include group in full path if specified
                if gitlab_group:
                    gitlab_full_path = f"{gitlab_group}/{project_name}"
                else:
                    gitlab_full_path = project_name
                projects.append((letter, number, gitlab_full_path))

    return projects


def load_project_config(config_file: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Load project configuration (BACKWARD COMPATIBLE)

    Args:
        config_file: Path to config file (optional)

    Returns:
        List of (project_id, project_name) tuples for active project types

    Example:
        projects = load_project_config()
        # Returns: [
        #   ('a01', 'ba_project_a01_battleship'),
        #   ('a02', 'ba_project_a02_battleship'),
        #   ...
        # ]
    """
    types_config = load_project_types()
    active_types = types_config.get('active_project_types', ['a'])

    projects = []
    for letter in active_types:
        letter_projects = get_projects_by_letter(letter)
        for letter_val, number, gitlab_name in letter_projects:
            project_id = f"{letter_val}{number}"
            projects.append((project_id, gitlab_name))

    return projects


def load_project_names(config_file: Optional[str] = None) -> List[str]:
    """
    Load only project names from configuration

    Args:
        config_file: Path to config file

    Returns:
        List of project names

    Example:
        names = load_project_names()
        # Returns: ['ba_project_a01_battleship', 'ba_project_a02_battleship', ...]
    """
    projects = load_project_config(config_file)
    return [name for _, name in projects]


def get_project_count(config_file: Optional[str] = None) -> int:
    """
    Get total number of configured projects

    Args:
        config_file: Path to config file

    Returns:
        Number of projects
    """
    return len(load_project_config(config_file))


def get_project_mapping(letter: str) -> Dict[str, str]:
    """
    Get mapping of project IDs to GitLab names for a specific letter.

    Args:
        letter: Project letter (a, b, c, etc.)

    Returns:
        Dict mapping project IDs to GitLab names
        Example: {'A01': 'ba_project_a01_battleship', ...}
    """
    letter_projects = get_projects_by_letter(letter)
    mapping = {}

    for letter_val, number, gitlab_name in letter_projects:
        project_id = f"{letter_val.upper()}{number}"
        mapping[project_id] = gitlab_name

    return mapping


# Example usage
if __name__ == "__main__":
    print("Testing config_loader...")
    print()

    # Test project types
    types = load_project_types()
    print(f"Active project types: {types.get('active_project_types', [])}")
    print()

    # Test loading projects
    projects = load_project_config()
    print(f"Total projects: {len(projects)}")
    print("Projects:")
    for project_id, gitlab_name in projects:
        print(f"  {project_id}: {gitlab_name}")
    print()

    # Test letter extraction
    test_names = [
        'ba_project_a01_battleship',
        'ba_project_b05_game',
        'a01',
        'b10'
    ]
    print("Letter extraction tests:")
    for name in test_names:
        letter = get_project_letter(name)
        number = get_project_number(name)
        print(f"  {name} -> letter: {letter}, number: {number}")
