"""
Configuration Loader - Load project configuration from JSON

Handles loading projects from config_projects.json
"""

import json
import os
from typing import List, Tuple, Dict, Any


def load_project_config(config_file: str = '../../config_projects.json') -> List[Tuple[str, str]]:
    """
    Load project configuration from JSON file

    Args:
        config_file: Path to config file (relative to script location)

    Returns:
        List of (project_id, project_name) tuples

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid

    Example:
        projects = load_project_config()
        # Returns: [
        #   ('namespace/project1', 'project1'),
        #   ('namespace/project2', 'project2')
        # ]
    """
    # Handle relative paths
    if not os.path.isabs(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, config_file)

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Please create config_projects.json with project definitions"
        )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'gitlab_projects' not in config:
            raise ValueError("Config file missing 'gitlab_projects' key")

        projects = []
        for project in config['gitlab_projects']:
            if 'id' not in project or 'name' not in project:
                print(f"Warning: Skipping invalid project entry: {project}")
                continue
            projects.append((project['id'], project['name']))

        if not projects:
            raise ValueError("No valid projects found in config")

        return projects

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


def load_project_names(config_file: str = '../../config_projects.json') -> List[str]:
    """
    Load only project names from configuration

    Args:
        config_file: Path to config file

    Returns:
        List of project names

    Example:
        names = load_project_names()
        # Returns: ['project1', 'project2', ...]
    """
    projects = load_project_config(config_file)
    return [name for _, name in projects]


def get_project_count(config_file: str = '../../config_projects.json') -> int:
    """
    Get total number of configured projects

    Args:
        config_file: Path to config file

    Returns:
        Number of projects
    """
    return len(load_project_config(config_file))
