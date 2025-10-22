"""
File Manager - Save and load JSON files

Handles all file I/O operations for the project
Supports letter-based project structure (a, b, c, etc.)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from scripts.core.config_loader import get_project_letter, get_project_number
except ModuleNotFoundError:
    from .config_loader import get_project_letter, get_project_number


class FileManager:
    """
    Manages file I/O operations for JSON data

    Handles:
    - Creating directories
    - Saving JSON files
    - Loading JSON files
    - Pretty printing with proper encoding
    """

    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure directory exists, create if it doesn't

        Args:
            directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_json(
        data: Any,
        file_path: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """
        Save data to JSON file with pretty formatting

        Args:
            data: Data to save (must be JSON serializable)
            file_path: Output file path
            indent: JSON indentation (default: 2)
            ensure_ascii: Escape non-ASCII characters (default: False)

        Example:
            FileManager.save_json(issues, 'data/issues.json')
        """
        # Ensure parent directory exists
        directory = os.path.dirname(file_path)
        if directory:
            FileManager.ensure_directory(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    @staticmethod
    def load_json(file_path: str) -> Any:
        """
        Load data from JSON file

        Args:
            file_path: Input file path

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid

        Example:
            issues = FileManager.load_json('data/issues.json')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")

    @staticmethod
    def save_project_data(
        project_data: Dict[str, Any],
        output_base_dir: str,
        project_type: Optional[str] = None
    ) -> List[str]:
        """
        Save complete project data to organized folder structure

        Uses letter-based structure: output_base_dir/letter/project_id/

        Args:
            project_data: Complete project data dictionary
            output_base_dir: Base output directory
            project_type: Optional project type override (e.g., 'a', 'b')

        Returns:
            List of saved file paths

        Example:
            files = FileManager.save_project_data(data, 'data_raw')
            # Saves to: data_raw/a/a01/ (for project_name='ba_project_a01_battleship')
        """
        project_name = project_data['project_name']

        # Extract letter and number from project name
        letter = project_type if project_type else get_project_letter(project_name)
        number = get_project_number(project_name)
        project_id = f"{letter}{number}"

        # Create letter-based directory structure: data_raw/a/a01/
        project_dir = os.path.join(output_base_dir, letter, project_id)
        FileManager.ensure_directory(project_dir)

        files_saved = []

        # Save each data type
        data_files = {
            'issues.json': project_data['issues'],
            'branches.json': project_data['branches'],
            'merge_requests.json': project_data['merge_requests'],
            'all_commits.json': project_data['all_commits'],
            'commits_by_mr.json': project_data['commits_by_mr'],
            'pipelines.json': project_data['pipelines'],
            'artifacts.json': project_data['artifacts'],
            'coverage.json': project_data['coverage']
        }

        for filename, data in data_files.items():
            file_path = os.path.join(project_dir, filename)
            FileManager.save_json(data, file_path)
            files_saved.append(file_path)

        # Save metadata
        metadata = {
            'project_id': project_data['project_id'],
            'project_name': project_data['project_name'],
            'fetch_date': project_data['fetch_date'],
            'stats': project_data['stats']
        }
        metadata_file = os.path.join(project_dir, 'metadata.json')
        FileManager.save_json(metadata, metadata_file)
        files_saved.append(metadata_file)

        return files_saved

    @staticmethod
    def load_project_data(
        project_name: str,
        data_dir: str = '../data_raw',
        project_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load complete project data from folder

        Supports both old flat structure and new letter-based structure

        Args:
            project_name: Project name (e.g., 'ba_project_a01_battleship' or 'a01')
            data_dir: Base data directory
            project_type: Optional project type override (e.g., 'a', 'b')

        Returns:
            Dictionary with all project data

        Raises:
            FileNotFoundError: If project folder or files don't exist

        Example:
            data = FileManager.load_project_data('ba_project_a01_battleship')
            # Tries: data_raw/a/a01/ first, then data_raw/ba_project_a01_battleship/
        """
        # Try new letter-based structure first
        letter = project_type if project_type else get_project_letter(project_name)
        number = get_project_number(project_name)
        project_id = f"{letter}{number}"

        new_project_dir = os.path.join(data_dir, letter, project_id)
        old_project_dir = os.path.join(data_dir, project_name)

        # Try new structure first, fall back to old structure
        if os.path.exists(new_project_dir):
            project_dir = new_project_dir
        elif os.path.exists(old_project_dir):
            project_dir = old_project_dir
        else:
            raise FileNotFoundError(
                f"Project folder not found in either location:\n"
                f"  New: {new_project_dir}\n"
                f"  Old: {old_project_dir}"
            )

        data = {
            'project_name': project_name,
            'issues': FileManager.load_json(os.path.join(project_dir, 'issues.json')),
            'branches': FileManager.load_json(os.path.join(project_dir, 'branches.json')),
            'merge_requests': FileManager.load_json(os.path.join(project_dir, 'merge_requests.json')),
            'all_commits': FileManager.load_json(os.path.join(project_dir, 'all_commits.json')),
            'commits_by_mr': FileManager.load_json(os.path.join(project_dir, 'commits_by_mr.json')),
            'pipelines': FileManager.load_json(os.path.join(project_dir, 'pipelines.json')),
            'metadata': FileManager.load_json(os.path.join(project_dir, 'metadata.json'))
        }

        return data

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size in bytes

        Args:
            file_path: File path

        Returns:
            File size in bytes
        """
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        Format byte size to human-readable string

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
