"""
File Manager - Save and load JSON files

Handles all file I/O operations for the project
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


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
        output_base_dir: str
    ) -> List[str]:
        """
        Save complete project data to organized folder structure

        Args:
            project_data: Complete project data dictionary
            output_base_dir: Base output directory

        Returns:
            List of saved file paths

        Example:
            files = FileManager.save_project_data(data, 'data_raw')
        """
        project_name = project_data['project_name']
        project_dir = os.path.join(output_base_dir, project_name)
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
        data_dir: str = '../data_raw'
    ) -> Dict[str, Any]:
        """
        Load complete project data from folder

        Args:
            project_name: Project name
            data_dir: Base data directory

        Returns:
            Dictionary with all project data

        Raises:
            FileNotFoundError: If project folder or files don't exist

        Example:
            data = FileManager.load_project_data('ba_project_a01_battleship')
        """
        project_dir = os.path.join(data_dir, project_name)

        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project folder not found: {project_dir}")

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
