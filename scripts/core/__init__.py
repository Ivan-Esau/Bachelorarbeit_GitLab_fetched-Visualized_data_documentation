"""
Core components for GitLab data collection

Provides:
- GitLabClient: Base API client with authentication and pagination
- load_project_config: Load projects from configuration file
- FileManager: Save and load JSON files
"""

from .gitlab_client import GitLabClient
from .config_loader import load_project_config
from .file_manager import FileManager

__all__ = ['GitLabClient', 'load_project_config', 'FileManager']
