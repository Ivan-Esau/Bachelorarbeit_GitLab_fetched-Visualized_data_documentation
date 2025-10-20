"""
Base Fetcher - Abstract base class for all fetchers

Provides common interface and functionality for all data fetchers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import requests


class BaseFetcher(ABC):
    """
    Abstract base class for all GitLab data fetchers

    Each fetcher must implement the fetch() method
    """

    def __init__(self, client):
        """
        Initialize fetcher with GitLab client

        Args:
            client: GitLabClient instance
        """
        self.client = client

    @abstractmethod
    def fetch(self, project_id: str) -> Any:
        """
        Fetch data for a project

        Args:
            project_id: GitLab project ID or path

        Returns:
            Fetched data (type depends on fetcher)
        """
        pass

    def _quote_project_id(self, project_id: str) -> str:
        """
        URL-encode project ID for API calls

        Args:
            project_id: Project ID or path

        Returns:
            URL-encoded project ID
        """
        return requests.utils.quote(str(project_id), safe="")
