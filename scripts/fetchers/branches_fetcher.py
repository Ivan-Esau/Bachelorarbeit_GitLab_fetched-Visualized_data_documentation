"""
Branches Fetcher - Fetch branches from GitLab
"""

from typing import List, Dict, Any
from .base_fetcher import BaseFetcher


class BranchesFetcher(BaseFetcher):
    """Fetches ALL branches for a project"""

    def fetch(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL branches for a project

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of all branches

        API: GET /projects/:id/repository/branches
        """
        print(f"  Fetching branches...")

        endpoint = f'/projects/{self._quote_project_id(project_id)}/repository/branches'
        branches = self.client.get_paginated(endpoint)

        print(f"    Found {len(branches)} branches")

        return branches
