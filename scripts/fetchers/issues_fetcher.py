"""
Issues Fetcher - Fetch issues from GitLab
"""

from typing import List, Dict, Any
from .base_fetcher import BaseFetcher


class IssuesFetcher(BaseFetcher):
    """Fetches ALL issues for a project"""

    def fetch(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL issues for a project

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of all issues

        API: GET /projects/:id/issues
        """
        print(f"  Fetching issues...")

        endpoint = f'/projects/{self._quote_project_id(project_id)}/issues'
        params = {
            'scope': 'all',
            'order_by': 'created_at',
            'sort': 'asc'
        }

        issues = self.client.get_paginated(endpoint, params)
        print(f"    Found {len(issues)} issues")

        return issues
