"""
Merge Requests Fetcher - Fetch merge requests from GitLab
"""

from typing import List, Dict, Any
from .base_fetcher import BaseFetcher


class MergeRequestsFetcher(BaseFetcher):
    """Fetches ALL merge requests for a project"""

    def fetch(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL merge requests for a project

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of all merge requests

        API: GET /projects/:id/merge_requests
        """
        print(f"  Fetching merge requests...")

        endpoint = f'/projects/{self._quote_project_id(project_id)}/merge_requests'
        params = {
            'state': 'all',
            'order_by': 'created_at',
            'sort': 'asc'
        }

        merge_requests = self.client.get_paginated(endpoint, params)
        print(f"    Found {len(merge_requests)} merge requests")

        return merge_requests
