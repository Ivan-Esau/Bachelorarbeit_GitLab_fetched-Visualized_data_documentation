"""
Commits Fetcher - Fetch commits from GitLab

Fetches TWO types of commit data:
1. ALL commits from all branches
2. Commits per merge request (for linkage)
"""

from typing import List, Dict, Any
from .base_fetcher import BaseFetcher


class CommitsFetcher(BaseFetcher):
    """Fetches commits for a project"""

    def fetch_all_commits(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL commits from ALL branches

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of all commits (complete repository history)

        API: GET /projects/:id/repository/commits?all=true
        """
        print(f"  Fetching ALL commits from all branches...")

        endpoint = f'/projects/{self._quote_project_id(project_id)}/repository/commits'
        params = {
            'all': 'true',  # Include commits from all branches
            'order': 'default',  # Chronological order
            'with_stats': 'false'  # Don't need line change stats
        }

        commits = self.client.get_paginated(endpoint, params)
        print(f"    Found {len(commits)} commits (all branches)")

        return commits

    def fetch_mr_commits(self, project_id: str, mr_iid: int) -> List[Dict[str, Any]]:
        """
        Fetch commits for a specific merge request

        Args:
            project_id: GitLab project ID or path
            mr_iid: Merge request IID (internal ID)

        Returns:
            List of commits for this MR

        API: GET /projects/:id/merge_requests/:mr_iid/commits
        """
        endpoint = f'/projects/{self._quote_project_id(project_id)}/merge_requests/{mr_iid}/commits'
        commits = self.client.get_paginated(endpoint)
        return commits

    def fetch_commits_by_mr(
        self,
        project_id: str,
        merge_requests: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch commits for all merge requests

        Args:
            project_id: GitLab project ID or path
            merge_requests: List of merge request data

        Returns:
            Dictionary mapping MR IID (as string) to list of commits

        Example:
            {
                "10": [commit1, commit2, ...],
                "11": [commit3, commit4, ...]
            }
        """
        print(f"  Fetching commits for merge requests...")

        commits_by_mr = {}

        for i, mr in enumerate(merge_requests, 1):
            if i % 10 == 0 or i == 1:
                print(f"    Processing MR {i}/{len(merge_requests)}")

            mr_iid = mr['iid']
            commits = self.fetch_mr_commits(project_id, mr_iid)
            commits_by_mr[str(mr_iid)] = commits

        return commits_by_mr

    def fetch(self, project_id: str, merge_requests: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch both types of commit data

        Args:
            project_id: GitLab project ID or path
            merge_requests: List of MRs (optional, for MR commits)

        Returns:
            Dictionary with 'all_commits' and 'commits_by_mr'
        """
        result = {
            'all_commits': self.fetch_all_commits(project_id),
            'commits_by_mr': {}
        }

        if merge_requests:
            result['commits_by_mr'] = self.fetch_commits_by_mr(project_id, merge_requests)

        return result
