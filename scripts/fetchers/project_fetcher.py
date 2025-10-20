"""
Project Fetcher - Orchestrates all data fetching for a project

Coordinates all individual fetchers to collect complete project data
"""

from datetime import datetime
from typing import Dict, Any
from core.gitlab_client import GitLabClient
from core.file_manager import FileManager
from .issues_fetcher import IssuesFetcher
from .branches_fetcher import BranchesFetcher
from .merge_requests_fetcher import MergeRequestsFetcher
from .commits_fetcher import CommitsFetcher
from .pipelines_fetcher import PipelinesFetcher
from .artifacts_fetcher import ArtifactsFetcher
from .coverage_extractor import CoverageExtractor


class ProjectFetcher:
    """
    Orchestrates data fetching for a complete project

    Manages all individual fetchers and combines their results
    into a complete project dataset
    """

    def __init__(self, client: GitLabClient):
        """
        Initialize project fetcher with client

        Args:
            client: GitLabClient instance
        """
        self.client = client

        # Initialize all fetchers
        self.issues_fetcher = IssuesFetcher(client)
        self.branches_fetcher = BranchesFetcher(client)
        self.merge_requests_fetcher = MergeRequestsFetcher(client)
        self.commits_fetcher = CommitsFetcher(client)
        self.pipelines_fetcher = PipelinesFetcher(client)
        self.artifacts_fetcher = ArtifactsFetcher(client)
        self.coverage_extractor = CoverageExtractor(client)

    def fetch_complete_project(
        self,
        project_id: str,
        project_name: str
    ) -> Dict[str, Any]:
        """
        Fetch complete data for a project

        Args:
            project_id: GitLab project ID or path
            project_name: Project name (for organization)

        Returns:
            Dictionary with all project data

        Example:
            {
                'project_id': '...',
                'project_name': 'ba_project_a01_battleship',
                'fetch_date': '2025-10-20T...',
                'issues': [...],
                'branches': [...],
                'merge_requests': [...],
                'all_commits': [...],
                'commits_by_mr': {...},
                'pipelines': [...],
                'artifacts': [...],
                'stats': {...}
            }
        """
        print(f"\n{'='*80}")
        print(f"FETCHING: {project_name}")
        print(f"{'='*80}")

        # Reset request counter for this project
        self.client.reset_request_count()

        # Initialize project data structure
        project_data = {
            'project_id': project_id,
            'project_name': project_name,
            'fetch_date': datetime.now().isoformat(),
            'issues': [],
            'branches': [],
            'merge_requests': [],
            'all_commits': [],
            'commits_by_mr': {},
            'pipelines': [],
            'artifacts': [],
            'coverage': [],
            'stats': {}
        }

        # 1. Fetch issues
        project_data['issues'] = self.issues_fetcher.fetch(project_id)

        # 2. Fetch branches
        project_data['branches'] = self.branches_fetcher.fetch(project_id)

        # 3. Fetch merge requests
        project_data['merge_requests'] = self.merge_requests_fetcher.fetch(project_id)

        # 4. Fetch commits (both types)
        commits_data = self.commits_fetcher.fetch(
            project_id,
            project_data['merge_requests']
        )
        project_data['all_commits'] = commits_data['all_commits']
        project_data['commits_by_mr'] = commits_data['commits_by_mr']

        # 5. Fetch pipelines with jobs
        project_data['pipelines'] = self.pipelines_fetcher.fetch(project_id)

        # 6. Fetch artifacts metadata (from pipelines/jobs)
        project_data['artifacts'] = self.artifacts_fetcher.fetch(
            project_id,
            project_data['pipelines']
        )

        # 7. Extract coverage data (simple mode - identify coverage artifacts)
        project_data['coverage'] = self.coverage_extractor.extract_simple_from_artifacts(
            project_data['artifacts']
        )

        # 8. Calculate statistics
        project_data['stats'] = self._calculate_stats(project_data)

        # Print summary
        self._print_summary(project_data)

        return project_data

    def _calculate_stats(self, project_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate basic statistics for the project

        Args:
            project_data: Complete project data

        Returns:
            Dictionary with statistics
        """
        # Calculate artifact statistics
        artifact_stats = self.artifacts_fetcher.get_artifact_statistics(
            project_data['artifacts']
        )

        return {
            'total_issues': len(project_data['issues']),
            'total_branches': len(project_data['branches']),
            'total_merge_requests': len(project_data['merge_requests']),
            'total_commits': len(project_data['all_commits']),
            'total_pipelines': len(project_data['pipelines']),
            'total_artifacts': len(project_data['artifacts']),
            'artifact_size_mb': artifact_stats.get('total_size_mb', 0.0),
            'jobs_with_coverage': len(project_data['coverage']),
            'total_api_requests': self.client.get_request_count()
        }

    def _print_summary(self, project_data: Dict[str, Any]) -> None:
        """
        Print summary of fetched data

        Args:
            project_data: Complete project data
        """
        stats = project_data['stats']

        print(f"\n  Summary:")
        print(f"    Issues: {stats['total_issues']}")
        print(f"    Branches: {stats['total_branches']}")
        print(f"    Merge Requests: {stats['total_merge_requests']}")
        print(f"    Total Commits: {stats['total_commits']}")
        print(f"    Pipelines: {stats['total_pipelines']}")
        print(f"    Artifacts: {stats['total_artifacts']} ({stats['artifact_size_mb']:.1f} MB)")
        print(f"    Jobs with Coverage: {stats['jobs_with_coverage']}")
        print(f"    API Requests Made: {stats['total_api_requests']}")

    def save(self, project_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save project data to files

        Args:
            project_data: Complete project data
            output_dir: Output directory
        """
        files = FileManager.save_project_data(project_data, output_dir)

        print(f"\n  Saved 9 files to {output_dir}/{project_data['project_name']}/")
        for file_path in files:
            import os
            filename = os.path.basename(file_path)
            size = FileManager.get_file_size(file_path)
            size_str = FileManager.format_size(size)
            print(f"    - {filename:<25} ({size_str:>10})")
