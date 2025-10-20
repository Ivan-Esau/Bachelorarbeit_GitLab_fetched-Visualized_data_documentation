"""
Pipelines Fetcher - Fetch pipelines and jobs from GitLab
"""

from typing import List, Dict, Any, Optional
from .base_fetcher import BaseFetcher


class PipelinesFetcher(BaseFetcher):
    """Fetches pipelines with their jobs for a project"""

    def fetch_pipelines_basic(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch basic pipeline information

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of basic pipeline data

        API: GET /projects/:id/pipelines
        """
        print(f"  Fetching pipelines...")

        endpoint = f'/projects/{self._quote_project_id(project_id)}/pipelines'
        params = {
            'order_by': 'id',
            'sort': 'asc'
        }

        pipelines = self.client.get_paginated(endpoint, params)
        print(f"    Found {len(pipelines)} pipelines")

        return pipelines

    def fetch_pipeline_details(
        self,
        project_id: str,
        pipeline_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a single pipeline

        Args:
            project_id: GitLab project ID or path
            pipeline_id: Pipeline ID

        Returns:
            Pipeline details or None if error

        API: GET /projects/:id/pipelines/:pipeline_id
        """
        endpoint = f'/projects/{self._quote_project_id(project_id)}/pipelines/{pipeline_id}'
        return self.client.get_single(endpoint)

    def fetch_pipeline_jobs(self, project_id: str, pipeline_id: int) -> List[Dict[str, Any]]:
        """
        Fetch all jobs for a pipeline

        Args:
            project_id: GitLab project ID or path
            pipeline_id: Pipeline ID

        Returns:
            List of jobs

        API: GET /projects/:id/pipelines/:pipeline_id/jobs
        """
        endpoint = f'/projects/{self._quote_project_id(project_id)}/pipelines/{pipeline_id}/jobs'
        return self.client.get_paginated(endpoint)

    def fetch(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Fetch complete pipeline data with jobs

        Args:
            project_id: GitLab project ID or path

        Returns:
            List of pipelines with their jobs embedded

        Example:
            [
                {
                    'id': 123,
                    'status': 'success',
                    ...
                    'jobs': [job1, job2, ...]
                }
            ]
        """
        # Fetch basic pipeline info
        pipelines_basic = self.fetch_pipelines_basic(project_id)

        # Fetch detailed data for each pipeline
        print(f"  Fetching pipeline details and jobs...")
        detailed_pipelines = []

        for i, pipeline in enumerate(pipelines_basic, 1):
            if i % 10 == 0 or i == 1:
                print(f"    Processing pipeline {i}/{len(pipelines_basic)} (ID: {pipeline['id']})")

            pipeline_id = pipeline['id']

            # Get detailed pipeline info
            details = self.fetch_pipeline_details(project_id, pipeline_id)
            if not details:
                continue

            # Get all jobs for this pipeline
            jobs = self.fetch_pipeline_jobs(project_id, pipeline_id)

            # Combine pipeline data with jobs
            detailed_pipelines.append({
                'id': pipeline_id,
                'iid': pipeline.get('iid'),
                'status': pipeline.get('status'),
                'ref': pipeline.get('ref'),
                'sha': pipeline.get('sha'),
                'created_at': pipeline.get('created_at'),
                'updated_at': pipeline.get('updated_at'),
                'started_at': details.get('started_at') if details else None,
                'finished_at': details.get('finished_at') if details else None,
                'duration': details.get('duration') if details else None,
                'queued_duration': details.get('queued_duration') if details else None,
                'web_url': pipeline.get('web_url'),
                'jobs': jobs
            })

        return detailed_pipelines
