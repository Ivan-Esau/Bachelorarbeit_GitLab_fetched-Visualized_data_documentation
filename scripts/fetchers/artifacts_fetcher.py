"""
Artifacts Fetcher - Fetch artifact metadata from GitLab jobs

Retrieves metadata about all artifacts created by pipeline jobs.
Does NOT download artifact content (to save space/time), only metadata.
"""

from typing import List, Dict, Any
from .base_fetcher import BaseFetcher


class ArtifactsFetcher(BaseFetcher):
    """
    Fetcher for job artifact metadata

    Collects information about artifacts created by pipeline jobs:
    - Artifact filename, size, type
    - Creation and expiration dates
    - Which job/pipeline created it
    """

    def fetch(self, project_id: str, pipelines: List[Dict]) -> List[Dict]:
        """
        Fetch artifact metadata for all jobs across all pipelines

        Args:
            project_id: GitLab project ID (can be 'namespace/project')
            pipelines: List of pipeline objects (must include 'id' and 'jobs')

        Returns:
            List of artifact metadata dictionaries

        Example output:
        [
            {
                "artifact_id": "unique_id",
                "filename": "coverage.xml",
                "size": 12345,
                "file_type": "cobertura",
                "created_at": "2024-01-15T10:30:00Z",
                "expires_at": "2024-02-15T10:30:00Z",
                "job_id": 123,
                "job_name": "test",
                "job_stage": "test",
                "job_status": "success",
                "pipeline_id": 456,
                "pipeline_status": "success"
            }
        ]
        """
        all_artifacts = []
        jobs_with_artifacts = 0
        total_jobs = 0

        print(f"  Collecting artifact metadata from {len(pipelines)} pipelines...")

        for pipeline in pipelines:
            pipeline_id = pipeline.get('id')
            pipeline_status = pipeline.get('status')

            # Each pipeline should have 'jobs' embedded (from PipelinesFetcher)
            jobs = pipeline.get('jobs', [])

            for job in jobs:
                total_jobs += 1

                # Check if job has artifacts
                artifacts_file = job.get('artifacts_file')
                artifacts = job.get('artifacts', [])

                # Method 1: Check artifacts_file (older API format)
                # Note: Don't use elif - both can exist simultaneously
                if artifacts_file:
                    artifact_metadata = {
                        'artifact_id': f"{job['id']}_archive",
                        'filename': artifacts_file.get('filename', 'artifacts.zip'),
                        'size': artifacts_file.get('size', 0),
                        'file_type': 'archive',  # Default type
                        'created_at': job.get('created_at'),
                        'expires_at': job.get('artifacts_expire_at'),
                        'job_id': job['id'],
                        'job_name': job.get('name', 'unknown'),
                        'job_stage': job.get('stage', 'unknown'),
                        'job_status': job.get('status', 'unknown'),
                        'pipeline_id': pipeline_id,
                        'pipeline_status': pipeline_status
                    }
                    all_artifacts.append(artifact_metadata)

                # Method 2: Check artifacts array (newer API format)
                # Extract individual artifacts (jacoco, junit, etc.)
                if artifacts and len(artifacts) > 0:
                    for idx, artifact in enumerate(artifacts):
                        artifact_metadata = {
                            'artifact_id': f"{job['id']}_artifact_{idx}",
                            'filename': artifact.get('filename', f"artifact_{idx}"),
                            'size': artifact.get('size', 0),
                            'file_type': artifact.get('file_type', 'unknown'),
                            'file_format': artifact.get('file_format', 'unknown'),
                            'created_at': job.get('created_at'),
                            'expires_at': job.get('artifacts_expire_at'),
                            'job_id': job['id'],
                            'job_name': job.get('name', 'unknown'),
                            'job_stage': job.get('stage', 'unknown'),
                            'job_status': job.get('status', 'unknown'),
                            'pipeline_id': pipeline_id,
                            'pipeline_status': pipeline_status
                        }
                        all_artifacts.append(artifact_metadata)

                # Count jobs with artifacts (either format)
                if artifacts_file or (artifacts and len(artifacts) > 0):
                    jobs_with_artifacts += 1

        print(f"    Found {len(all_artifacts)} artifacts from {jobs_with_artifacts}/{total_jobs} jobs")

        return all_artifacts

    def fetch_with_download_urls(self, project_id: str, pipelines: List[Dict]) -> List[Dict]:
        """
        Fetch artifact metadata with download URLs (for future use)

        Note: This method constructs download URLs but does NOT download artifacts.
        Use this if you need to download artifacts later.

        Args:
            project_id: GitLab project ID
            pipelines: List of pipeline objects

        Returns:
            List of artifact metadata with download URLs
        """
        artifacts = self.fetch(project_id, pipelines)

        # Add download URLs
        quoted_project_id = self._quote_project_id(project_id)

        for artifact in artifacts:
            job_id = artifact['job_id']
            artifact['download_url'] = (
                f"/projects/{quoted_project_id}/jobs/{job_id}/artifacts"
            )

        return artifacts

    def get_artifact_statistics(self, artifacts: List[Dict]) -> Dict[str, Any]:
        """
        Calculate statistics about artifacts

        Args:
            artifacts: List of artifact metadata

        Returns:
            Dictionary with statistics
        """
        if not artifacts:
            return {
                'total_artifacts': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0.0,
                'by_type': {},
                'by_stage': {},
                'by_status': {}
            }

        total_size = sum(a.get('size', 0) for a in artifacts)

        # Count by type
        by_type = {}
        for artifact in artifacts:
            file_type = artifact.get('file_type', 'unknown')
            by_type[file_type] = by_type.get(file_type, 0) + 1

        # Count by stage
        by_stage = {}
        for artifact in artifacts:
            stage = artifact.get('job_stage', 'unknown')
            by_stage[stage] = by_stage.get(stage, 0) + 1

        # Count by job status
        by_status = {}
        for artifact in artifacts:
            status = artifact.get('job_status', 'unknown')
            by_status[status] = by_status.get(status, 0) + 1

        return {
            'total_artifacts': len(artifacts),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_type': by_type,
            'by_stage': by_stage,
            'by_status': by_status
        }
