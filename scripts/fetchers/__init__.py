"""
Fetchers package - Modular data fetchers for GitLab

Provides specialized fetchers for each data type:
- IssuesFetcher: Fetch issues
- BranchesFetcher: Fetch branches
- MergeRequestsFetcher: Fetch merge requests
- CommitsFetcher: Fetch commits (all and by MR)
- PipelinesFetcher: Fetch pipelines with jobs
- ArtifactsFetcher: Fetch artifact metadata from jobs
- CoverageExtractor: Extract coverage data from artifacts
- ProjectFetcher: Orchestrates all fetchers
"""

from .base_fetcher import BaseFetcher
from .issues_fetcher import IssuesFetcher
from .branches_fetcher import BranchesFetcher
from .merge_requests_fetcher import MergeRequestsFetcher
from .commits_fetcher import CommitsFetcher
from .pipelines_fetcher import PipelinesFetcher
from .artifacts_fetcher import ArtifactsFetcher
from .coverage_extractor import CoverageExtractor
from .project_fetcher import ProjectFetcher

__all__ = [
    'BaseFetcher',
    'IssuesFetcher',
    'BranchesFetcher',
    'MergeRequestsFetcher',
    'CommitsFetcher',
    'PipelinesFetcher',
    'ArtifactsFetcher',
    'CoverageExtractor',
    'ProjectFetcher'
]
