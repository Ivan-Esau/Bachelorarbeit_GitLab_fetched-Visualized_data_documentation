"""
GitLab API Client - Base class for API communication

Handles:
- Authentication
- Pagination
- Rate limiting
- Error handling
"""

import requests
import time
from typing import List, Dict, Any, Optional


class GitLabClient:
    """
    Base GitLab API client with pagination and rate limiting

    Provides common functionality for all GitLab API interactions:
    - Automatic pagination
    - Rate limiting (50ms between requests)
    - Error handling
    - Request counting
    """

    def __init__(self, gitlab_url: str, private_token: str):
        """
        Initialize GitLab API client

        Args:
            gitlab_url: GitLab instance URL (e.g., 'https://gitlab.example.com')
            private_token: GitLab personal access token
        """
        self.gitlab_url = gitlab_url.rstrip('/')
        self.api_base = f'{self.gitlab_url}/api/v4'
        self.headers = {'PRIVATE-TOKEN': private_token}
        self.request_count = 0

    def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generic paginated GET request - fetches ALL pages

        Args:
            endpoint: API endpoint (e.g., '/projects/123/issues')
            params: Query parameters (optional)
            per_page: Items per page (default: 100, max: 100)

        Returns:
            List of all items from all pages

        Example:
            issues = client.get_paginated('/projects/123/issues', {'state': 'all'})
        """
        all_items = []
        page = 1
        params = params or {}

        while True:
            params['page'] = page
            params['per_page'] = per_page

            try:
                url = f'{self.api_base}{endpoint}'
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                response.raise_for_status()
                self.request_count += 1

                items = response.json()
                if not items:
                    break

                all_items.extend(items)

                # Check if there are more pages
                if len(items) < per_page:
                    break

                page += 1
                time.sleep(0.05)  # Rate limiting: 50ms between requests

            except requests.exceptions.RequestException as e:
                print(f"    Warning: Error fetching page {page} from {endpoint}: {e}")
                break

        return all_items

    def get_single(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Single GET request (not paginated)

        Args:
            endpoint: API endpoint
            params: Query parameters (optional)

        Returns:
            Response JSON or None if error

        Example:
            pipeline = client.get_single('/projects/123/pipelines/456')
        """
        try:
            url = f'{self.api_base}{endpoint}'
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            self.request_count += 1
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"    Warning: Error fetching {endpoint}: {e}")
            return None

    def get_raw(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[bytes]:
        """
        Single GET request for raw content (not JSON)

        Used for downloading artifact files (XML, binary, etc.) that should
        NOT be parsed as JSON.

        Args:
            endpoint: API endpoint
            params: Query parameters (optional)

        Returns:
            Response content as bytes or None if error

        Example:
            xml_data = client.get_raw('/projects/123/jobs/456/artifacts/coverage.xml')
        """
        try:
            url = f'{self.api_base}{endpoint}'
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            self.request_count += 1

            # Return raw content (bytes), not JSON
            return response.content if response.content else None

        except requests.exceptions.RequestException as e:
            print(f"    Warning: Error fetching {endpoint}: {e}")
            return None

    def reset_request_count(self):
        """Reset the request counter (useful for per-project tracking)"""
        self.request_count = 0

    def get_request_count(self) -> int:
        """Get the current request count"""
        return self.request_count
