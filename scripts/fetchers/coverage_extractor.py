"""
Coverage Extractor - Download and parse coverage artifacts

Extracts actual coverage percentages from JaCoCo and Cobertura XML artifacts.
Provides quantitative metrics for research analysis.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import urllib.parse
import gzip
import io


class CoverageExtractor:
    """
    Extract coverage data from artifacts

    Downloads coverage artifacts (JaCoCo and Cobertura XML) and parses them
    to extract actual coverage percentages and metrics.
    """

    def __init__(self, client):
        """
        Initialize coverage extractor

        Args:
            client: GitLabClient instance
        """
        self.client = client

    def _quote_project_id(self, project_id: str) -> str:
        """
        URL-encode project ID for API requests

        Args:
            project_id: Project ID (can contain slashes)

        Returns:
            URL-encoded project ID
        """
        return urllib.parse.quote(project_id, safe='')

    def extract_from_artifacts(
        self,
        project_id: str,
        artifacts: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Extract coverage data from coverage artifacts

        Args:
            project_id: GitLab project ID
            artifacts: List of artifact metadata (from ArtifactsFetcher)

        Returns:
            List of coverage data dictionaries

        Example output:
        [
            {
                "job_id": 5865,
                "job_name": "test",
                "job_stage": "test",
                "pipeline_id": 1234,
                "artifact_id": "5865_archive",
                "coverage_percentage": 87.5,
                "lines_covered": 350,
                "lines_total": 400,
                "line_rate": 0.875,
                "branches_covered": 45,
                "branches_total": 50,
                "branch_rate": 0.9,
                "complexity": 125,
                "parse_status": "success",
                "parse_error": null
            }
        ]
        """
        # Filter for coverage artifacts (JaCoCo or Cobertura)
        coverage_artifacts = [
            a for a in artifacts
            if a.get('file_type') in ['jacoco', 'cobertura'] or
               'coverage' in a.get('filename', '').lower()
        ]

        if not coverage_artifacts:
            print("    No coverage artifacts found")
            return []

        print(f"  Extracting coverage from {len(coverage_artifacts)} artifacts...")

        all_coverage_data = []
        successful = 0
        failed = 0

        for artifact in coverage_artifacts:
            try:
                # Download and parse artifact
                coverage_data = self._download_and_parse_coverage(
                    project_id,
                    artifact
                )

                if coverage_data:
                    all_coverage_data.append(coverage_data)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"    Warning: Failed to extract coverage from job {artifact['job_id']}: {e}")
                # Add failed entry
                all_coverage_data.append({
                    'job_id': artifact['job_id'],
                    'job_name': artifact['job_name'],
                    'job_stage': artifact['job_stage'],
                    'pipeline_id': artifact['pipeline_id'],
                    'artifact_id': artifact['artifact_id'],
                    'parse_status': 'error',
                    'parse_error': str(e)
                })
                failed += 1

        print(f"    Successfully extracted: {successful}, Failed: {failed}")

        return all_coverage_data

    def _download_and_parse_coverage(
        self,
        project_id: str,
        artifact: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Download coverage artifact and parse it

        Args:
            project_id: GitLab project ID
            artifact: Artifact metadata (contains file_type, filename)

        Returns:
            Coverage data dictionary or None
        """
        job_id = artifact['job_id']
        file_type = artifact.get('file_type')
        filename = artifact.get('filename', '')

        # Determine artifact path based on type
        if file_type == 'jacoco':
            # JaCoCo coverage file path
            artifact_path = 'target/site/jacoco/jacoco.xml'
        elif file_type == 'cobertura':
            # Cobertura coverage file path
            artifact_path = 'coverage.xml'
        else:
            return None

        quoted_project_id = self._quote_project_id(project_id)
        artifact_url = f'/projects/{quoted_project_id}/jobs/{job_id}/artifacts/{artifact_path}'

        try:
            # Download specific artifact file (raw content, not JSON)
            artifact_data = self.client.get_raw(artifact_url)

            if not artifact_data:
                # Return expired status
                return {
                    'job_id': artifact['job_id'],
                    'job_name': artifact['job_name'],
                    'job_stage': artifact['job_stage'],
                    'pipeline_id': artifact['pipeline_id'],
                    'artifact_id': artifact['artifact_id'],
                    'coverage_format': file_type,
                    'parse_status': 'expired',
                    'parse_error': 'No data returned - artifact likely expired'
                }

            # Decompress if gzipped (GitLab often compresses artifacts)
            # artifact_data is now always bytes from get_raw()
            # Try to decompress gzip
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(artifact_data)) as gz:
                    xml_content = gz.read().decode('utf-8')
            except (OSError, gzip.BadGzipFile):
                # Not gzipped, treat as plain bytes
                xml_content = artifact_data.decode('utf-8')

            # Parse based on format
            if file_type == 'jacoco':
                coverage_data = self._parse_jacoco_xml(xml_content)
            else:
                coverage_data = self._parse_cobertura_xml(xml_content)

            if coverage_data:
                # Add artifact metadata
                coverage_data.update({
                    'job_id': artifact['job_id'],
                    'job_name': artifact['job_name'],
                    'job_stage': artifact['job_stage'],
                    'pipeline_id': artifact['pipeline_id'],
                    'artifact_id': artifact['artifact_id'],
                    'coverage_format': file_type,
                    'parse_status': 'success',
                    'parse_error': None
                })

                return coverage_data

        except Exception as e:
            # Artifact might be expired or inaccessible
            if '404' in str(e) or 'not found' in str(e).lower() or 'expired' in str(e).lower():
                # Add entry for expired artifact
                return {
                    'job_id': artifact['job_id'],
                    'job_name': artifact['job_name'],
                    'job_stage': artifact['job_stage'],
                    'pipeline_id': artifact['pipeline_id'],
                    'artifact_id': artifact['artifact_id'],
                    'coverage_format': file_type,
                    'parse_status': 'expired',
                    'parse_error': 'Artifact expired or not found'
                }
            raise  # Re-raise other errors

        return None

    def _parse_cobertura_xml(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse Cobertura XML format

        Args:
            xml_content: XML string

        Returns:
            Coverage metrics dictionary

        Cobertura XML format:
        <coverage line-rate="0.875" branch-rate="0.9" lines-covered="350"
                  lines-valid="400" branches-covered="45" branches-valid="50"
                  complexity="125" timestamp="...">
        """
        try:
            root = ET.fromstring(xml_content)

            # Extract attributes from <coverage> root element
            line_rate = float(root.get('line-rate', 0))
            branch_rate = float(root.get('branch-rate', 0))
            lines_covered = int(root.get('lines-covered', 0))
            lines_valid = int(root.get('lines-valid', 0))
            branches_covered = int(root.get('branches-covered', 0))
            branches_valid = int(root.get('branches-valid', 0))
            complexity = int(root.get('complexity', 0))

            # Calculate percentage
            coverage_percentage = round(line_rate * 100, 2)

            return {
                'coverage_percentage': coverage_percentage,
                'lines_covered': lines_covered,
                'lines_total': lines_valid,
                'line_rate': line_rate,
                'branches_covered': branches_covered,
                'branches_total': branches_valid,
                'branch_rate': branch_rate,
                'complexity': complexity
            }

        except ET.ParseError as e:
            # Not valid XML
            return None
        except Exception as e:
            # Other parsing errors
            return None

    def _parse_jacoco_xml(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JaCoCo XML format

        Args:
            xml_content: XML string

        Returns:
            Coverage metrics dictionary

        JaCoCo XML format:
        <report name="...">
          <counter type="INSTRUCTION" missed="X" covered="Y"/>
          <counter type="BRANCH" missed="X" covered="Y"/>
          <counter type="LINE" missed="X" covered="Y"/>
          <counter type="COMPLEXITY" missed="X" covered="Y"/>
          <counter type="METHOD" missed="X" covered="Y"/>
          <counter type="CLASS" missed="X" covered="Y"/>
        </report>
        """
        try:
            root = ET.fromstring(xml_content)

            # Find counters at report level (overall metrics)
            counters = {}
            for counter in root.findall('.//counter[@type]'):
                counter_type = counter.get('type')
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                counters[counter_type] = {
                    'missed': missed,
                    'covered': covered,
                    'total': missed + covered
                }

            # Calculate coverage percentages
            lines = counters.get('LINE', {'missed': 0, 'covered': 0, 'total': 0})
            branches = counters.get('BRANCH', {'missed': 0, 'covered': 0, 'total': 0})
            instructions = counters.get('INSTRUCTION', {'missed': 0, 'covered': 0, 'total': 0})
            complexity = counters.get('COMPLEXITY', {'missed': 0, 'covered': 0, 'total': 0})

            # Line coverage percentage
            if lines['total'] > 0:
                line_coverage = (lines['covered'] / lines['total']) * 100
            else:
                line_coverage = 0.0

            # Branch coverage percentage
            if branches['total'] > 0:
                branch_coverage = (branches['covered'] / branches['total']) * 100
            else:
                branch_coverage = 0.0

            return {
                'coverage_percentage': round(line_coverage, 2),
                'lines_covered': lines['covered'],
                'lines_total': lines['total'],
                'line_rate': round(line_coverage / 100, 4) if lines['total'] > 0 else 0.0,
                'branches_covered': branches['covered'],
                'branches_total': branches['total'],
                'branch_rate': round(branch_coverage / 100, 4) if branches['total'] > 0 else 0.0,
                'instructions_covered': instructions['covered'],
                'instructions_total': instructions['total'],
                'complexity': complexity['total']
            }

        except ET.ParseError as e:
            # Not valid XML
            return None
        except Exception as e:
            # Other parsing errors
            return None

    def get_coverage_statistics(
        self,
        coverage_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about coverage data

        Args:
            coverage_data: List of coverage data

        Returns:
            Statistics dictionary
        """
        if not coverage_data:
            return {
                'total_jobs_with_coverage': 0,
                'average_coverage': 0.0,
                'min_coverage': 0.0,
                'max_coverage': 0.0,
                'median_coverage': 0.0,
                'jobs_above_80_percent': 0,
                'jobs_above_90_percent': 0
            }

        # Filter only successful parses
        successful = [
            c for c in coverage_data
            if c.get('parse_status') == 'success' and
               c.get('coverage_percentage') is not None
        ]

        if not successful:
            return {
                'total_jobs_with_coverage': 0,
                'total_attempted': len(coverage_data),
                'parse_failures': len(coverage_data),
                'average_coverage': 0.0
            }

        coverages = [c['coverage_percentage'] for c in successful]

        return {
            'total_jobs_with_coverage': len(successful),
            'total_attempted': len(coverage_data),
            'parse_failures': len(coverage_data) - len(successful),
            'average_coverage': round(sum(coverages) / len(coverages), 2),
            'min_coverage': min(coverages),
            'max_coverage': max(coverages),
            'median_coverage': sorted(coverages)[len(coverages) // 2],
            'jobs_above_80_percent': len([c for c in coverages if c >= 80]),
            'jobs_above_90_percent': len([c for c in coverages if c >= 90]),
            'total_lines_covered': sum(c['lines_covered'] for c in successful),
            'total_lines': sum(c['lines_total'] for c in successful)
        }

    def extract_simple_from_artifacts(
        self,
        artifacts: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Simple extraction - identify coverage artifacts WITHOUT downloading

        This is faster and doesn't consume API quota, but only identifies
        which jobs have coverage, not the actual percentages.

        Args:
            artifacts: List of artifact metadata

        Returns:
            List of coverage artifact info (no percentages)
        """
        coverage_artifacts = [
            a for a in artifacts
            if a.get('file_type') in ['jacoco', 'cobertura'] or
               'coverage' in a.get('filename', '').lower()
        ]

        return [
            {
                'job_id': a['job_id'],
                'job_name': a['job_name'],
                'job_stage': a['job_stage'],
                'pipeline_id': a['pipeline_id'],
                'artifact_id': a['artifact_id'],
                'artifact_filename': a['filename'],
                'artifact_size': a['size'],
                'coverage_format': a.get('file_type', 'unknown'),
                'has_coverage_artifact': True,
                'coverage_percentage': None,  # Would need to download
                'note': 'Coverage artifact exists but not downloaded/parsed'
            }
            for a in coverage_artifacts
        ]
