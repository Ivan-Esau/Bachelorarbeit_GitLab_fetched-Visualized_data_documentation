"""
Extract All Coverage - Extract coverage data from all projects using existing pipeline data

Reads pipelines.json files and extracts coverage data from JaCoCo artifacts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from core import GitLabClient, load_project_config, FileManager
from fetchers import CoverageExtractor
import json
from datetime import datetime

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def extract_coverage_from_pipelines(project_id: str, project_name: str, client: GitLabClient):
    """Extract coverage data from pipeline data"""

    print(f"\n{'='*80}")
    print(f"EXTRACTING COVERAGE: {project_name}")
    print(f"{'='*80}")

    # Load pipelines from saved data
    data_dir = os.path.join('..', 'data_raw', project_name)
    pipelines_file = os.path.join(data_dir, 'pipelines.json')

    if not os.path.exists(pipelines_file):
        print(f"  [SKIP] Pipelines file not found")
        return None

    with open(pipelines_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    print(f"  Loaded {len(pipelines)} pipelines")

    # Extract JaCoCo artifacts from job artifacts
    jacoco_artifacts = []
    for pipeline in pipelines:
        pipeline_id = pipeline.get('id')
        for job in pipeline.get('jobs', []):
            for artifact in job.get('artifacts', []):
                if artifact.get('file_type') == 'jacoco':
                    jacoco_artifacts.append({
                        'job_id': job['id'],
                        'job_name': job.get('name', 'unknown'),
                        'job_stage': job.get('stage', 'unknown'),
                        'pipeline_id': pipeline_id,
                        'artifact_id': f"{job['id']}_jacoco",
                        'filename': artifact.get('filename', 'jacoco-coverage.xml.gz'),
                        'size': artifact.get('size', 0),
                        'file_type': 'jacoco',
                        'file_format': artifact.get('file_format', 'gzip')
                    })

    print(f"  Found {len(jacoco_artifacts)} JaCoCo coverage artifacts")

    if not jacoco_artifacts:
        print(f"  [SKIP] No coverage artifacts found")
        return None

    # Extract coverage
    coverage_extractor = CoverageExtractor(client)

    print(f"  Attempting to extract coverage data...")

    coverage_data = []
    successful = 0
    expired = 0
    failed = 0

    for artifact in jacoco_artifacts:
        try:
            result = coverage_extractor._download_and_parse_coverage(
                project_id,
                artifact
            )

            if result:
                coverage_data.append(result)
                status = result.get('parse_status', 'unknown')
                if status == 'success':
                    successful += 1
                elif status == 'expired':
                    expired += 1
                else:
                    failed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"    Warning: Job {artifact['job_id']} failed: {e}")
            failed += 1

    print(f"    Results: {successful} successful, {expired} expired, {failed} failed")

    # Calculate statistics
    if successful > 0:
        successful_coverage = [c for c in coverage_data if c.get('parse_status') == 'success']
        coverages = [c['coverage_percentage'] for c in successful_coverage]
        avg_coverage = sum(coverages) / len(coverages)
        min_coverage = min(coverages)
        max_coverage = max(coverages)

        print(f"  Coverage Statistics:")
        print(f"    Average: {avg_coverage:.2f}%")
        print(f"    Min: {min_coverage:.2f}%")
        print(f"    Max: {max_coverage:.2f}%")

    # Save coverage data
    coverage_file = os.path.join(data_dir, 'coverage.json')
    with open(coverage_file, 'w', encoding='utf-8') as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved coverage data to: {coverage_file}")

    return {
        'project_name': project_name,
        'total_jacoco_artifacts': len(jacoco_artifacts),
        'successful_extractions': successful,
        'expired_artifacts': expired,
        'failed_extractions': failed,
        'coverage_data': coverage_data
    }


def main():
    """Extract coverage from all projects"""

    print("="*80)
    print("COVERAGE EXTRACTION - ALL PROJECTS")
    print("="*80)

    # Create client
    gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.nibbler.fh-swf.de/')
    gitlab_token = os.getenv('GITLAB_TOKEN')

    if not gitlab_token:
        print("[ERROR] GITLAB_TOKEN environment variable not set")
        return 1

    client = GitLabClient(gitlab_url, gitlab_token)

    # Load projects (uses default path '../../config_projects.json' relative to core/)
    projects = load_project_config()

    if not projects:
        print("[ERROR] No projects found in config")
        return 1

    print(f"\nProcessing {len(projects)} projects...\n")

    all_results = []

    for project_id, project_name in projects:
        try:
            result = extract_coverage_from_pipelines(project_id, project_name, client)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {project_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*80}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*80}\n")

    total_artifacts = sum(r['total_jacoco_artifacts'] for r in all_results)
    total_successful = sum(r['successful_extractions'] for r in all_results)
    total_expired = sum(r['expired_artifacts'] for r in all_results)
    total_failed = sum(r['failed_extractions'] for r in all_results)

    print(f"Total JaCoCo artifacts found: {total_artifacts}")
    print(f"Successful extractions: {total_successful}")
    print(f"Expired artifacts: {total_expired}")
    print(f"Failed extractions: {total_failed}")

    if total_successful > 0:
        # Calculate overall coverage statistics
        all_successful_coverage = []
        for result in all_results:
            successful_coverage = [
                c for c in result['coverage_data']
                if c.get('parse_status') == 'success'
            ]
            all_successful_coverage.extend(successful_coverage)

        if all_successful_coverage:
            coverages = [c['coverage_percentage'] for c in all_successful_coverage]
            print(f"\nOverall Coverage Statistics:")
            print(f"  Average: {sum(coverages) / len(coverages):.2f}%")
            print(f"  Min: {min(coverages):.2f}%")
            print(f"  Max: {max(coverages):.2f}%")
            print(f"  Above 80%: {len([c for c in coverages if c >= 80])}")
            print(f"  Above 90%: {len([c for c in coverages if c >= 90])}")

    # Save summary
    summary_file = os.path.join('..', 'data_raw', 'coverage_extraction_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'total_projects': len(all_results),
            'total_artifacts': total_artifacts,
            'successful_extractions': total_successful,
            'expired_artifacts': total_expired,
            'failed_extractions': total_failed,
            'projects': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Summary saved to: {summary_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
