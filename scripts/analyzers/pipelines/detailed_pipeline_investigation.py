"""
Detailed Pipeline Investigation Script
Project-by-Project, Branch-by-Branch Analysis

This script performs a comprehensive investigation of pipeline failures, cancellations,
and stuck pipelines, analyzing each project and branch individually to identify:
- WHEN pipelines got stuck (timeline analysis)
- WHY pipelines were cancelled (stage/job analysis)
- WHAT patterns exist per branch (success/failure trends)
- HOW long pipelines took before getting cancelled

Supports letter-based project structure (a, b, c, etc.)

Author: Data Analysis System
Date: 2025-10-21
Updated: 2025-10-22 (Letter-based structure support)
"""

import json
import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from scripts.core.config_loader import get_projects_by_letter, get_project_mapping
    from scripts.core.path_helpers import get_data_file, get_analysis_file, ensure_dir
except ModuleNotFoundError:
    from core.config_loader import get_projects_by_letter, get_project_mapping
    from core.path_helpers import get_data_file, get_analysis_file, ensure_dir


def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime."""
    if not ts_str:
        return None
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))


def analyze_single_pipeline(pipeline: Dict, project_key: str) -> Dict:
    """
    Deeply analyze a single pipeline to extract all relevant information.
    """
    # Basic pipeline info
    analysis = {
        'project': project_key,
        'pipeline_id': pipeline['id'],
        'pipeline_iid': pipeline['iid'],
        'branch': pipeline['ref'],
        'status': pipeline['status'],
        'sha': pipeline['sha'],

        # Timestamps
        'created_at': pipeline['created_at'],
        'started_at': pipeline.get('started_at'),
        'finished_at': pipeline.get('finished_at'),

        # Durations
        'duration_seconds': pipeline.get('duration'),
        'queued_duration_seconds': pipeline.get('queued_duration'),

        # Job information
        'compile_job_status': None,
        'compile_job_duration': None,
        'compile_job_started': None,
        'compile_job_finished': None,

        'test_job_status': None,
        'test_job_duration': None,
        'test_job_started': None,
        'test_job_finished': None,

        # Analysis flags
        'compile_failed': False,
        'test_failed': False,
        'compile_cancelled': False,
        'test_cancelled': False,
        'stuck': False,
        'stuck_reason': None,

        # Commit info
        'commit_title': None,
        'commit_message': None,
    }

    # Analyze jobs
    for job in pipeline.get('jobs', []):
        stage = job.get('stage', '')

        # Get commit info from first job
        if analysis['commit_title'] is None and 'commit' in job:
            analysis['commit_title'] = job['commit'].get('title', '')
            analysis['commit_message'] = job['commit'].get('message', '')

        if stage == 'compile':
            analysis['compile_job_status'] = job['status']
            analysis['compile_job_duration'] = job.get('duration')
            analysis['compile_job_started'] = job.get('started_at')
            analysis['compile_job_finished'] = job.get('finished_at')

            if job['status'] == 'failed':
                analysis['compile_failed'] = True
            elif job['status'] == 'canceled':
                analysis['compile_cancelled'] = True

        elif stage == 'test':
            analysis['test_job_status'] = job['status']
            analysis['test_job_duration'] = job.get('duration')
            analysis['test_job_started'] = job.get('started_at')
            analysis['test_job_finished'] = job.get('finished_at')

            if job['status'] == 'failed':
                analysis['test_failed'] = True
            elif job['status'] == 'canceled':
                analysis['test_cancelled'] = True

    # Determine if stuck (duration > 300 seconds = 5 minutes)
    if pipeline.get('duration'):
        if pipeline['duration'] > 300:
            analysis['stuck'] = True

            # Determine stuck reason
            if analysis['test_cancelled']:
                analysis['stuck_reason'] = 'test_timeout'
            elif analysis['compile_cancelled']:
                analysis['stuck_reason'] = 'compile_timeout'
            else:
                analysis['stuck_reason'] = 'unknown_timeout'

    # Calculate derived metrics
    if analysis['duration_seconds']:
        analysis['duration_minutes'] = round(analysis['duration_seconds'] / 60, 2)
    else:
        analysis['duration_minutes'] = None

    # Identify issue number from branch name
    if 'issue-' in analysis['branch']:
        parts = analysis['branch'].split('-')
        for i, part in enumerate(parts):
            if part == 'issue' and i + 1 < len(parts):
                analysis['issue_number'] = parts[i + 1]
                break
        else:
            analysis['issue_number'] = 'unknown'
    else:
        analysis['issue_number'] = 'master' if analysis['branch'] == 'master' else 'unknown'

    return analysis


def analyze_branch(pipelines: List[Dict], project_key: str, branch_name: str) -> Dict:
    """
    Analyze all pipelines for a specific branch.
    """
    branch_pipelines = [p for p in pipelines if p['ref'] == branch_name]

    if not branch_pipelines:
        return None

    # Sort by creation time
    branch_pipelines.sort(key=lambda x: x['created_at'])

    # Count by status
    status_counts = {
        'success': sum(1 for p in branch_pipelines if p['status'] == 'success'),
        'failed': sum(1 for p in branch_pipelines if p['status'] == 'failed'),
        'canceled': sum(1 for p in branch_pipelines if p['status'] == 'canceled'),
    }

    total = len(branch_pipelines)

    # Analyze each pipeline
    detailed_pipelines = [analyze_single_pipeline(p, project_key) for p in branch_pipelines]

    # Count failure/cancellation reasons
    compile_failures = sum(1 for p in detailed_pipelines if p['compile_failed'])
    test_failures = sum(1 for p in detailed_pipelines if p['test_failed'])
    compile_cancels = sum(1 for p in detailed_pipelines if p['compile_cancelled'])
    test_cancels = sum(1 for p in detailed_pipelines if p['test_cancelled'])
    stuck_count = sum(1 for p in detailed_pipelines if p['stuck'])

    # Calculate statistics
    durations = [p['duration_seconds'] for p in detailed_pipelines if p['duration_seconds'] is not None]

    summary = {
        'project': project_key,
        'branch': branch_name,
        'total_pipelines': total,

        # Status distribution
        'success_count': status_counts['success'],
        'failed_count': status_counts['failed'],
        'canceled_count': status_counts['canceled'],

        'success_rate': round(status_counts['success'] / total * 100, 2) if total > 0 else 0,
        'failure_rate': round(status_counts['failed'] / total * 100, 2) if total > 0 else 0,
        'cancellation_rate': round(status_counts['canceled'] / total * 100, 2) if total > 0 else 0,

        # Failure/cancellation breakdown
        'compile_failures': compile_failures,
        'test_failures': test_failures,
        'compile_cancellations': compile_cancels,
        'test_cancellations': test_cancels,
        'stuck_pipelines': stuck_count,

        # Duration statistics
        'median_duration_sec': round(np.median(durations), 2) if durations else None,
        'mean_duration_sec': round(np.mean(durations), 2) if durations else None,
        'max_duration_sec': round(max(durations), 2) if durations else None,
        'min_duration_sec': round(min(durations), 2) if durations else None,

        # Timeline info
        'first_pipeline_date': branch_pipelines[0]['created_at'],
        'last_pipeline_date': branch_pipelines[-1]['created_at'],

        # Extract issue number
        'issue_number': detailed_pipelines[0]['issue_number'] if detailed_pipelines else 'unknown',
    }

    # Identify primary problem
    if stuck_count > total * 0.5:
        summary['primary_problem'] = 'STUCK_PIPELINES'
        summary['problem_severity'] = 'CRITICAL'
    elif test_cancels > total * 0.7:
        summary['primary_problem'] = 'TEST_TIMEOUTS'
        summary['problem_severity'] = 'HIGH'
    elif test_failures > total * 0.7:
        summary['primary_problem'] = 'TEST_FAILURES'
        summary['problem_severity'] = 'HIGH'
    elif compile_failures > total * 0.3:
        summary['primary_problem'] = 'COMPILE_FAILURES'
        summary['problem_severity'] = 'MEDIUM'
    elif status_counts['canceled'] > total * 0.3:
        summary['primary_problem'] = 'FREQUENT_CANCELLATIONS'
        summary['problem_severity'] = 'MEDIUM'
    elif status_counts['failed'] > total * 0.3:
        summary['primary_problem'] = 'FREQUENT_FAILURES'
        summary['problem_severity'] = 'MEDIUM'
    elif status_counts['success'] > total * 0.7:
        summary['primary_problem'] = 'NONE'
        summary['problem_severity'] = 'LOW'
    else:
        summary['primary_problem'] = 'MIXED_ISSUES'
        summary['problem_severity'] = 'MEDIUM'

    return summary, detailed_pipelines


def analyze_project(project_key: str, project_name: str, project_type: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze all branches in a project.

    Args:
        project_key: Project ID (e.g., 'A01')
        project_name: GitLab project name (e.g., 'ba_project_a01_battleship')
        project_type: Project type letter (e.g., 'a')
    """
    print(f"Analyzing {project_key}...")

    # Load pipeline data using new path helpers
    pipeline_file = get_data_file(project_name, 'pipelines.json', project_type)

    if not pipeline_file.exists():
        print(f"  [SKIP] Pipeline file not found: {pipeline_file}")
        return [], []

    with open(pipeline_file, 'r', encoding='utf-8') as f:
        pipelines = json.load(f)

    print(f"  Total pipelines: {len(pipelines)}")

    # Get unique branches
    branches = sorted(set(p['ref'] for p in pipelines))
    print(f"  Branches: {len(branches)}")

    # Analyze each branch
    branch_summaries = []
    all_detailed_pipelines = []

    for branch in branches:
        result = analyze_branch(pipelines, project_key, branch)
        if result:
            summary, detailed = result
            branch_summaries.append(summary)
            all_detailed_pipelines.extend(detailed)

            # Print branch summary
            issue = summary['issue_number']
            status_emoji = {
                'CRITICAL': '[!!!]',
                'HIGH': '[!!]',
                'MEDIUM': '[!]',
                'LOW': '[OK]'
            }[summary['problem_severity']]

            print(f"    {status_emoji} Issue #{issue}: {summary['total_pipelines']} pipelines, " +
                  f"{summary['success_rate']:.1f}% success, " +
                  f"Problem: {summary['primary_problem']}")

    return branch_summaries, all_detailed_pipelines


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze pipeline failures by project and branch')
    parser.add_argument(
        '--project-type',
        type=str,
        default='a',
        help='Project type letter (a, b, c, etc.). Default: a'
    )
    args = parser.parse_args()

    project_type = args.project_type

    print("=" * 80)
    print("DETAILED PIPELINE INVESTIGATION")
    print("Project-by-Project, Branch-by-Branch Analysis")
    print("=" * 80)
    print(f"Project Type: {project_type}")
    print()

    # Load projects for specified type
    projects_data = get_projects_by_letter(project_type)
    project_mapping = get_project_mapping(project_type)

    if not projects_data:
        print(f"[ERROR] No projects found for type '{project_type}'")
        return

    all_branch_summaries = []
    all_detailed_pipelines = []

    # Analyze each project
    for letter, number, gitlab_name in projects_data:
        project_key = f"{letter.upper()}{number}"
        branch_summaries, detailed_pipelines = analyze_project(project_key, gitlab_name, project_type)
        all_branch_summaries.extend(branch_summaries)
        all_detailed_pipelines.extend(detailed_pipelines)
        print()

    # Create output directory using path helpers
    output_dir = Path(get_analysis_file(project_type, 'pipeline_investigation', 'dummy')).parent
    ensure_dir(output_dir)

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Branch summaries
    branch_df = pd.DataFrame(all_branch_summaries)
    branch_output = get_analysis_file(project_type, 'pipeline_investigation', 'branch_summaries.csv')
    branch_df.to_csv(branch_output, index=False, encoding='utf-8')
    print(f"[OK] Branch summaries: {branch_output}")
    print(f"     Total branches analyzed: {len(branch_df)}")

    # Detailed pipeline analysis
    pipeline_df = pd.DataFrame(all_detailed_pipelines)
    pipeline_output = get_analysis_file(project_type, 'pipeline_investigation', 'detailed_pipeline_analysis.csv')
    pipeline_df.to_csv(pipeline_output, index=False, encoding='utf-8')
    print(f"[OK] Detailed pipelines: {pipeline_output}")
    print(f"     Total pipelines analyzed: {len(pipeline_df)}")

    # Create problem-specific reports
    print()
    print("Creating problem-specific reports...")

    # Critical branches (severity = CRITICAL or HIGH)
    critical = branch_df[branch_df['problem_severity'].isin(['CRITICAL', 'HIGH'])].copy()
    critical = critical.sort_values(['problem_severity', 'cancellation_rate'], ascending=[True, False])
    critical_output = get_analysis_file(project_type, 'pipeline_investigation', 'critical_branches.csv')
    critical.to_csv(critical_output, index=False, encoding='utf-8')
    print(f"[OK] Critical branches: {critical_output}")
    print(f"     Found {len(critical)} critical/high-severity branches")

    # Stuck pipelines
    stuck = pipeline_df[pipeline_df['stuck'] == True].copy()
    stuck = stuck.sort_values('duration_seconds', ascending=False)
    stuck_output = get_analysis_file(project_type, 'pipeline_investigation', 'stuck_pipelines_detailed.csv')
    stuck.to_csv(stuck_output, index=False, encoding='utf-8')
    print(f"[OK] Stuck pipelines: {stuck_output}")
    print(f"     Found {len(stuck)} stuck pipelines")

    # Test timeouts (test_cancelled = True)
    test_timeouts = pipeline_df[pipeline_df['test_cancelled'] == True].copy()
    test_timeout_output = get_analysis_file(project_type, 'pipeline_investigation', 'test_timeout_pipelines.csv')
    test_timeouts.to_csv(test_timeout_output, index=False, encoding='utf-8')
    print(f"[OK] Test timeouts: {test_timeout_output}")
    print(f"     Found {len(test_timeouts)} test timeout pipelines")

    # Test failures
    test_failures = pipeline_df[pipeline_df['test_failed'] == True].copy()
    test_fail_output = get_analysis_file(project_type, 'pipeline_investigation', 'test_failure_pipelines.csv')
    test_failures.to_csv(test_fail_output, index=False, encoding='utf-8')
    print(f"[OK] Test failures: {test_fail_output}")
    print(f"     Found {len(test_failures)} test failure pipelines")

    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall statistics
    total_pipelines = len(pipeline_df)
    total_branches = len(branch_df)

    print(f"Total Projects: {len(projects_data)}")
    print(f"Total Branches: {total_branches}")
    print(f"Total Pipelines: {total_pipelines}")
    print()

    # Problem distribution
    print("BRANCH PROBLEM DISTRIBUTION:")
    problem_dist = branch_df['primary_problem'].value_counts()
    for problem, count in problem_dist.items():
        pct = count / total_branches * 100
        print(f"  {problem}: {count} branches ({pct:.1f}%)")
    print()

    # Severity distribution
    print("BRANCH SEVERITY DISTRIBUTION:")
    severity_dist = branch_df['problem_severity'].value_counts()
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_dist.get(severity, 0)
        pct = count / total_branches * 100
        print(f"  {severity}: {count} branches ({pct:.1f}%)")
    print()

    # Top problematic branches
    print("TOP 10 MOST PROBLEMATIC BRANCHES:")
    top_problematic = branch_df.nlargest(10, 'cancellation_rate')
    for idx, row in top_problematic.iterrows():
        print(f"  {row['project']}-Issue#{row['issue_number']}: " +
              f"{row['cancellation_rate']:.1f}% cancelled, " +
              f"{row['stuck_pipelines']} stuck, " +
              f"Problem: {row['primary_problem']}")
    print()

    # Projects ranked by success rate
    print("PROJECTS RANKED BY AVERAGE SUCCESS RATE:")
    project_stats = branch_df.groupby('project').agg({
        'success_rate': 'mean',
        'total_pipelines': 'sum'
    }).sort_values('success_rate', ascending=False)

    for project, row in project_stats.iterrows():
        print(f"  {project}: {row['success_rate']:.1f}% avg success (n={row['total_pipelines']} total)")

    print()
    print("=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
