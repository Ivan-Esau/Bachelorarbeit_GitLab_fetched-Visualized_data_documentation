"""
Analyze Pass Rates - Calculate pipeline and job success rates

Extracts pass rate metrics from existing GitLab data:
- Pipeline pass rate
- Job pass rate
- Pass rate by stage
- Pass rate by branch
- Pass rate over time
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any


def load_project_data(project_name: str, data_dir: str = '../data_raw') -> Dict:
    """Load all data for a project"""
    project_dir = os.path.join(data_dir, project_name)

    data = {}
    files = ['pipelines.json', 'branches.json', 'merge_requests.json', 'issues.json']

    for filename in files:
        filepath = os.path.join(project_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                key = filename.replace('.json', '')
                data[key] = json.load(f)

    return data


def calculate_pipeline_pass_rates(pipelines: List[Dict]) -> Dict[str, Any]:
    """
    Calculate pipeline pass rate metrics

    Returns:
        Dictionary with pass rate statistics
    """
    if not pipelines:
        return {'total': 0, 'success': 0, 'failed': 0, 'pass_rate': 0.0}

    status_counts = defaultdict(int)
    for pipeline in pipelines:
        status = pipeline.get('status', 'unknown')
        status_counts[status] += 1

    total = len(pipelines)
    success = status_counts['success']
    failed = status_counts['failed']
    canceled = status_counts['canceled']
    skipped = status_counts['skipped']

    # Pass rate = successful / (successful + failed)
    # Exclude canceled and skipped from denominator
    denominator = success + failed
    pass_rate = (success / denominator * 100) if denominator > 0 else 0.0

    return {
        'total_pipelines': total,
        'success': success,
        'failed': failed,
        'canceled': canceled,
        'skipped': skipped,
        'other': total - success - failed - canceled - skipped,
        'pass_rate': round(pass_rate, 2),
        'all_statuses': dict(status_counts)
    }


def calculate_job_pass_rates(pipelines: List[Dict]) -> Dict[str, Any]:
    """
    Calculate job pass rate metrics

    Returns:
        Dictionary with job pass rate statistics
    """
    all_jobs = []
    for pipeline in pipelines:
        jobs = pipeline.get('jobs', [])
        all_jobs.extend(jobs)

    if not all_jobs:
        return {'total': 0, 'success': 0, 'failed': 0, 'pass_rate': 0.0}

    status_counts = defaultdict(int)
    for job in all_jobs:
        status = job.get('status', 'unknown')
        status_counts[status] += 1

    total = len(all_jobs)
    success = status_counts['success']
    failed = status_counts['failed']
    canceled = status_counts['canceled']
    skipped = status_counts['skipped']

    denominator = success + failed
    pass_rate = (success / denominator * 100) if denominator > 0 else 0.0

    return {
        'total_jobs': total,
        'success': success,
        'failed': failed,
        'canceled': canceled,
        'skipped': skipped,
        'other': total - success - failed - canceled - skipped,
        'pass_rate': round(pass_rate, 2),
        'all_statuses': dict(status_counts)
    }


def calculate_stage_pass_rates(pipelines: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate pass rate per pipeline stage (build, test, deploy, etc.)

    Returns:
        Dictionary mapping stage name to pass rate metrics
    """
    stage_jobs = defaultdict(list)

    for pipeline in pipelines:
        jobs = pipeline.get('jobs', [])
        for job in jobs:
            stage = job.get('stage', 'unknown')
            stage_jobs[stage].append(job)

    stage_stats = {}
    for stage, jobs in stage_jobs.items():
        status_counts = defaultdict(int)
        for job in jobs:
            status = job.get('status', 'unknown')
            status_counts[status] += 1

        total = len(jobs)
        success = status_counts['success']
        failed = status_counts['failed']

        denominator = success + failed
        pass_rate = (success / denominator * 100) if denominator > 0 else 0.0

        stage_stats[stage] = {
            'total_jobs': total,
            'success': success,
            'failed': failed,
            'pass_rate': round(pass_rate, 2)
        }

    return stage_stats


def calculate_branch_pass_rates(pipelines: List[Dict], branches: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate pass rate per branch

    Returns:
        Dictionary mapping branch name to pass rate metrics
    """
    branch_pipelines = defaultdict(list)

    for pipeline in pipelines:
        ref = pipeline.get('ref', 'unknown')
        branch_pipelines[ref].append(pipeline)

    branch_stats = {}
    for branch_name, branch_pipes in branch_pipelines.items():
        status_counts = defaultdict(int)
        for pipeline in branch_pipes:
            status = pipeline.get('status', 'unknown')
            status_counts[status] += 1

        total = len(branch_pipes)
        success = status_counts['success']
        failed = status_counts['failed']

        denominator = success + failed
        pass_rate = (success / denominator * 100) if denominator > 0 else 0.0

        branch_stats[branch_name] = {
            'total_pipelines': total,
            'success': success,
            'failed': failed,
            'pass_rate': round(pass_rate, 2)
        }

    return branch_stats


def analyze_pass_rate_trends(pipelines: List[Dict]) -> Dict[str, Any]:
    """
    Analyze pass rate trends over time

    Returns:
        Dictionary with temporal trends
    """
    # Sort pipelines by created date
    sorted_pipes = sorted(pipelines, key=lambda p: p.get('created_at', ''))

    if len(sorted_pipes) < 10:
        return {'note': 'Not enough pipelines for trend analysis'}

    # Calculate rolling pass rate (last N pipelines)
    window_size = 10
    rolling_pass_rates = []

    for i in range(len(sorted_pipes) - window_size + 1):
        window = sorted_pipes[i:i + window_size]
        success = sum(1 for p in window if p.get('status') == 'success')
        failed = sum(1 for p in window if p.get('status') == 'failed')
        denominator = success + failed
        pass_rate = (success / denominator * 100) if denominator > 0 else 0.0
        rolling_pass_rates.append(round(pass_rate, 2))

    # First 10 vs last 10
    first_10 = sorted_pipes[:10]
    last_10 = sorted_pipes[-10:]

    def calc_rate(pipes):
        success = sum(1 for p in pipes if p.get('status') == 'success')
        failed = sum(1 for p in pipes if p.get('status') == 'failed')
        denom = success + failed
        return round((success / denom * 100) if denom > 0 else 0.0, 2)

    first_rate = calc_rate(first_10)
    last_rate = calc_rate(last_10)

    return {
        'first_10_pipelines_pass_rate': first_rate,
        'last_10_pipelines_pass_rate': last_rate,
        'improvement': round(last_rate - first_rate, 2),
        'rolling_average_pass_rate': round(sum(rolling_pass_rates) / len(rolling_pass_rates), 2) if rolling_pass_rates else 0.0
    }


def analyze_project(project_name: str, data_dir: str = '../data_raw') -> Dict[str, Any]:
    """Analyze pass rates for a single project"""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {project_name}")
    print(f"{'='*80}")

    data = load_project_data(project_name, data_dir)

    pipelines = data.get('pipelines', [])
    branches = data.get('branches', [])

    if not pipelines:
        print("  No pipeline data found")
        return {}

    # Calculate all metrics
    pipeline_stats = calculate_pipeline_pass_rates(pipelines)
    job_stats = calculate_job_pass_rates(pipelines)
    stage_stats = calculate_stage_pass_rates(pipelines)
    branch_stats = calculate_branch_pass_rates(pipelines, branches)
    trend_stats = analyze_pass_rate_trends(pipelines)

    # Print summary
    print(f"\nPipeline Pass Rate: {pipeline_stats['pass_rate']}%")
    print(f"   Total Pipelines: {pipeline_stats['total_pipelines']}")
    print(f"   Success: {pipeline_stats['success']}")
    print(f"   Failed: {pipeline_stats['failed']}")
    print(f"   Canceled: {pipeline_stats['canceled']}")

    print(f"\nJob Pass Rate: {job_stats['pass_rate']}%")
    print(f"   Total Jobs: {job_stats['total_jobs']}")
    print(f"   Success: {job_stats['success']}")
    print(f"   Failed: {job_stats['failed']}")

    print(f"\nPass Rate by Stage:")
    for stage, stats in sorted(stage_stats.items(), key=lambda x: x[1]['pass_rate'], reverse=True):
        print(f"   {stage:<15}: {stats['pass_rate']:>6}% ({stats['success']}/{stats['total_jobs']} jobs)")

    print(f"\nTop 5 Branches by Pass Rate:")
    sorted_branches = sorted(branch_stats.items(), key=lambda x: (x[1]['pass_rate'], x[1]['total_pipelines']), reverse=True)
    for branch, stats in sorted_branches[:5]:
        print(f"   {branch:<30}: {stats['pass_rate']:>6}% ({stats['success']}/{stats['total_pipelines']} pipelines)")

    if 'improvement' in trend_stats:
        print(f"\nTrends:")
        print(f"   First 10 pipelines: {trend_stats['first_10_pipelines_pass_rate']}%")
        print(f"   Last 10 pipelines: {trend_stats['last_10_pipelines_pass_rate']}%")
        print(f"   Improvement: {trend_stats['improvement']:+.2f}%")

    return {
        'project_name': project_name,
        'pipeline_pass_rate': pipeline_stats,
        'job_pass_rate': job_stats,
        'stage_pass_rates': stage_stats,
        'branch_pass_rates': branch_stats,
        'trends': trend_stats
    }


def main():
    """Analyze all projects"""

    print("="*80)
    print("PASS RATE ANALYSIS")
    print("="*80)

    data_dir = '../data_raw'

    # Get all project directories
    projects = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('ba_project_')]

    projects.sort()

    all_results = []

    for project in projects:
        result = analyze_project(project, data_dir)
        if result:
            all_results.append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY - ALL PROJECTS")
    print(f"{'='*80}\n")

    print(f"{'Project':<35} {'Pipeline Pass':>15} {'Job Pass':>15}")
    print("-"*80)

    for result in all_results:
        project = result['project_name']
        pipeline_rate = result['pipeline_pass_rate']['pass_rate']
        job_rate = result['job_pass_rate']['pass_rate']
        print(f"{project:<35} {pipeline_rate:>14.2f}% {job_rate:>14.2f}%")

    # Overall average
    if all_results:
        avg_pipeline_rate = sum(r['pipeline_pass_rate']['pass_rate'] for r in all_results) / len(all_results)
        avg_job_rate = sum(r['job_pass_rate']['pass_rate'] for r in all_results) / len(all_results)

        print("-"*80)
        print(f"{'AVERAGE':<35} {avg_pipeline_rate:>14.2f}% {avg_job_rate:>14.2f}%")

    # Save results
    output_file = os.path.join(data_dir, 'pass_rate_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == '__main__':
    main()
