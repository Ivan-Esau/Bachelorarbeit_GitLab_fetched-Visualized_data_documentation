"""
Modular GitLab Data Fetcher - Clean Architecture Version

Uses modular components for better maintainability:
- Core: Client, config loader, file manager
- Fetchers: Individual fetchers for each data type
- Orchestrator: Project fetcher to coordinate everything

Supports letter-based project structure (a, b, c, etc.)

This is the NEW modular version. The old fetch_raw_data.py is kept for reference.
"""

import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Import core components
from core import GitLabClient, FileManager
from core.config_loader import load_project_types, get_projects_by_letter

# Import project fetcher
from fetchers import ProjectFetcher


def main():
    """Main execution - fetch all project data using modular system"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch GitLab data for projects')
    parser.add_argument(
        '--project-type',
        type=str,
        default='a',
        help='Project type letter (a, b, c, etc.). Default: a'
    )
    args = parser.parse_args()

    project_type = args.project_type

    # Load environment variables
    load_dotenv('../.env')

    GITLAB_URL = os.getenv('GITLAB_URL', 'https://gitlab.com')
    GITLAB_TOKEN = os.getenv('GITLAB_TOKEN', '')

    if not GITLAB_TOKEN:
        print("ERROR: GITLAB_TOKEN not found in .env file")
        return

    # Load projects from config for specified type
    try:
        projects_data = get_projects_by_letter(project_type)
        # Convert to (project_id, project_name) tuples
        # project_id is the full GitLab path for API calls
        # project_name is also the full path (used for file organization and metadata)
        projects = [(gitlab_full_path, gitlab_full_path) for letter, num, gitlab_full_path in projects_data]
    except Exception as e:
        print(f"ERROR: {e}")
        return

    if not projects:
        print(f"ERROR: No projects found for type '{project_type}'")
        print(f"Check config/project_types.json to enable this project type")
        return

    print("="*80)
    print("MODULAR GITLAB DATA FETCHER")
    print("="*80)
    print(f"GitLab URL: {GITLAB_URL}")
    print(f"Project Type: {project_type}")
    print(f"Projects: {len(projects)}")
    print()

    # Create output directory
    output_dir = '../data_raw'
    FileManager.ensure_directory(output_dir)

    # Initialize GitLab client
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN)

    # Initialize project fetcher
    project_fetcher = ProjectFetcher(client)

    # Fetch each project
    all_metadata = []
    successful_projects = 0

    for project_id, project_name in projects:
        try:
            # Fetch complete data
            project_data = project_fetcher.fetch_complete_project(project_id, project_name)

            # Save data with project type
            project_fetcher.save(project_data, output_dir, project_type)

            # Track metadata
            all_metadata.append({
                'project_name': project_name,
                'stats': project_data['stats']
            })

            successful_projects += 1

        except Exception as e:
            print(f"\nERROR processing {project_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save overall summary in type-specific location
    summary = {
        'fetch_date': datetime.now().isoformat(),
        'project_type': project_type,
        'projects_fetched': successful_projects,
        'projects_total': len(projects),
        'projects': all_metadata
    }

    # Save summary in metadata folder
    metadata_dir = os.path.join(output_dir, '_metadata')
    FileManager.ensure_directory(metadata_dir)
    summary_file = os.path.join(metadata_dir, f'fetch_summary_type_{project_type}.json')
    FileManager.save_json(summary, summary_file)

    # Print final summary
    print("\n" + "="*80)
    print("DATA FETCH COMPLETE")
    print("="*80)
    print(f"Projects fetched: {successful_projects}/{len(projects)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")

    # Print statistics table
    if all_metadata:
        print("\nSummary:")
        print(f"{'Project':<30} {'Issues':>8} {'Branches':>10} {'MRs':>8} {'Commits':>10} {'Pipelines':>10} {'Artifacts':>10}")
        print("-" * 90)
        for meta in all_metadata:
            stats = meta['stats']
            print(f"{meta['project_name']:<30} {stats['total_issues']:>8} {stats['total_branches']:>10} "
                  f"{stats['total_merge_requests']:>8} {stats['total_commits']:>10} {stats['total_pipelines']:>10} "
                  f"{stats['total_artifacts']:>10}")


if __name__ == '__main__':
    main()
