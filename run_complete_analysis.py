"""
Master Script - Complete Analysis Pipeline
Runs: Fetch → Analyze → Visualize

Run this script to perform complete data collection and analysis
"""

import subprocess
import sys
import os
from pathlib import Path


def run_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and report success/failure

    Args:
        script_path: Path to script
        description: What the script does

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)

        print(f"\n✓ SUCCESS: {description}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {description}")
        print("\nError output:")
        print(e.stderr)
        print("\nStdout:")
        print(e.stdout)
        return False


def main():
    """Run complete analysis pipeline"""

    print("="*80)
    print("GITLAB DATA ANALYSIS - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("  1. Fetch raw data from GitLab (all projects)")
    print("  2. Analyze and link the data (issues → MRs → pipelines)")
    print("  3. Create visualizations (charts and tables)")
    print()

    # Check if .env exists
    env_file = Path('.env')
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please create a .env file with:")
        print("  GITLAB_URL=https://your-gitlab-url")
        print("  GITLAB_TOKEN=your-token")
        return

    response = input("Continue? This may take 10-30 minutes depending on data size. [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print("\nStarting pipeline...\n")

    # Step 1: Fetch raw data
    success1 = run_script(
        'scripts/fetch_raw_data.py',
        'Fetching raw data from GitLab'
    )

    if not success1:
        print("\n⚠ Data fetch failed. Cannot continue.")
        return

    # Step 2: Analyze data
    success2 = run_script(
        'scripts/analyze_raw_data.py',
        'Analyzing and linking data'
    )

    if not success2:
        print("\n⚠ Analysis failed. Cannot continue.")
        return

    # Step 3: Create visualizations
    success3 = run_script(
        'scripts/visualize_raw_data.py',
        'Creating visualizations'
    )

    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    if success1 and success2 and success3:
        print("\n✓ All steps completed successfully!")
        print("\nOutput locations:")
        print("  - Raw data: data_raw/")
        print("  - Analysis: data_raw/_analysis.json")
        print("  - Charts: output_raw/")
        print("\nGenerated charts:")
        print("  - 00_summary_table.png")
        print("  - 01_issue_counts.png")
        print("  - 02_issue_completion.png")
        print("  - 03_pipeline_statuses.png")
        print("  - 04_mr_status_distribution.png")
        print("  - 05_success_rate.png")
        print("  - 06_pipeline_counts.png")
    else:
        print("\n⚠ Some steps failed. Check output above.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
