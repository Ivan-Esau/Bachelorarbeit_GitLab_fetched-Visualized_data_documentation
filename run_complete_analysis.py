"""
Master Script - Complete Analysis Pipeline
Runs: Fetch → Extract Coverage → Analyze → Compare

Run this script to perform complete data collection and analysis
"""

import subprocess
import sys
import os
from pathlib import Path


def run_script(script_path: str, args: list, description: str) -> bool:
    """
    Run a Python script with arguments and report success/failure

    Args:
        script_path: Path to script
        args: List of command-line arguments
        description: What the script does

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*100)
    print(f"STEP: {description}")
    print("="*100)

    try:
        cmd = [sys.executable, script_path] + args
        result = subprocess.run(
            cmd,
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

        print(f"\n[OK] SUCCESS: {description}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] FAILED: {description}")
        print("\nError output:")
        print(e.stderr)
        print("\nStdout:")
        print(e.stdout)
        return False


def main():
    """Run complete analysis pipeline"""

    print("="*100)
    print("GITLAB DATA ANALYSIS - COMPLETE PIPELINE")
    print("="*100)
    print("\nThis pipeline will execute:")
    print("  PHASE 1: Data Fetching")
    print("    - Fetch all raw data from GitLab for Type A projects (10 projects)")
    print("    - Fetch all raw data from GitLab for Type B projects (10 projects)")
    print()
    print("  PHASE 2: Coverage Extraction")
    print("    - Extract JaCoCo coverage data from artifacts (Type A)")
    print("    - Extract JaCoCo coverage data from artifacts (Type B)")
    print()
    print("  PHASE 3: Per-Project Analysis")
    print("    - Branch lifecycle analysis (both types)")
    print("    - Branch metrics heatmaps (both types)")
    print("    - Pipeline duration analysis (both types)")
    print("    - Pipeline job success analysis (both types)")
    print("    - Pipeline success summary (both types)")
    print("    - Pipeline investigation - failure analysis (both types)")
    print("    - Coverage per branch visualization (both types)")
    print()
    print("  PHASE 4: Quality Analysis")
    print("    - Merge quality analysis")
    print("    - Runner correlation analysis")
    print()
    print("  PHASE 5: Type A vs Type B Comparisons")
    print("    - Pipeline success comparison")
    print("    - Merge success comparison")
    print("    - Coverage per branch comparison")
    print("    - Pipeline investigation comparison (problems & cancellations)")
    print()

    # Check if .env exists
    env_file = Path('.env')
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please create a .env file with:")
        print("  GITLAB_URL=https://your-gitlab-url")
        print("  GITLAB_TOKEN=your-token")
        return

    response = input("Continue? This may take 30-60 minutes depending on data size. [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print("\nStarting pipeline...\n")

    results = []

    # ========================================================================
    # PHASE 1: DATA FETCHING
    # ========================================================================
    print("\n" + "#"*100)
    print("# PHASE 1: DATA FETCHING")
    print("#"*100)

    results.append(run_script(
        'scripts/fetch_modular.py',
        ['--project-type', 'a'],
        'Fetching Type A projects data from GitLab'
    ))

    results.append(run_script(
        'scripts/fetch_modular.py',
        ['--project-type', 'b'],
        'Fetching Type B projects data from GitLab'
    ))

    # ========================================================================
    # PHASE 2: COVERAGE EXTRACTION
    # ========================================================================
    print("\n" + "#"*100)
    print("# PHASE 2: COVERAGE EXTRACTION")
    print("#"*100)

    results.append(run_script(
        'scripts/extract_all_coverage.py',
        ['--project-type', 'a'],
        'Extracting coverage data from Type A artifacts'
    ))

    results.append(run_script(
        'scripts/extract_all_coverage.py',
        ['--project-type', 'b'],
        'Extracting coverage data from Type B artifacts'
    ))

    # ========================================================================
    # PHASE 3: PER-PROJECT ANALYSIS
    # ========================================================================
    print("\n" + "#"*100)
    print("# PHASE 3: PER-PROJECT ANALYSIS")
    print("#"*100)

    # Branch Lifecycle
    results.append(run_script(
        'scripts/analyzers/branch_lifecycle/analyze_branch_lifecycle.py',
        ['--project-type', 'a'],
        'Analyzing branch lifecycle (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/branch_lifecycle/analyze_branch_lifecycle.py',
        ['--project-type', 'b'],
        'Analyzing branch lifecycle (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/branch_lifecycle/visualize_branch_duration_boxplot.py',
        [],
        'Creating branch duration boxplot (summary)'
    ))

    # Branch Metrics
    results.append(run_script(
        'scripts/analyzers/branch_metrics/visualize_branch_heatmap.py',
        ['--project-type', 'a'],
        'Creating branch metrics heatmaps (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/branch_metrics/visualize_branch_heatmap.py',
        ['--project-type', 'b'],
        'Creating branch metrics heatmaps (Type B)'
    ))

    # Pipeline Analysis
    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_durations.py',
        ['--project-type', 'a'],
        'Analyzing pipeline durations (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_durations.py',
        ['--project-type', 'b'],
        'Analyzing pipeline durations (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_job_success.py',
        ['--project-type', 'a'],
        'Analyzing pipeline job success - failed only (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_job_success.py',
        ['--project-type', 'b'],
        'Analyzing pipeline job success - failed only (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_job_success_all.py',
        ['--project-type', 'a'],
        'Analyzing pipeline job success - all statuses (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_job_success_all.py',
        ['--project-type', 'b'],
        'Analyzing pipeline job success - all statuses (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_success_summary.py',
        ['--project-type', 'a'],
        'Creating pipeline success summary (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_success_summary.py',
        ['--project-type', 'b'],
        'Creating pipeline success summary (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/detailed_pipeline_investigation.py',
        ['--project-type', 'a'],
        'Detailed pipeline investigation (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/detailed_pipeline_investigation.py',
        ['--project-type', 'b'],
        'Detailed pipeline investigation (Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_investigation.py',
        ['--project-type', 'a'],
        'Visualizing pipeline investigation - failure types (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/pipelines/visualize_pipeline_investigation.py',
        ['--project-type', 'b'],
        'Visualizing pipeline investigation - failure types (Type B)'
    ))

    # Coverage Analysis
    results.append(run_script(
        'scripts/analyzers/coverage/visualize_final_coverage_per_branch.py',
        ['--project-type', 'a'],
        'Visualizing coverage per branch (Type A)'
    ))

    results.append(run_script(
        'scripts/analyzers/coverage/visualize_final_coverage_per_branch.py',
        ['--project-type', 'b'],
        'Visualizing coverage per branch (Type B)'
    ))

    # ========================================================================
    # PHASE 4: QUALITY ANALYSIS
    # ========================================================================
    print("\n" + "#"*100)
    print("# PHASE 4: QUALITY ANALYSIS")
    print("#"*100)

    results.append(run_script(
        'scripts/analyzers/quality/analyze_merge_quality.py',
        [],
        'Analyzing merge quality (all projects)'
    ))

    results.append(run_script(
        'scripts/analyzers/quality/analyze_runner_correlation.py',
        [],
        'Analyzing runner correlation (all projects)'
    ))

    # ========================================================================
    # PHASE 5: TYPE A vs TYPE B COMPARISONS
    # ========================================================================
    print("\n" + "#"*100)
    print("# PHASE 5: TYPE A vs TYPE B COMPARISONS")
    print("#"*100)

    results.append(run_script(
        'scripts/analyzers/comparisons/compare_pipeline_success_a_vs_b.py',
        [],
        'Comparing pipeline success (Type A vs Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/comparisons/compare_merge_success_a_vs_b.py',
        [],
        'Comparing merge success (Type A vs Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/comparisons/compare_coverage_per_branch_a_vs_b.py',
        [],
        'Comparing coverage per branch (Type A vs Type B)'
    ))

    results.append(run_script(
        'scripts/analyzers/comparisons/compare_pipeline_investigation_a_vs_b.py',
        [],
        'Comparing pipeline investigation - problems & cancellations (Type A vs Type B)'
    ))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("PIPELINE COMPLETE")
    print("="*100)

    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"\n[OK] All {total_count} steps completed successfully!")
    else:
        print(f"\n[WARNING] {success_count}/{total_count} steps completed successfully.")
        print(f"          {total_count - success_count} steps failed. Check output above.")

    print("\nOutput locations:")
    print("  - Raw data: data_raw/a/ and data_raw/b/")
    print("  - Coverage data: data_raw/a/*/coverage.json and data_raw/b/*/coverage.json")
    print("  - Visualizations: visualizations/")
    print("    - visualizations/a/ (Type A per-project charts)")
    print("    - visualizations/b/ (Type B per-project charts)")
    print("    - visualizations/summary/ (Summary statistics)")
    print("    - visualizations/comparisons/ (Type A vs Type B comparisons)")
    print()
    print("Generated visualizations:")
    print("  Per-Project:")
    print("    - Branch lifecycle duration charts")
    print("    - Branch metrics heatmaps")
    print("    - Pipeline duration analysis")
    print("    - Pipeline job success charts")
    print("    - Pipeline investigation - failure type analysis")
    print("    - Coverage per branch charts")
    print()
    print("  Summary:")
    print("    - Branch duration boxplots")
    print("    - Pipeline success summaries")
    print("    - Quality analysis (merge quality, runner correlation)")
    print()
    print("  Comparisons:")
    print("    - Pipeline success: Type A vs Type B")
    print("    - Merge success: Type A vs Type B")
    print("    - Coverage per branch: Type A vs Type B")
    print("    - Pipeline investigation (problems & cancellations): Type A vs Type B")

    print("\n" + "="*100)


if __name__ == '__main__':
    main()
