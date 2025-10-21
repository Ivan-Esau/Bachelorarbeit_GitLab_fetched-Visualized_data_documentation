"""
Master Branch Metrics Visualization Script

Runs all branch-level visualizations:
1. Heat map table (all projects)
2. Per-project 4-panel charts (10 charts)
3. Coverage timeline charts (detailed)

This is the main entry point for generating all branch-level visualizations.

Usage:
    python visualize_all_branch_metrics.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def run_visualization(script_name, description):
    """
    Run a visualization script and report results

    Args:
        script_name: Name of the script file
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 100)
    print(f"RUNNING: {description}")
    print("=" * 100)

    script_path = Path('visualizers') / script_name

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd='visualizers',
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("Warnings/Info:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {description}")
            return True
        else:
            print(f"\n✗ FAILED: {description}")
            print(f"Return code: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT: {description} (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {description}")
        print(f"Exception: {e}")
        return False


def main():
    """Run all branch-level visualizations"""

    start_time = datetime.now()

    print("=" * 100)
    print("BRANCH METRICS VISUALIZATION SUITE")
    print("=" * 100)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This will create:")
    print("  1. Heat map table (all 60 branches)")
    print("  2. Per-project 4-panel charts (10 projects)")
    print()

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return 0

    # Track results
    results = {}

    # Step 1: Heat map table
    results['heatmap'] = run_visualization(
        'visualize_branch_heatmap.py',
        'Heat Map Table (all branches)'
    )

    # Step 2: Per-project charts
    results['per_project'] = run_visualization(
        'visualize_branch_metrics_per_project.py',
        'Per-Project 4-Panel Charts'
    )

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 100)
    print("VISUALIZATION SUITE COMPLETE")
    print("=" * 100)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print()

    # Results summary
    print("Results:")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    print()
    print(f"Overall: {success_count}/{total_count} visualization sets completed successfully")
    print()

    if success_count == total_count:
        print("✓ All visualizations completed successfully!")
        print()
        print("Output locations:")
        print("  - visualizations/branch_metrics/")
        print("    - branch_metrics_summary.png")
        print("    - branch_metrics_all.csv")
        print("    - a01_branch_heatmap.png ... a10_branch_heatmap.png")
        print("    - a01_branch_metrics.png ... a10_branch_metrics.png")
    else:
        print("⚠ Some visualizations failed. Check output above for details.")

    print()
    print("=" * 100)

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
