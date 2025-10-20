"""
Test Modular System - Verify all modules load correctly

Quick verification that the modular system is working
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_core_imports():
    """Test that core modules import correctly"""
    print("Testing core imports...")
    try:
        from core import GitLabClient, load_project_config, FileManager
        print("  [OK] core.GitLabClient")
        print("  [OK] core.load_project_config")
        print("  [OK] core.FileManager")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def test_fetchers_imports():
    """Test that fetcher modules import correctly"""
    print("\nTesting fetcher imports...")
    try:
        from fetchers import (
            BaseFetcher,
            IssuesFetcher,
            BranchesFetcher,
            MergeRequestsFetcher,
            CommitsFetcher,
            PipelinesFetcher,
            ArtifactsFetcher,
            CoverageExtractor,
            ProjectFetcher
        )
        print("  [OK] fetchers.BaseFetcher")
        print("  [OK] fetchers.IssuesFetcher")
        print("  [OK] fetchers.BranchesFetcher")
        print("  [OK] fetchers.MergeRequestsFetcher")
        print("  [OK] fetchers.CommitsFetcher")
        print("  [OK] fetchers.PipelinesFetcher")
        print("  [OK] fetchers.ArtifactsFetcher")
        print("  [OK] fetchers.CoverageExtractor")
        print("  [OK] fetchers.ProjectFetcher")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def test_instantiation():
    """Test that classes can be instantiated"""
    print("\nTesting class instantiation...")
    try:
        from core import GitLabClient
        from fetchers import ProjectFetcher

        # Create client (with dummy values)
        client = GitLabClient('https://gitlab.com', 'dummy-token')
        print("  [OK] GitLabClient instantiated")

        # Create project fetcher
        fetcher = ProjectFetcher(client)
        print("  [OK] ProjectFetcher instantiated")

        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def test_config_loading():
    """Test config loading (if config file exists)"""
    print("\nTesting config loading...")
    try:
        from core import load_project_config

        projects = load_project_config('../config_projects.json')
        print(f"  [OK] Loaded {len(projects)} projects from config")

        for project_id, project_name in projects[:3]:  # Show first 3
            print(f"     - {project_name}")

        if len(projects) > 3:
            print(f"     ... and {len(projects) - 3} more")

        return True
    except FileNotFoundError:
        print("  [WARN] Config file not found (this is OK for testing)")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("MODULAR SYSTEM VERIFICATION")
    print("="*80)

    tests = [
        test_core_imports,
        test_fetchers_imports,
        test_instantiation,
        test_config_loading
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[SUCCESS] All tests passed! Modular system is working correctly.")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test(s) failed. Please check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
