# Artifact Data Analysis - Research and Investigation

**Date**: 2025-10-22
**Researcher**: Data Analysis System

---

## Available Data

### Data Source

**Location**: `data_raw/{type}/{project}/artifacts.json`

All 20 projects (A01-A10, B01-B10) have artifact data available.

### Data Structure

Each artifact entry contains:
```json
{
  "artifact_id": "14338_archive",
  "filename": "artifacts.zip",
  "size": 183418,                    // bytes
  "file_type": "archive",             // archive, junit, jacoco, metadata, trace
  "file_format": "zip",               // zip, gzip, null
  "created_at": "2025-10-12T02:20:55.109Z",
  "expires_at": "2025-11-11T02:21:15.228Z",
  "job_id": 14338,
  "job_name": "test_job",
  "job_stage": "test",                // test, compile
  "job_status": "success",
  "pipeline_id": 5865,
  "pipeline_status": "success"
}
```

---

## Example: Project A01

**Statistics:**
- Total artifacts: 179
- Unique pipelines: 35
- Unique jobs: 64
- Total size: 9.97 MB

**File Type Distribution:**
- `trace`: 64 artifacts (0.42 MB) - Job logs
- `archive`: 46 artifacts (8.73 MB) - Main artifact archives
- `metadata`: 23 artifacts (0.07 MB) - Pipeline metadata
- `junit`: 23 artifacts (0.67 MB) - Test results
- `jacoco`: 23 artifacts (0.08 MB) - Code coverage reports

**Job Stage Distribution:**
- `test` stage: 144 artifacts
- `compile` stage: 35 artifacts

**Size Distribution by Type:**
1. `archive`: 8.73 MB (87.6%)
2. `junit`: 0.67 MB (6.7%)
3. `trace`: 0.42 MB (4.2%)
4. `jacoco`: 0.08 MB (0.8%)
5. `metadata`: 0.07 MB (0.7%)

---

## Possible Metrics

### 1. Artifact Count Metrics
- Total artifacts per branch
- Total artifacts per project
- Artifacts per pipeline (average)
- Artifacts by file type
- Artifacts by job stage

### 2. Size Metrics
- Total artifact size per branch (MB)
- Total artifact size per project (MB)
- Average artifact size per pipeline
- Size distribution by file type
- Size distribution by job stage

### 3. Pipeline Coverage
- Percentage of pipelines with artifacts
- Percentage of successful pipelines with artifacts
- Artifact generation rate over time

### 4. Type Distribution
- Proportion of each file type (count)
- Proportion of each file type (size)
- Archive vs Test artifacts ratio

---

## Proposed Visualizations

### Option 1: Artifact Count by Type (Per Project)

**Visualization**: Stacked bar chart
- X-axis: Branches (e.g., Issue #1, Issue #2, ...)
- Y-axis: Number of artifacts
- Stacks: File types (archive, junit, jacoco, metadata, trace)
- One chart per project (A01-A10, B01-B10)

**What it shows**: How many artifacts of each type are generated per branch.

### Option 2: Artifact Size by Type (Per Project)

**Visualization**: Stacked bar chart
- X-axis: Branches
- Y-axis: Total size (MB)
- Stacks: File types
- One chart per project

**What it shows**: How much storage each branch consumes, broken down by artifact type.

### Option 3: Total Artifact Size per Branch (Per Project)

**Visualization**: Simple bar chart
- X-axis: Branches
- Y-axis: Total size (MB)
- One bar per branch
- One chart per project

**What it shows**: Overall artifact storage footprint per branch.

### Option 4: Artifact Generation Over Time (Per Project)

**Visualization**: Line chart
- X-axis: Pipeline number (chronological)
- Y-axis: Cumulative artifact size (MB)
- One line per branch
- One chart per project

**What it shows**: How artifact storage grows over the project lifecycle.

### Option 5: Artifact Type Distribution (Per Project Summary)

**Visualization**: Pie chart or donut chart
- Segments: File types
- Values: Percentage of total size
- One chart per project

**What it shows**: Which artifact types consume the most storage.

### Option 6: Artifact Summary Statistics (Per Project)

**Visualization**: Table
- Rows: Branches
- Columns:
  - Total Artifacts
  - Total Size (MB)
  - Archive Count
  - Test Artifacts (junit + jacoco)
  - Trace Logs
  - Average Size per Artifact

**What it shows**: Complete statistical overview per branch.

---

## Recommended Approach

### Phase 1: Basic Analysis (Recommended to start)

1. **Create per-project artifact size visualization** (Option 3)
   - Simple bar chart showing total size per branch
   - Easy to understand and compare
   - Shows overall artifact footprint

2. **Create per-project artifact count by type** (Option 1)
   - Stacked bar showing how many artifacts per type
   - Reveals artifact generation patterns

3. **Create summary statistics CSV** (Option 6)
   - Comprehensive table for detailed analysis
   - Can be used for further comparisons

### Phase 2: Advanced Analysis (Optional)

4. **Artifact size by type** (Option 2)
   - Shows which types consume most storage

5. **Artifact generation over time** (Option 4)
   - Shows growth patterns during development

6. **Type distribution summary** (Option 5)
   - High-level overview per project

---

## Key Questions to Answer

1. **Do branches with more pipelines generate more artifacts?**
   - Compare artifact count vs pipeline count

2. **Is there a correlation between artifact size and project success?**
   - Compare artifact metrics with branch success rates

3. **Which branches consume the most storage?**
   - Identify storage-heavy branches

4. **Are test artifacts consistently generated?**
   - Check junit/jacoco presence across pipelines

5. **Type A vs Type B artifact comparison**
   - Do different project types generate different artifact patterns?

---

## Data Quality Notes

- All projects have complete artifact data
- Artifact sizes are in bytes (need conversion to MB for visualization)
- Some artifacts have `expires_at: null` (no expiration)
- File types are well-defined and consistent
- Job stages are consistently labeled (compile, test)

---

## Next Steps

1. **Choose visualization approach** (recommend Phase 1 options)
2. **Create analyzer script** (e.g., `analyze_artifacts.py`)
3. **Generate visualizations** per project
4. **Create summary statistics** across all projects
5. **Compare Type A vs Type B** (optional future comparison)

---

**Document Created**: 2025-10-22
**Status**: Ready for implementation
