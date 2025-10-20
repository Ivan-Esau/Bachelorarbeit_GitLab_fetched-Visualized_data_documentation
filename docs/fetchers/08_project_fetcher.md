# Project Fetcher (Orchestrator)

## Zweck
**Orchestriert alle einzelnen Fetcher** um vollständige Projekt-Daten zu sammeln. Dies ist der Haupt-Entry-Point für den Daten-Export.

## Exportierte Daten

### Datenstruktur
Erstellt **9 JSON-Dateien** pro Projekt im Ordner `data_raw/{project_name}/`:

1. **`issues.json`** - Alle Issues
2. **`branches.json`** - Alle Branches
3. **`merge_requests.json`** - Alle Merge Requests
4. **`all_commits.json`** - Alle Commits (alle Branches)
5. **`commits_by_mr.json`** - Commits gruppiert nach MR
6. **`pipelines.json`** - Alle Pipelines mit Jobs
7. **`artifacts.json`** - Artifact-Metadaten
8. **`coverage.json`** - Coverage-Informationen
9. **`project_metadata.json`** - Projekt-Metadaten und Statistiken

### Project Metadata Struktur
```json
{
  "project_id": "123",
  "project_name": "ba_project_a01_battleship",
  "fetch_date": "2025-10-20T15:30:00.123456",
  "stats": {
    "total_issues": 5,
    "total_branches": 12,
    "total_merge_requests": 10,
    "total_commits": 156,
    "total_pipelines": 72,
    "total_artifacts": 45,
    "artifact_size_mb": 12.5,
    "jobs_with_coverage": 10,
    "total_api_requests": 234
  }
}
```

## Wie funktioniert der Export?

### Ausführungsreihenfolge

Die Fetcher werden in **spezifischer Reihenfolge** ausgeführt, da manche von anderen abhängen:

```
1. Issues          (unabhängig)
2. Branches        (unabhängig)
3. Merge Requests  (unabhängig)
4. Commits         (benötigt: merge_requests)
5. Pipelines       (unabhängig)
6. Artifacts       (benötigt: pipelines)
7. Coverage        (benötigt: artifacts)
8. Statistics      (benötigt: alle vorherigen)
```

### Koordinierte Ausführung

```python
def fetch_complete_project(project_id, project_name):
    # 1. Issues
    issues = issues_fetcher.fetch(project_id)

    # 2. Branches
    branches = branches_fetcher.fetch(project_id)

    # 3. Merge Requests
    merge_requests = merge_requests_fetcher.fetch(project_id)

    # 4. Commits (both types)
    commits = commits_fetcher.fetch(project_id, merge_requests)

    # 5. Pipelines with jobs
    pipelines = pipelines_fetcher.fetch(project_id)

    # 6. Artifacts (from pipelines)
    artifacts = artifacts_fetcher.fetch(project_id, pipelines)

    # 7. Coverage (simple mode)
    coverage = coverage_extractor.extract_simple_from_artifacts(artifacts)

    # 8. Calculate statistics
    stats = calculate_stats(all_data)

    # 9. Save everything
    save_project_data(output_dir)
```

## Verwendung

```python
from fetchers.project_fetcher import ProjectFetcher
from core.gitlab_client import GitLabClient

# Initialize
client = GitLabClient(token, base_url)
fetcher = ProjectFetcher(client)

# Fetch complete project
project_data = fetcher.fetch_complete_project(
    project_id='123',
    project_name='ba_project_a01_battleship'
)

# Save to disk
fetcher.save(project_data, output_dir='data_raw')
```

## Initialisierte Fetcher

Der Project Fetcher initialisiert alle einzelnen Fetcher:

```python
self.issues_fetcher = IssuesFetcher(client)
self.branches_fetcher = BranchesFetcher(client)
self.merge_requests_fetcher = MergeRequestsFetcher(client)
self.commits_fetcher = CommitsFetcher(client)
self.pipelines_fetcher = PipelinesFetcher(client)
self.artifacts_fetcher = ArtifactsFetcher(client)
self.coverage_extractor = CoverageExtractor(client)
```

## Statistiken

Berechnet automatisch Zusammenfassungen:
- Anzahl aller Entitäten
- Artifact-Größen
- API-Request-Zähler
- Coverage-Jobs

## Ausgabe während Ausführung

```
================================================================================
FETCHING: ba_project_a01_battleship
================================================================================

  Fetching issues...
    Found 5 issues
  Fetching branches...
    Found 12 branches
  Fetching merge requests...
    Found 10 merge requests
  Fetching ALL commits from all branches...
    Found 156 commits (all branches)
  Fetching commits for merge requests...
    Processing MR 1/10
    Processing MR 10/10
  Fetching pipelines...
    Found 72 pipelines
  Fetching pipeline details and jobs...
    Processing pipeline 1/72 (ID: 6201)
    Processing pipeline 10/72 (ID: 6210)
    ...
  Collecting artifact metadata from 72 pipelines...
    Found 45 artifacts from 30/144 jobs

  Summary:
    Issues: 5
    Branches: 12
    Merge Requests: 10
    Total Commits: 156
    Pipelines: 72
    Artifacts: 45 (12.5 MB)
    Jobs with Coverage: 10
    API Requests Made: 234

  Saved 9 files to data_raw/ba_project_a01_battleship/
    - issues.json              (  12.3 KB)
    - branches.json            (  45.6 KB)
    - merge_requests.json      (  23.4 KB)
    - all_commits.json         ( 234.5 KB)
    - commits_by_mr.json       ( 156.7 KB)
    - pipelines.json           ( 567.8 KB)
    - artifacts.json           (  34.2 KB)
    - coverage.json            (   5.6 KB)
    - project_metadata.json    (   2.1 KB)
```

## Besonderheiten
- **Request-Zähler:** Tracked API-Calls pro Projekt (reset bei jedem Projekt)
- **Fehlerbehandlung:** Einzelne Fetcher-Fehler stoppen nicht den Gesamtprozess
- **Abhängigkeiten:** Automatische Weitergabe von Daten zwischen Fetchern
- **Vollständigkeit:** Garantiert konsistente Datenstruktur

## Verwendung in der Pipeline
Typischerweise aufgerufen via `main.py` für alle Projekte:

```python
# Iterate over all projects
for project_id, project_name in projects:
    project_data = project_fetcher.fetch_complete_project(
        project_id, project_name
    )
    project_fetcher.save(project_data, 'data_raw')
```
