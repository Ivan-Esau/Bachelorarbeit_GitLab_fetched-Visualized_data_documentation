# GitLab Data Fetchers - Übersicht

Diese Dokumentation beschreibt alle Fetcher-Module, die GitLab-Daten für die Forschungsanalyse extrahieren.

## Architektur

```
BaseFetcher (Abstract)
    ├── IssuesFetcher
    ├── BranchesFetcher
    ├── MergeRequestsFetcher
    ├── CommitsFetcher
    ├── PipelinesFetcher
    ├── ArtifactsFetcher
    └── CoverageExtractor

ProjectFetcher (Orchestrator)
    └── Koordiniert alle obigen Fetcher
```

## Fetcher-Übersicht

| Fetcher | Output-Datei(en) | Beschreibung | API-Calls |
|---------|------------------|--------------|-----------|
| **IssuesFetcher** | `issues.json` | Alle Issues/Tickets des Projekts | ~1 |
| **BranchesFetcher** | `branches.json` | Alle Git-Branches | ~1 |
| **MergeRequestsFetcher** | `merge_requests.json` | Alle Merge Requests | ~1 |
| **CommitsFetcher** | `all_commits.json`<br>`commits_by_mr.json` | Commits (gesamt + pro MR) | ~1 + N |
| **PipelinesFetcher** | `pipelines.json` | CI/CD Pipelines mit Jobs | ~3N |
| **ArtifactsFetcher** | `artifacts.json` | Artifact-Metadaten | 0* |
| **CoverageExtractor** | `coverage.json` | Coverage-Metriken | 0-M** |
| **ProjectFetcher** | 9 Dateien | Orchestriert alle Fetcher | Summe |

*ArtifactsFetcher nutzt bereits geladene Pipeline-Daten
**CoverageExtractor: 0 im Simple-Mode, M im Full-Mode (M = Anzahl Coverage-Artifacts)

## Datenfluss

```
                    ProjectFetcher
                         |
        +----------------+----------------+
        |                |                |
    [Issues]        [Branches]      [Merge Requests]
                                          |
                                     [Commits]
                                    (all + by_mr)
                                          |
                                    [Pipelines]
                                     (+ Jobs)
                                          |
                                    [Artifacts]
                                     (Metadata)
                                          |
                                     [Coverage]
                                    (Simple/Full)
```

## Abhängigkeiten

### Unabhängige Fetcher
Diese können parallel laufen:
- IssuesFetcher
- BranchesFetcher
- MergeRequestsFetcher
- PipelinesFetcher

### Abhängige Fetcher
Diese benötigen Daten von anderen:
- **CommitsFetcher** ← benötigt MergeRequests
- **ArtifactsFetcher** ← benötigt Pipelines
- **CoverageExtractor** ← benötigt Artifacts

## Verwendung

### Einzelner Fetcher
```python
from core.gitlab_client import GitLabClient
from fetchers.issues_fetcher import IssuesFetcher

client = GitLabClient(token, url)
fetcher = IssuesFetcher(client)
issues = fetcher.fetch(project_id='123')
```

### Komplettes Projekt (Empfohlen)
```python
from core.gitlab_client import GitLabClient
from fetchers.project_fetcher import ProjectFetcher

client = GitLabClient(token, url)
fetcher = ProjectFetcher(client)

# Fetch complete project data
project_data = fetcher.fetch_complete_project(
    project_id='123',
    project_name='ba_project_a01_battleship'
)

# Save all files
fetcher.save(project_data, output_dir='data_raw')
```

## Output-Struktur

```
data_raw/
└── ba_project_a01_battleship/
    ├── issues.json              # Issues/Tickets
    ├── branches.json            # Git Branches
    ├── merge_requests.json      # Merge Requests
    ├── all_commits.json         # Alle Commits
    ├── commits_by_mr.json       # Commits pro MR
    ├── pipelines.json           # Pipelines + Jobs
    ├── artifacts.json           # Artifact-Metadaten
    ├── coverage.json            # Coverage-Daten
    └── project_metadata.json    # Statistiken
```

## Detaillierte Dokumentationen

1. [Issues Fetcher](01_issues_fetcher.md) - Issues/Tickets
2. [Branches Fetcher](02_branches_fetcher.md) - Git Branches
3. [Merge Requests Fetcher](03_merge_requests_fetcher.md) - Pull Requests
4. [Commits Fetcher](04_commits_fetcher.md) - Repository Commits
5. [Pipelines Fetcher](05_pipelines_fetcher.md) - CI/CD Pipelines
6. [Artifacts Fetcher](06_artifacts_fetcher.md) - Artifact Metadaten
7. [Coverage Extractor](07_coverage_extractor.md) - Code Coverage
8. [Project Fetcher](08_project_fetcher.md) - Orchestrator

## Performance-Hinweise

### API-Request-Zähler
Jeder Fetcher nutzt den zentralen Request-Counter:
```python
client.reset_request_count()  # Vor neuem Projekt
count = client.get_request_count()  # Nach Fetch
```

### Pagination
Alle Fetcher nutzen automatische Pagination:
- Standard-Pagination über `get_paginated()`
- Lädt alle Seiten automatisch
- Kein manuelles Paging nötig

### Rate Limiting
- GitLab API: ~600 requests/minute
- Automatische Delays zwischen Requests
- Request-Counter für Monitoring

## Typische API-Anfragen pro Projekt

| Projekt-Größe | Ungefähre Requests |
|---------------|-------------------|
| Klein (5 Issues, 50 Pipelines) | ~150 |
| Mittel (10 Issues, 100 Pipelines) | ~300 |
| Groß (20 Issues, 200 Pipelines) | ~600 |

**Hauptfaktor:** Pipelines (3 Requests pro Pipeline)

## Fehlerbehandlung

Alle Fetcher implementieren:
- Try-Catch für API-Fehler
- Logging von Fehlern
- Fortsetzung trotz Einzel-Fehler
- Status-Ausgaben (erfolg/fehlgeschlagen)

## Entwicklung

### Neuen Fetcher hinzufügen

1. Erbe von `BaseFetcher`
2. Implementiere `fetch()` Methode
3. Nutze `self.client` für API-Calls
4. Füge zu `ProjectFetcher` hinzu
5. Erstelle Dokumentation

Beispiel:
```python
from .base_fetcher import BaseFetcher

class MyFetcher(BaseFetcher):
    def fetch(self, project_id: str):
        endpoint = f'/projects/{self._quote_project_id(project_id)}/...'
        return self.client.get_paginated(endpoint)
```
