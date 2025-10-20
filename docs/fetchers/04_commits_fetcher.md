# Commits Fetcher

## Zweck
Lädt Commits auf **zwei verschiedene Arten**, um unterschiedliche Analysezwecke zu unterstützen.

## Exportierte Daten

Der Fetcher erstellt **ZWEI** Output-Dateien:

### 1. All Commits (`all_commits.json`)
- **Inhalt:** ALLE Commits aus ALLEN Branches
- **Verwendung:** Repository-Historie, Commit-Statistiken

### 2. Commits by MR (`commits_by_mr.json`)
- **Inhalt:** Dictionary mit Commits pro Merge Request
- **Verwendung:** Issue/Branch-Zuordnung, Entwicklungsdauer

## Datenstrukturen

### All Commits
```json
[
  {
    "id": "a1b2c3d4e5f6...",
    "short_id": "a1b2c3d4",
    "title": "feat: Implement fleet placement",
    "message": "feat: Implement fleet placement\n\nDetailed description...",
    "author_name": "Max Mustermann",
    "author_email": "max@example.com",
    "authored_date": "2025-10-14T10:30:00Z",
    "committed_date": "2025-10-14T10:30:00Z",
    "parent_ids": ["parent1", "parent2"],
    "web_url": "https://gitlab.../commit/a1b2c3d4"
  }
]
```

### Commits by MR
```json
{
  "1": [commit1, commit2, commit3],
  "2": [commit4, commit5],
  "3": [commit6]
}
```
**Key:** MR IID (interne ID) als String
**Value:** Array von Commits für diesen MR

## Wie funktioniert der Export?

### 1. All Commits
**Endpoint:** `GET /projects/:id/repository/commits`
**Parameter:**
- `all: true` - Commits aus ALLEN Branches (nicht nur default)
- `order: default` - Chronologische Reihenfolge
- `with_stats: false` - Keine Zeilen-Statistiken (spart Zeit)

### 2. Commits by MR
**Endpoint:** `GET /projects/:id/merge_requests/:mr_iid/commits`
**Methode:** Iteriert über alle MRs und lädt deren Commits einzeln

## Verwendung

```python
from fetchers.commits_fetcher import CommitsFetcher

fetcher = CommitsFetcher(gitlab_client)

# Variante 1: Beide Datentypen laden
result = fetcher.fetch(project_id='123', merge_requests=mrs)
all_commits = result['all_commits']
commits_by_mr = result['commits_by_mr']

# Variante 2: Nur alle Commits
all_commits = fetcher.fetch_all_commits(project_id='123')

# Variante 3: Nur Commits pro MR
commits_by_mr = fetcher.fetch_commits_by_mr(project_id='123', merge_requests=mrs)
```

## Besonderheiten
- **Zwei Datenquellen:** Unterschiedliche API-Endpoints für unterschiedliche Zwecke
- **MR-Zuordnung:** `commits_by_mr` ermöglicht Branch/Issue-Tracking
- **Vollständige Historie:** `all_commits` erfasst auch Commits ohne MR

## Verwendung in der Analyse
- **Issue-Dauer:** Erster bis letzter Commit pro Issue (via commits_by_mr)
- **Commit-Häufigkeit:** Anzahl Commits pro Branch/Issue
- **Entwicklungsaktivität:** Zeitliche Verteilung der Commits
- **Commit-Nachrichten:** Conventional Commits analysieren
