# Merge Requests Fetcher

## Zweck
Lädt alle Merge Requests (Pull Requests) eines Projekts herunter.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Merge-Request-Objekten
- **Output-Datei:** `merge_requests.json`

### Enthaltene Felder
- `id`, `iid` - Globale und interne MR-ID
- `title`, `description` - Titel und Beschreibung
- `state` - Status (opened, closed, merged)
- `merged` - Ob MR gemerged wurde (true/false)
- `source_branch`, `target_branch` - Quell- und Ziel-Branch
- `created_at`, `updated_at`, `merged_at`, `closed_at` - Zeitstempel
- `author`, `assignees`, `reviewers` - Beteiligte Personen
- `merge_commit_sha` - Hash des Merge-Commits
- `labels` - Tags
- `web_url` - GitLab-URL zum MR

## Wie funktioniert der Export?

### GitLab API Endpoint
```
GET /projects/:id/merge_requests
```

### Parameter
- `state: all` - Alle MRs (offen, geschlossen, gemerged)
- `order_by: created_at` - Sortierung nach Erstelldatum
- `sort: asc` - Aufsteigende Reihenfolge

### Pagination
Verwendet `get_paginated()` um alle MRs zu laden.

## Verwendung

```python
from fetchers.merge_requests_fetcher import MergeRequestsFetcher

fetcher = MergeRequestsFetcher(gitlab_client)
merge_requests = fetcher.fetch(project_id='123')

# Ausgabe: Liste von MR-Dictionaries
# Beispiel: [{
#   'id': 1234,
#   'iid': 1,
#   'title': 'Resolve "US-01: Place Fleet on a Concealed Grid"',
#   'state': 'merged',
#   'source_branch': 'feature/issue-1-us-01',
#   'target_branch': 'master',
#   'created_at': '2025-10-14T08:00:00Z',
#   'merged_at': '2025-10-14T14:30:00Z'
# }]
```

## Besonderheiten
- Lädt **ALLE** MRs (nicht nur offene/gemergede)
- Chronologische Sortierung
- Enthält vollständige Branch-Namen
- Merge-Zeitstempel für Dauer-Berechnung

## Verwendung in der Analyse
- MR-Dauer berechnen (created_at → merged_at)
- Anzahl MRs pro Issue (via Titel-Verknüpfung)
- Review-Zeit analysieren
- Merge-Status tracken
- Branch → Issue Mapping (via source_branch)
