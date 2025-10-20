# Branches Fetcher

## Zweck
Lädt alle Git-Branches eines Projekts herunter.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Branch-Objekten
- **Output-Datei:** `branches.json`

### Enthaltene Felder
- `name` - Branch-Name (z.B. "feature/issue-1-us-01-place-fleet")
- `merged` - Ob Branch gemerged wurde (true/false)
- `protected` - Ob Branch geschützt ist
- `default` - Ob es der Default-Branch ist
- `commit` - Letzter Commit auf diesem Branch
  - `id`, `short_id` - Commit-Hash
  - `title`, `message` - Commit-Nachricht
  - `author_name`, `author_email` - Autor
  - `created_at` - Commit-Zeitstempel
- `web_url` - GitLab-URL zum Branch

## Wie funktioniert der Export?

### GitLab API Endpoint
```
GET /projects/:id/repository/branches
```

### Parameter
Keine zusätzlichen Parameter - lädt alle Branches.

### Pagination
Verwendet `get_paginated()` für Projekte mit vielen Branches.

## Verwendung

```python
from fetchers.branches_fetcher import BranchesFetcher

fetcher = BranchesFetcher(gitlab_client)
branches = fetcher.fetch(project_id='123')

# Ausgabe: Liste von Branch-Dictionaries
# Beispiel: [{
#   'name': 'feature/issue-1-us-01-place-fleet-on-a-conceal',
#   'merged': true,
#   'default': false,
#   'commit': {
#     'id': 'a1b2c3d4...',
#     'title': 'feat: Implement fleet placement',
#     'created_at': '2025-10-14T10:30:00Z'
#   }
# }]
```

## Besonderheiten
- Inkludiert alle Branches (active und merged)
- Enthält Informationen zum letzten Commit auf jedem Branch
- Zeigt Merge-Status

## Verwendung in der Analyse
- Anzahl erstellter Branches
- Branch-Namenskonventionen prüfen
- Merge-Status analysieren
- Branch-Lebenszyklen via Commits tracken
