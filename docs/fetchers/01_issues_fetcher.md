# Issues Fetcher

## Zweck
Lädt alle Issues (Aufgaben/Tickets) eines GitLab-Projekts herunter.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Issue-Objekten
- **Output-Datei:** `issues.json`

### Enthaltene Felder
- `id`, `iid` - Globale und interne Issue-ID
- `title`, `description` - Titel und Beschreibung
- `state` - Status (opened, closed)
- `created_at`, `updated_at`, `closed_at` - Zeitstempel
- `author`, `assignees` - Beteiligte Personen
- `labels` - Tags/Kategorien
- `milestone` - Zugeordneter Meilenstein
- `time_stats` - Zeiterfassung (time_estimate, total_time_spent)

## Wie funktioniert der Export?

### GitLab API Endpoint
```
GET /projects/:id/issues
```

### Parameter
- `scope: all` - Alle Issues (offen und geschlossen)
- `order_by: created_at` - Sortierung nach Erstelldatum
- `sort: asc` - Aufsteigende Reihenfolge

### Pagination
Verwendet automatische Pagination via `get_paginated()` um alle Issues zu laden.

## Verwendung

```python
from fetchers.issues_fetcher import IssuesFetcher

fetcher = IssuesFetcher(gitlab_client)
issues = fetcher.fetch(project_id='123')

# Ausgabe: Liste von Issue-Dictionaries
# Beispiel: [{
#   'id': 658,
#   'iid': 1,
#   'title': 'US-01: Place Fleet on a Concealed Grid',
#   'state': 'closed',
#   'created_at': '2025-10-13T22:09:47.000Z',
#   ...
# }]
```

## Besonderheiten
- Lädt **ALLE** Issues (nicht nur offene)
- Chronologische Sortierung (älteste zuerst)
- Inkludiert geschlossene Issues mit Schließdatum

## Verwendung in der Analyse
- Issue-Dauer berechnen (created_at → closed_at)
- Anzahl Issues pro Projekt
- Issue-Labels analysieren
