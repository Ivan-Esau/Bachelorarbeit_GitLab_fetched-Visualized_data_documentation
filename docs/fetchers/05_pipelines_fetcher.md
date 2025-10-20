# Pipelines Fetcher

## Zweck
Lädt alle CI/CD Pipelines mit ihren Jobs (Build, Test, etc.) herunter.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Pipeline-Objekten (inkl. eingebettete Jobs)
- **Output-Datei:** `pipelines.json`

### Enthaltene Felder (Pipeline-Level)
- `id`, `iid` - Globale und interne Pipeline-ID
- `status` - Status (success, failed, canceled, skipped)
- `ref` - Branch-Name (z.B. "feature/issue-1-...")
- `sha` - Commit-Hash
- `created_at`, `updated_at`, `started_at`, `finished_at` - Zeitstempel
- `duration` - Laufzeit in Sekunden (kann NULL sein bei canceled)
- `queued_duration` - Wartezeit bis Start
- `web_url` - GitLab-URL zur Pipeline
- **`jobs`** - Array mit allen Jobs dieser Pipeline (siehe unten)

### Enthaltene Felder (Job-Level)
Jede Pipeline enthält ein `jobs` Array mit Job-Objekten:
- `id` - Job-ID
- `name` - Job-Name (z.B. "compile_job", "test_job")
- `stage` - Stage (z.B. "compile", "test")
- `status` - Job-Status (success, failed, canceled)
- `created_at`, `started_at`, `finished_at` - Zeitstempel
- `duration` - Laufzeit in Sekunden
- `coverage` - Code-Coverage-Prozentsatz (falls vorhanden)
- `artifacts` - Array mit Artifact-Informationen
- `artifacts_file` - Artifact-Datei-Metadaten
- `web_url` - GitLab-URL zum Job

## Wie funktioniert der Export?

### Mehrstufiger Prozess

#### Schritt 1: Basis-Pipelines laden
**Endpoint:** `GET /projects/:id/pipelines`
**Parameter:**
- `order_by: id` - Sortierung nach ID
- `sort: asc` - Aufsteigende Reihenfolge

→ Liefert Liste aller Pipelines (ohne Jobs)

#### Schritt 2: Pipeline-Details laden
**Endpoint:** `GET /projects/:id/pipelines/:pipeline_id`

Für jede Pipeline einzeln:
- Holt detaillierte Informationen (duration, started_at, etc.)

#### Schritt 3: Jobs laden
**Endpoint:** `GET /projects/:id/pipelines/:pipeline_id/jobs`

Für jede Pipeline einzeln:
- Lädt alle Jobs mit vollständigen Metadaten
- Inkludiert Artifact-Informationen

#### Schritt 4: Kombinieren
Kombiniert Pipeline-Daten mit Jobs zu einem einzigen Objekt:
```json
{
  "id": 123,
  "status": "success",
  "duration": 45,
  "jobs": [job1, job2, ...]
}
```

## Verwendung

```python
from fetchers.pipelines_fetcher import PipelinesFetcher

fetcher = PipelinesFetcher(gitlab_client)

# Vollständige Pipelines mit Jobs
pipelines = fetcher.fetch(project_id='123')

# Einzelne Methoden
pipelines_basic = fetcher.fetch_pipelines_basic(project_id='123')
details = fetcher.fetch_pipeline_details(project_id='123', pipeline_id=456)
jobs = fetcher.fetch_pipeline_jobs(project_id='123', pipeline_id=456)
```

## Besonderheiten
- **Drei API-Calls pro Pipeline:** Basic → Details → Jobs
- **Canceled Pipelines:** Haben `duration=None`, Jobs existieren trotzdem
- **Eingebettete Jobs:** Jobs sind direkt in Pipeline-Objekt enthalten
- **Artifact-Metadaten:** Jobs enthalten Artifact-Informationen

## Verwendung in der Analyse
- **Pipeline-Erfolgsraten:** Status-Verteilung (success/failed/canceled)
- **Pipeline-Dauer:** Ausführungszeiten analysieren (ohne canceled)
- **Job-Erfolgsraten:** Build vs. Test Erfolg
- **Branch-Pipelines:** Pipelines pro Issue/Branch zählen
- **Coverage-Tracking:** Via Job-Coverage-Feld
- **Artifact-Analyse:** Basis für Coverage-Extraktion
