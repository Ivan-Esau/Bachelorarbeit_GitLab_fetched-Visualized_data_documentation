# Artifacts Fetcher

## Zweck
Sammelt Metadaten über alle Artifacts (Build-Ausgaben wie Coverage-Reports, Test-Results, etc.) **OHNE** die Dateien herunterzuladen.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Artifact-Metadaten
- **Output-Datei:** `artifacts.json`

### Enthaltene Felder
- `artifact_id` - Eindeutige ID (z.B. "5865_archive", "5865_artifact_0")
- `filename` - Dateiname (z.B. "coverage.xml", "artifacts.zip")
- `size` - Dateigröße in Bytes
- `file_type` - Typ (jacoco, cobertura, archive, junit, ...)
- `file_format` - Format (z.B. "gzip", "zip")
- `created_at` - Erstellungszeitpunkt
- `expires_at` - Ablaufdatum
- `job_id`, `job_name`, `job_stage`, `job_status` - Zugehöriger Job
- `pipeline_id`, `pipeline_status` - Zugehörige Pipeline

## Wie funktioniert der Export?

### Datenquelle
**KEINE direkten API-Calls!** Nutzt bereits geladene Pipeline-Daten.

### Prozess
1. Iteriert über alle Pipelines (aus `pipelines.json`)
2. Für jede Pipeline: Iteriert über alle Jobs
3. Für jeden Job: Prüft ob Artifacts vorhanden
4. Extrahiert Metadaten aus zwei möglichen Formaten:
   - `artifacts_file` (älteres API-Format)
   - `artifacts` Array (neueres API-Format mit Typ-Informationen)

### Zwei API-Formate

#### Format 1: artifacts_file (Archive)
```json
{
  "artifacts_file": {
    "filename": "artifacts.zip",
    "size": 123456
  }
}
```
→ Erstellt Artifact mit `file_type: "archive"`

#### Format 2: artifacts Array (Typed)
```json
{
  "artifacts": [
    {
      "file_type": "jacoco",
      "filename": "coverage.xml",
      "size": 5432,
      "file_format": "gzip"
    }
  ]
}
```
→ Erstellt Artifact mit spezifischem `file_type`

## Verwendung

```python
from fetchers.artifacts_fetcher import ArtifactsFetcher

fetcher = ArtifactsFetcher(gitlab_client)

# Benötigt Pipeline-Daten als Input
artifacts = fetcher.fetch(
    project_id='123',
    pipelines=pipelines  # Von PipelinesFetcher
)

# Statistiken berechnen
stats = fetcher.get_artifact_statistics(artifacts)
print(stats)
# {
#   'total_artifacts': 45,
#   'total_size_mb': 12.5,
#   'by_type': {'jacoco': 10, 'cobertura': 5, 'archive': 30},
#   'by_stage': {'test': 15, 'compile': 30},
#   'by_status': {'success': 40, 'failed': 5}
# }
```

## Besonderheiten
- **Keine Downloads:** Nur Metadaten, spart Bandbreite und Speicher
- **Zwei Formate:** Unterstützt beide GitLab API-Formate
- **Coverage-Filter:** Identifiziert Coverage-Artifacts für Extractor
- **Pipeline-Abhängigkeit:** Benötigt bereits geladene Pipeline-Daten

## Verwendung in der Analyse
- **Coverage-Identifikation:** Filter für `file_type: jacoco/cobertura`
- **Artifact-Statistiken:** Anzahl, Größe, Typen
- **Job-Analyse:** Welche Jobs erzeugen Artifacts?
- **Speicherverbrauch:** Gesamtgröße tracken
- **Basis für Coverage-Extraktion:** Input für CoverageExtractor
