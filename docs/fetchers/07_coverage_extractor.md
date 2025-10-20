# Coverage Extractor

## Zweck
**LÄDT und PARST** Coverage-Artifacts (JaCoCo und Cobertura XML), um tatsächliche Coverage-Prozentsätze zu extrahieren.

## Exportierte Daten

### Datenstruktur
- **Format:** JSON-Array mit Coverage-Daten
- **Output-Datei:** `coverage.json`

### Enthaltene Felder
- `job_id`, `job_name`, `job_stage` - Zugehöriger Job
- `pipeline_id` - Zugehörige Pipeline
- `artifact_id` - Referenz zum Artifact
- **`coverage_percentage`** - Gesamtabdeckung in Prozent
- **`lines_covered`** - Anzahl abgedeckte Zeilen
- **`lines_total`** - Anzahl Gesamtzeilen
- **`line_rate`** - Line Coverage (0.0 - 1.0)
- `branches_covered` - Abgedeckte Verzweigungen (optional)
- `branches_total` - Gesamtverzweigungen (optional)
- `branch_rate` - Branch Coverage (optional)
- `complexity` - Code-Komplexität (optional)
- `parse_status` - Status (success, error)
- `parse_error` - Fehlermeldung bei Fehler

## Wie funktioniert der Export?

### Zwei Modi

#### Modus 1: Simple (Standard)
**Funktion:** `extract_simple_from_artifacts(artifacts)`
- Identifiziert Coverage-Artifacts
- Erstellt Liste **OHNE Download/Parsing**
- Nur Metadaten

#### Modus 2: Full Download & Parse
**Funktion:** `extract_from_artifacts(project_id, artifacts)`
- Filtert Coverage-Artifacts (file_type: jacoco/cobertura)
- **Lädt jedes Artifact herunter** via GitLab API
- **Parst XML-Inhalt**
- Extrahiert Coverage-Metriken

### Download-Prozess

1. **Artifact-Pfad bestimmen**
   - JaCoCo: `build/reports/jacoco/test/jacocoTestReport.xml`
   - Cobertura: `coverage.xml`

2. **Download via API**
   - Endpoint: `GET /projects/:id/jobs/:job_id/artifacts/:artifact_path`
   - Dekompression: Automatisch (gzip)

3. **XML-Parsing**
   - JaCoCo-Format: `<report>` → `<counter type="LINE/BRANCH">`
   - Cobertura-Format: `<coverage line-rate="0.85" branch-rate="0.90">`

### JaCoCo Parsing
```xml
<report>
  <counter type="LINE" missed="50" covered="350"/>
  <counter type="BRANCH" missed="5" covered="45"/>
  <counter type="COMPLEXITY" ... />
</report>
```

### Cobertura Parsing
```xml
<coverage line-rate="0.875" branch-rate="0.9" lines-covered="350" lines-valid="400" ...>
</coverage>
```

## Verwendung

```python
from fetchers.coverage_extractor import CoverageExtractor

extractor = CoverageExtractor(gitlab_client)

# Modus 1: Simple (nur Identifikation)
coverage_list = extractor.extract_simple_from_artifacts(artifacts)

# Modus 2: Full (Download + Parse)
coverage_data = extractor.extract_from_artifacts(
    project_id='123',
    artifacts=artifacts  # Von ArtifactsFetcher
)

# Beispiel-Output:
# [{
#   'job_id': 5865,
#   'job_name': 'test',
#   'coverage_percentage': 87.5,
#   'lines_covered': 350,
#   'lines_total': 400,
#   'line_rate': 0.875,
#   'branches_covered': 45,
#   'branches_total': 50,
#   'branch_rate': 0.9,
#   'parse_status': 'success'
# }]
```

## Besonderheiten
- **LÄDT Artifacts herunter** (im Gegensatz zu ArtifactsFetcher)
- **Zwei Formate:** JaCoCo und Cobertura
- **Fehlerbehandlung:** Protokolliert Parse-Fehler
- **Kompression:** Automatische gzip-Dekompression
- **Pfad-Konventionen:** Kennt Standard-Pfade für beide Formate

## Verwendung in der Analyse
- **Coverage-Prozentsätze:** Quantitative Metriken
- **Coverage-Entwicklung:** Über Zeit tracken
- **Branch Coverage:** Zusätzlich zu Line Coverage
- **Job-Vergleich:** Coverage pro Branch/Issue
- **Qualitätsmetriken:** Kombination mit Test-Erfolgsraten

## Performance-Hinweis
⚠️ **Download-intensiv!** Lädt jedes Coverage-Artifact herunter.
- Standardmodus in `project_fetcher.py`: **Simple** (kein Download)
- Full-Modus nur bei Bedarf aktivieren
