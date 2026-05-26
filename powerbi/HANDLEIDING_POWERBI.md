# Handleiding — Power BI-rapport HDSR ensemble-afvoer

Kort document om samen met `ensemble_forecast.pbix` per mail te delen.

## 1. Wat zit er in het pakket?

| Bestand | Beschrijving |
|---------|--------------|
| `ensemble_forecast.pbip` | **Aanbevolen** — volledig datamodel (sampledata) + rapportproject |
| `ensemble_forecast.pbix` | Power BI-rapport (na openen PBIP: *Opslaan als* pbix, of `scripts/build_pbix.ps1` op Windows) |
| `sources/` | Power Query (M), DAX en Deneb-specificatie |
| `sample_data/` | Voorbeelddata (zelfde schema als Unity Catalog) |

**Databron (productie):** `dbw_datascience_tst_weu_001.default.output_forecast`  
**Workspace:** `adb-3159846276042543.3.azuredatabricks.net`  
**Kolommen:** `datetime`, `ensemble_id`, `afvoergeb_id`, `value`

De grafiek volgt de Python-referentie `plot_ensemble_forecast.py`: 30 ensemble-lijnen, banden 5–95% en 25–75%, mediaan.

---

## 2. Vereisten

- [Power BI Desktop](https://powerbi.microsoft.com/desktop/) (recente versie)
- Leesrechten op catalog `dbw_datascience_tst_weu_001`, schema `default`, tabel `output_forecast`
- Toegang tot een **SQL warehouse** in Databricks (voor de connector)
- Optioneel: Power BI Pro om te publiceren naar de Power BI-service

---

## 3. Rapport openen

**Aanbevolen (eerste keer):**

1. Dubbelklik `ensemble_forecast.pbip` in Power BI Desktop.
2. Het model laadt sampledata uit `sample_data/` (werkt offline).
3. Voeg de rapportpagina toe volgens §8.2–8.4 (eenmalig, ca. 10 minuten) als de pagina nog leeg is.
4. **Bestand** → **Opslaan als** → `ensemble_forecast.pbix` (voor mail of collega’s zonder PBIP).

**Productie (Databricks):**

1. Vervang de `FactEnsemble`-bron door de Databricks-connector (§7).
2. Log in met **Microsoft Entra ID** en vul het **HTTP-pad** van het SQL warehouse in (Databricks → **SQL Warehouses** → *Connection details*).

> Het meegeleverde `ensemble_forecast.pbix` is pas compleet na stap 4 hierboven of na `scripts/build_pbix.ps1` (Windows).

---

## 4. Rapport gebruiken

1. Kies linksboven een **afvoergebied** in de slicer (`afvoergeb_id`). Alle unieke gebieden uit de tabel zijn beschikbaar (ca. 40 HDSR-polders).
2. De grafiek toont voor dat gebied:
   - lichtblauwe lijnen: elk ensemblelid;
   - oranje vlakken: 5–95% en 25–75%;
   - zwarte gestippelde lijn: mediaan.
3. Controleer met **AFVG16** of de vorm overeenkomt met de PNG uit `plot_ensemble_forecast.py`.

KPI-tegels (indien aanwezig): laatste mediaan en aantal ensembleleden (verwacht: 30).

---

## 5. Gegevens vernieuwen

Na een nieuwe Databricks-run (`postprocess_ensembles.py`):

1. **Start** → **Gegevens vernieuwen** (of *Refresh*).
2. Wacht tot de import uit `output_forecast` klaar is.

De tabel wordt bij elke postprocess-run volledig vervangen (`TRUNCATE` + reload).

---

## 6. Publiceren en delen

**Via Power BI-service (aanbevolen voor collega’s):**

1. **Start** → **Publiceren** → kies een workspace.
2. Deel het rapport via de service-link (geen pbix nodig).

**Via mail (pbix + deze handleiding):**

- Voeg `ensemble_forecast.pbix` en `HANDLEIDING_POWERBI.md` toe.
- Ontvanger heeft Desktop + Databricks-rechten nodig voor vernieuwen.

**PDF-export (statische slide):** **Bestand** → **Exporteren** → **PowerPoint/PDF**.

---

## 7. Databricks-connector (handmatig instellen)

Als de verbinding ontbreekt of je het rapport zelf opbouwt:

1. **Gegevens ophalen** → **Azure** → **Azure Databricks** (of **Databricks**).
2. **Server host name:** `adb-3159846276042543.3.azuredatabricks.net`
3. **HTTP path:** jouw SQL-warehouse-pad
4. **Catalog:** `dbw_datascience_tst_weu_001` · **Database/schema:** `default` · **Tabel:** `output_forecast`
5. **Gegevensconnectiviteitsmodus:** **Importeren**
6. Hernoem de query naar `FactEnsemble` en plak de M-code uit `sources/FactEnsemble.databricks.pq` indien nodig.

Voeg daarna queries `FactStats` en `DimAfvoergeb` toe (zie `sources/*.pq`).

**Relaties (Modelweergave):**

- `DimAfvoergeb[afvoergeb_id]` → `FactEnsemble[afvoergeb_id]` (1:*)
- `DimAfvoergeb[afvoergeb_id]` → `FactStats[afvoergeb_id]` (1:*)

---

## 8. Rapport zelf bouwen (eenmalig)

Gebruik dit als `ensemble_forecast.pbix` niet opent of je vanaf `ensemble_forecast.pbip` werkt.

### 8.1 Model

1. Open `ensemble_forecast.pbip` in Power BI Desktop (dubbelklik het `.pbip`-bestand).
2. Controleer in **Power Query** drie tabellen: `FactEnsemble`, `FactStats`, `DimAfvoergeb`.
3. Vervang de `FactEnsemble`-bron door de Databricks-connector (§7) of laat sample-CSV staan voor testen.
4. **Sluit en toepassen** → controleer relaties (§7).

### 8.2 Slicer

1. Nieuwe pagina: titel **HDSR ensemble-afvoer (1 uur)**.
2. Visual **Slicer** → veld `DimAfvoergeb[afvoergeb_id]`.
3. Stijl: **Dropdown**, enkelvoudige selectie.

### 8.3 Grafiek (Deneb — aanbevolen)

1. Installeer **Deneb** uit AppSource (*Meer visualisaties*).
2. Voeg Deneb toe; koppel velden:
   - `FactEnsemble`: `datetime`, `ensemble_id`, `afvoergeb_id`, `value`
   - `FactStats`: `datetime`, `afvoergeb_id`, `p05`, `p25`, `median`, `p75`, `p95`
3. Open **Deneb** → **Create new specification** → plak/inspireer op `sources/deneb_ensemble_forecast.json` en bouw vier lagen (lijnen, twee banden, mediaan) zoals in `plot_ensemble_forecast.py`.

### 8.4 Grafiek (alleen standaard-visuals)

1. **Line chart** op `FactEnsemble`: X = `datetime`, Y = `value`, Legend = `ensemble_id` (alle lijnen dezelfde kleur #1f77b4).
2. **Line chart** op `FactStats`: `median`, `p05`, `p95`, `p25`, `p75` op `datetime` (oranje/grijs).
3. Leg beide grafieken over elkaar (zelfde positie) en zet de slicer bovenaan.

### 8.5 Opslaan als pbix

**Bestand** → **Opslaan als** → `ensemble_forecast.pbix`.

Plak DAX uit `sources/measures.dax` voor optionele KPI’s.

---

## 9. Problemen oplossen

| Probleem | Oplossing |
|----------|-----------|
| Geen gebieden in slicer | Vernieuw `FactEnsemble`; controleer tabel `output_forecast` in Databricks |
| Lege grafiek | Controleer of slicer één `afvoergeb_id` heeft geselecteerd |
| Databricks-login mislukt | Gebruik Entra ID; vraag warehouse-rechten na bij beheerder |
| Verkeerde percentielen | `FactStats` moet per `datetime` + `afvoergeb_id` groeperen (zie `sources/FactStats.pq`) |

---

## 10. Contact / broncode

Neuralhydrology-pipeline: `jobs/postprocess_ensembles.py` → Unity Catalog-tabel `output_forecast`.  
Repo-map: `powerbi/`.
