# Power BI — HDSR ensemble-afvoer

Rapport en bronbestanden voor visualisatie van `dbw_datascience_tst_weu_001.default.output_forecast`.

| Bestand | Doel |
|---------|------|
| [HANDLEIDING_POWERBI.md](HANDLEIDING_POWERBI.md) | Korte gebruikershandleiding (mail) |
| [ensemble_forecast.pbix](ensemble_forecast.pbix) | Power BI-rapport |
| [ensemble_forecast.pbip](ensemble_forecast.pbip) | Projectformaat (model + rapport, beheer) |
| [sources/](sources/) | Power Query, DAX, Deneb |
| [sample_data/](sample_data/) | Voorbeeld-CSV voor lokaal testen |

**Grafiek:** zelfde opzet als `neural_hydrology/src/neural_hydrology/analysis/plot_ensemble_forecast.py`.

**Sample data genereren:**

```bash
python3 powerbi/scripts/generate_sample_data.py
```

**Pbix opnieuw bouwen (Windows, optioneel):** zie `scripts/build_pbix.ps1`.
