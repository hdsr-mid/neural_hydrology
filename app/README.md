# NeuralHydrology Dash-app — zonder Unity Catalog-resource

Deze variant leest uitsluitend bestanden uit de eigen Workspace-appmap.

## Werkwijze

1. Zet `app.py`, `app.yaml`, `requirements.txt` en `assets/style.css` in:
   `/Workspace/Shared/neural_hydrology_dash_app`
2. Run in het visualisatie-notebook sectie 10.
3. Die sectie maakt automatisch:
   - `data/rob_export_2022.csv.gz`
   - `data/manifest.csv`
   - `data/nse_1h.csv`
   - `data/models/AFVG*.csv.gz`
4. Maak/deploy daarna een Databricks custom app vanaf:
   `/Workspace/Shared/neural_hydrology_dash_app`

Er hoeft geen Unity Catalog Volume als app-resource toegevoegd te worden.
