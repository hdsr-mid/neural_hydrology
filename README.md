# HDSR Afvoervoorspelling met Neural Hydrology

Dit project gebruikt deep learning (LSTM) modellen om afvoeren te voorspellen voor de afvoergebieden van Hoogheemraadschap De Stichtse Rijn (HDSR). Het project is gebaseerd op de [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) bibliotheek.

## Overzicht

Het project bevat experimenten met verschillende LSTM varianten voor het voorspellen van afvoeren in 40 polders/afvoergebieden binnen het beheergebied van HDSR. De modellen gebruiken meteorologische data en gebiedskenmerken om accurate afvoervoorspellingen te maken.

## Projectstructuur

```
neural_hydrology/
├── README.md                 # Dit bestand
├── .gitignore               # Git ignore configuratie
├── config.yml               # NeuralHydrology run-config (data, model, features, train/valid/test)
├── scripts/                 # Python scripts
│   ├── training/           # Training scripts
│   ├── analysis/           # Analyse en evaluatie scripts
│   └── preprocessing/      # Preprocessing/hulpscripts
├── data/                   # Dataset
│   ├── attributes/         # Gebiedskenmerken van polders
│   ├── time_series/        # Tijdreeks data (voorbeelden)
│   └── hdsr_polders.txt    # Lijst van ids afvoergebieden
└── notebooks/              # Jupyter notebooks voor analyse
```

## Datasets

### Afvoergebieden
Het project werkt met 40 afvoergebieden van HDSR. De lijst staat in `data/hdsr_polders.txt`.

### Data bestanden
- **Attributes**: `polders_data_aangevuld.csv` - Gebiedskenmerken van alle polders
- **Time series**: NetCDF bestanden (`.nc`) met meteorologische en hydrologische tijdreeksen per polder
- **Voorbeelden**: Alleen AFVG1, AFVG13 en AFVG15 zijn meegeleverd als voorbeelden (vanwege bestandsgrootte)
- **Let op**: De volledige `time_series/` (NetCDF) dataset wordt in Databricks gebruikt vanaf een Volume (zie `config.yml`) en zit niet in deze repository.


## Model varianten

Het project test verschillende LSTM configuraties:

1. **MTSLSTM** - Multi-Timescale LSTM
2. **MTSLSTM + Embedding** - Met embedding layer voor categorische features
3. **MTSLSTM + One-Hot Encoding** - Met one-hot encoded features
4. **Statische Multi-Timescale LSTM** - Varianten met statische features

De configuratie van een run staat in `config.yml`. Voor experimenten maak je hiervan doorgaans varianten (bijv. per model/feature-set).

## Belangrijkste scripts

### Training
- `batch_train_single.py` - Batch training voor single LSTM model per afvoergebied
- `run_model.py` - Basis model training script voor een of meerdere configuraties
- `hyperparameter_optimalisatie.py` - Optuna hyperparameter optimalisatie (Databricks + MLflow)
- `batch_train_model.py` - Retrain/ensemble op basis van gekozen HPO trial (Databricks + MLflow)

### Analyse
- `best_model.py` - Evaluatie van beste modellen
- `map_hdsr.py` - Visualisatie van HDSR gebied

## Lokaal gebruik

### Installatie
```bash
# (Aanbevolen) maak een virtuele omgeving en installeer dependencies voor dit project
python -m venv .venv
source .venv/bin/activate
pip install -r neural_hydrology/requirements.txt

# Of kloon de repository en installeer lokaal
git clone https://github.com/neuralhydrology/neuralhydrology.git
cd neuralhydrology
pip install -e .
```

### Training
```bash
# Train een model met de configuratie in neural_hydrology/config.yml
python neural_hydrology/scripts/training/run_model.py

# Batch training voor single LSTM model per afvoergebied
python neural_hydrology/scripts/training/batch_train_single.py
```

### Analyse
```bash
# Evaluatie/overzicht beste modellen
python neural_hydrology/scripts/analysis/best_model.py
```

## Databricks
Deze branch is bedoeld om het NeuralHydrology-framework te draaien op Databricks. De scripts in `scripts/training/` zijn hierop ingericht (paden onder `/Workspace/...` en `/Volumes/...`, en MLflow tracking via `Databricks`).

### Installatie en update van repo in Databricks
- **Repo toevoegen**: gebruik Databricks *Repos* om deze Git-repository te clonen en deze branch te selecteren.
- **Repo updaten**:
  - Werk bij via de *Pull* actie in *Repos* (of wissel van branch indien nodig).
  - Sommige scripts verwijzen naar een vaste plek voor de basisconfig, bv. `BASE_CONFIG = "/Workspace/Shared/neural_hydrology_fork/config.yml"`. Zorg dat dit pad klopt voor de `config.yml`.
- **Cluster dependencies**:
  - Installeer Python libraries vanaf `requirements.txt`.
  - Voor GPU-training: kies een cluster met CUDA/GPU runtime. De scripts schakelen automatisch tussen GPU/CPU op basis van `torch.cuda.is_available()`.

### Uitvoeren van runs voor training (incl. opzet hyperparameter optimalisatie)
#### Training van één run
- **Configuratie**: `neural_hydrology/config.yml` is de centrale config.
  - **Data**: in deze branch wijst `data_dir` naar een Databricks Volume genaamd `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/input`.
  - **Outputs**: Op Databricks schrijft de repo outputs naar een Volume genaamd `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output`.
- **Run starten**: voer `neural_hydrology/scripts/training/run_model.py` uit.
  - Dit script start een run met `config.yml` en maakt daarna test-evaluatie + figuren per polder in de run-folder.

#### Hyperparameter optimalisatie (Optuna + MLflow)
Voor HPO gebruik je `neural_hydrology/scripts/training/hyperparameter_optimalisatie.py`:
- **MLflow**: het script zet `MLFLOW_TRACKING_URI=databricks` en logt naar een experiment onder `/Shared/...`.
- **Belangrijke instellingen in het script**:
  - `BASE_CONFIG`: pad naar de basis `config.yml` die per trial wordt aangepast.
  - `OUTPUT_DIR` en `RUNS_DIR`: outputlocaties op een Volume (standaard onder `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/output`).
  - `N_TRIALS`: aantal Optuna trials.
- **Wat er gebeurt**:
  - Per trial wordt een eigen config geschreven en als MLflow artifact gelogd.
  - NeuralHydrology wordt gestart via `start_run(...)`.
  - De objective haalt validatie-metrics uit TensorBoard logs (tags zoals `valid/mean_nse_1D` en `valid/mean_nse_1h`) en optimaliseert op de maximale gemiddelde NSE.

### Het maken van verwachtingen met een ensemble van neerslag
In deze repo staat nog geen kant-en-klaar “forecast pipeline” script, maar de aanbevolen werkwijze op Databricks is:
- **Stap 1: maak per neerslaglid een forcing-dataset**:
  - Zorg dat je voor elk ensemble member een eigen `time_series` NetCDF per polder beschikbaar maakt (zelfde variabelen/structuur als training), maar met neerslag vervangen door het betreffende member.
  - Schrif elk member weg in een aparte input-folder op een Volume, bv. `/Volumes/dbw_datascience_tst_weu_001/default/data_neuralhydrology/input/ens_001/`, `/ens_002/`, etc.
- **Stap 2: run model-inference per ensemble member**:
  - Gebruik het getrainde model (run-folder/checkpoint) en laat het modeleen verwachitng maken op de forecast-periode per ensemble member.
  - Praktisch: maak per member een kopie/variant van `config.yml` bijv. `config_ens1` waarin `data_dir` wijst naar de betreffende ensemble-map en de test/forecast-periode staat ingesteld.
- **Stap 3: combineer resultaten tot ensemble-statistieken**:
  - Combineer per polder de gesimuleerde afvoer \(Q_{sim}\) van alle leden tot minimaal:
    - ensemble-mean/median
    - percentielen (bv. P05/P50/P95) als onzekerheidsband
  - Sla de geaggregeerde outputs op (bij voorkeur op een Volume) en visualiseer per polder (band + median).

## Configuratie

De run-config staat in `config.yml`. Deze definieert o.a.:
- Model architectuur (LSTM variant)
- Input features
- Training parameters
- Data preprocessing
- Output metrics

## Resultaten

De training resultaten worden opgeslagen in een `runs/` folder (niet meegeleverd vanwege grootte). Elke run bevat:
- Getrainde model checkpoints
- Evaluatie metrics
- Visualisaties
- TensorBoard logs

## Notebooks

- `hyperparameter_importance.ipynb` - Analyse van hyperparameter importance en model performance

## Licentie

Dit project is ontwikkeld voor onderzoek binnen HDSR. Voor gebruik van de neuralhydrology bibliotheek, zie de [originele licentie](https://github.com/neuralhydrology/neuralhydrology/blob/master/LICENSE).

## Referenties

- Kratzert, F., et al. (2019). "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets." Hydrology and Earth System Sciences.
- [NeuralHydrology Documentation](https://neuralhydrology.readthedocs.io/)
