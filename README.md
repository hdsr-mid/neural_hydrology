# HDSR Afvoervoorspelling met Neural Hydrology

Dit project gebruikt deep learning (LSTM) modellen om afvoeren te voorspellen voor de afvoergebieden van Hoogheemraadschap De Stichtse Rijn (HDSR). Het project is gebaseerd op de [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) bibliotheek.

## Overzicht

Het project bevat experimenten met verschillende LSTM varianten voor het voorspellen van afvoeren in 40 polders/afvoergebieden binnen het beheergebied van HDSR. De modellen gebruiken meteorologische data en gebiedskenmerken om accurate afvoervoorspellingen te maken.

## Projectstructuur

```
neural_hydrology/
├── README.md                 # Dit bestand
├── .gitignore               # Git ignore configuratie
├── configs/                 # Configuratie bestanden
│   ├── experiment_configs/  # Specifieke experiment configuraties
│   └── template_configs/    # Template configuraties
├── scripts/                 # Python scripts
│   ├── training/           # Training scripts
│   └── analysis/           # Analyse en evaluatie scripts
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

## Model varianten

Het project test verschillende LSTM configuraties:

1. **MTSLSTM** - Multi-Timescale LSTM
2. **MTSLSTM + Embedding** - Met embedding layer voor categorische features
3. **MTSLSTM + One-Hot Encoding** - Met one-hot encoded features
4. **Statische Multi-Timescale LSTM** - Varianten met statische features

Elke variant heeft eigen configuratie bestanden in `configs/experiment_configs/`.

## Belangrijkste scripts

### Training
- `local_run_nh.py` - Hoofdscript voor lokale training runs
- `batch_train_single.py` - Batch training voor meerdere experimenten
- `run_model.py` - Basis model training script

### Analyse
- `feature_optimalisatie.py` - Feature selectie en optimalisatie
- `hyperparameter_optimalisatie.py` - Hyperparameter tuning
- `best_model.py` - Evaluatie van beste modellen
- `map_hdsr.py` - Visualisatie van HDSR gebied

## Gebruik

### Installatie
```bash
# Installeer neuralhydrology
pip install neuralhydrology

# Of kloon de repository en installeer lokaal
git clone https://github.com/neuralhydrology/neuralhydrology.git
cd neuralhydrology
pip install -e .
```

### Training
```bash
# Train een model met een specifieke configuratie
python scripts/training/local_run_nh.py --config configs/experiment_configs/1_mtslstm_emb.yml

# Batch training voor meerdere experimenten
python scripts/training/batch_train_single.py
```

### Analyse
```bash
# Feature optimalisatie
python scripts/analysis/feature_optimalisatie.py

# Hyperparameter tuning
python scripts/analysis/hyperparameter_optimalisatie.py
```

## Configuratie

Alle experiment configuraties staan in `configs/experiment_configs/`. Elke configuratie definieert:
- Model architectuur (LSTM variant)
- Input features
- Training parameters
- Data preprocessing
- Output metrics

Voorbeeld configuraties:
- `1_mtslstm_emb.yml` - MTSLSTM met embeddings
- `3_mtslstm.yml` - Basis MTSLSTM
- `10_mtslstm_emb_dyn.yml` - Dynamische variant

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
