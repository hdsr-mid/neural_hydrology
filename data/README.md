# Data beschrijving

## Overzicht
Deze folder bevat de datasets voor de HDSR afvoervoorspelling. Vanwege de grote bestandsgrootte zijn alleen voorbeeldbestanden meegeleverd.

## Folder structuur

### `attributes/`
Bevat gebiedskenmerken van alle polders:
- `polders_data_aangevuld.csv` - Statische kenmerken per afvoergebied

### `time_series/`
Bevat meteorologische en hydrologische tijdreeksen per polder in NetCDF formaat:
- `AFVG1.nc` - Tijdreeks data voor afvoergebied 1 (voorbeeld)
- `AFVG13.nc` - Tijdreeks data voor afvoergebied 13 (voorbeeld)  
- `AFVG15.nc` - Tijdreeks data voor afvoergebied 15 (voorbeeld)

**Let op**: Voor alle 41 afvoergebieden zijn .nc bestanden beschikbaar. Alleen voorbeelden zijn meegeleverd in deze repository.

## Data variabelen

### Time series (.nc bestanden)
- `precipitation` - Neerslag (mm/dag)
- `potential_evaporation` - Potentiële verdamping (mm/dag)
- `temperature` - Temperatuur (°C)
- `discharge` - Afvoer (m³/s) - target variabele

### Attributes (.csv bestand)
Statische gebiedskenmerken zoals:
- Oppervlakte
- Landgebruik percentages
- Bodemtype verdeling
- Drainage parameters

## Polder lijst
- `hdsr_polders.txt` - Lijst van 40 polders gebruikt in training
- `hdsr_polders_all.txt` - Volledige lijst van 41 polders

## Data gebruik
De time series data wordt gebruikt als input voor de LSTM modellen, waarbij de meteorologische variabelen worden gebruikt om afvoer te voorspellen. De attributes data wordt gebruikt voor statische input features.

Voor toegang tot de volledige dataset, neem contact op met HDSR of de projecteigenaar. 