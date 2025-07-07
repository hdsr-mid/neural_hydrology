import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

#%%
# Read the CSV file with geometry information
afvoergebieden_df = gpd.read_file('data/attributes/polders_data_aangevuld.csv')

# Convert the geom_simple column from string to geometry
afvoergebieden_df['geometry'] = gpd.GeoSeries.from_wkt(afvoergebieden_df['geom_simple'])

# Create a GeoDataFrame
afvoergebieden_df = gpd.GeoDataFrame(afvoergebieden_df, geometry='geometry', crs='EPSG:4326')

# Remove column VALUE
afvoergebieden_df = afvoergebieden_df.drop(columns=['VALUE', 'geom_simple'])

# Convert columns to numeric format
numeric_columns = ['maaiveldhoogte', 'maaiveldhoogte_median', 'maaiveldhoogte_p95_minus_p05',
                  'maaiveldhoogte_iqr', 'water_percentage', 'stedelijk_percentage',
                  'infiltratie', 'permabiliteit', 'oppervlak', 'water_opp', 'stedelijk_opp',
                  'klei1_zand0','stuw']

for column in numeric_columns:
    afvoergebieden_df[column] = pd.to_numeric(afvoergebieden_df[column], errors='coerce')

# convert landgebruik and bodem columns to category
afvoergebieden_df['landgebruik'] = afvoergebieden_df['landgebruik'].astype('category')
afvoergebieden_df['bodem'] = afvoergebieden_df['bodem'].astype('category')
afvoergebieden_df["stuw"] = afvoergebieden_df["stuw"].astype('category')

# Add column bodemtype based on klei1_zand0, set to 'klei' if 1, else 'zand'
afvoergebieden_df['bodemtype'] = np.where(afvoergebieden_df['klei1_zand0'] == 1, 'klei', 'zand')

# Set for column stuw the value "gestuwd" if the value is 1, else "bemalen"
afvoergebieden_df['afvoertype'] = np.where(afvoergebieden_df['stuw'] == 1, 'gestuwd', 'bemalen')

# Remove the columns: klei1_zand0, klei, veen, zand and stuw from df and numeric_columns
numeric_columns = [column for column in numeric_columns if column not in ['klei1_zand0', 'klei', 'veen', 'zand', 'stuw']]
afvoergebieden_df = afvoergebieden_df.drop(columns=['klei1_zand0', 'klei', 'veen', 'zand', 'stuw'])

# Ensure the geometry index is of type int64
afvoergebieden_df.index = afvoergebieden_df.index.astype('int64')

# Transform the geometry to EPSG:28992
afvoergebieden_df = afvoergebieden_df.to_crs('EPSG:28992')

# read test results from test_metrics.csv in folder runs/development_run_1303_180736_run1/test/model_epoch100
test_results = pd.read_csv('runs/development_run_2003_145410_run13_rerun/test/model_epoch100/test_metrics.csv')

# Add column NSE_1h to afvoergebieden_df based on the column basin in test_results and SHAPE_ID in afvoergebieden_df
afvoergebieden_df['NSE_1h'] = afvoergebieden_df['SHAPE_ID'].map(test_results.set_index('basin')['NSE_1h'])
numeric_columns.append('NSE_1h')

# Combine numeric and categorical columns for the dropdown
plot_columns = numeric_columns + ['landgebruik', 'bodem', 'bodemtype', 'afvoertype']
dropdown = widgets.Dropdown(
    options=plot_columns,
    value='maaiveldhoogte',
    description='Variable:',
)

def plot_map(variable):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Check if the variable is categorical or numeric
    if variable in ['landgebruik', 'bodem', 'bodemtype', 'afvoertype']:
        # Use a discrete colormap for categorical variables
        afvoergebieden_df.plot(column=variable, ax=ax, legend=True,
                    legend_kwds={'title': f'{variable}'},
                    cmap='Set3',  # discrete colormap suitable for categories
                    categorical=True,
                    missing_kwds={'color': 'lightgrey'})
    elif variable == 'NSE_1h':
        afvoergebieden_df.plot(column=variable, ax=ax, legend=True,
                legend_kwds={'label': f'{variable}'},
                cmap='RdYlGn',  # Red-Yellow-Green colormap
                vmin=-1, vmax=1,  # Set color range between -1 and 1
                missing_kwds={'color': 'lightgrey'})
    else:
        # Use continuous colormap for numeric variables
        afvoergebieden_df.plot(column=variable, ax=ax, legend=True,
            legend_kwds={'label': f'{variable}'},
            cmap='viridis',
            missing_kwds={'color': 'lightgrey'})
    
    # Add SHAPE_ID labels in the center of each polygon
    for idx, row in afvoergebieden_df.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(row['SHAPE_ID']), 
            color='white', ha='center', va='center', weight='bold', fontsize=9)
    
    # Add map elements
    ax.set_title(f'Afvoergebieden HDSR - {variable}')
    ax.axis('equal')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    plt.show()

# Create the interactive widget
interactive_plot = widgets.interactive(plot_map, variable=dropdown)
display(interactive_plot)
#%%