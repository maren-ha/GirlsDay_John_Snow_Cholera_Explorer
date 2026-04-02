import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import numpy as np

from src.data_schema import DATA_DIR

# Datensatz laden
df = pd.read_csv(DATA_DIR / 'cholera_datensatz_de.csv')

# Widget-Elemente
alter_slider = widgets.IntRangeSlider(
    value=[0, 80], min=0, max=80, step=1, description='Altersspanne:', continuous_update=False
)
geschlecht_dropdown = widgets.Dropdown(
    options=['Alle', 'Männlich', 'Weiblich'], value='Alle', description='Geschlecht:'
)
beruf_dropdown = widgets.Dropdown(
    options=['Alle'] + list(df['Beruf'].dropna().unique()), value='Alle', description='Beruf:'
)
pumpe_dropdown = widgets.Dropdown(
    options=['Pumpe A', 'Pumpe B', 'Pumpe C', 'Pumpe D'], value='Pumpe A', description='Pumpe:'
)
haushaltsgröße_kategorie_dropdown = widgets.Dropdown(
    options=['Alle'] + ['1-2', '3-4', '5-6', '7+'], value='Alle', description='Haushaltsgröße:'
)
rohes_gemüse_dropdown = widgets.Dropdown(
    options=['Alle'] + list(df['Rohkost-Konsum'].dropna().unique()), value='Alle', description='Rohes Gemüse:'
)
nächste_pumpe_dropdown = widgets.Dropdown(
    options=['Alle'] + ['Pumpe A', 'Pumpe B', 'Pumpe C', 'Pumpe D'], value='Alle', description='Nächste Pumpe:'
)
balkendiagramm_faktor_dropdown = widgets.SelectMultiple(
    options=df.columns.difference(['ID', 'Wohnort X', 'Wohnort Y', 'Gesundheitsstatus']),
    value=['Geschlecht'], description='Balkendiagramm-Faktoren:'
)
heatmap_x_dropdown = widgets.Dropdown(
    options=df.columns.difference(['ID', 'Wohnort X', 'Wohnort Y', 'Gesundheitsstatus']),
    value='Geschlecht', description='Heatmap X:'
)
heatmap_y_dropdown = widgets.Dropdown(
    options=df.columns.difference(['ID', 'Wohnort X', 'Wohnort Y', 'Gesundheitsstatus']),
    value='Beruf', description='Heatmap Y:'
)
bin_slider = widgets.IntSlider(
    value=5, min=2, max=20, step=1, description='Bin-Auflösung:'
)

def categorize_household_size(df):
    conditions = [
        (df['Haushaltsgröße'] <= 2),
        (df['Haushaltsgröße'] <= 4),
        (df['Haushaltsgröße'] <= 6),
        (df['Haushaltsgröße'] > 6)
    ]
    choices = ['1-2', '3-4', '5-6', '7+']
    df['Haushaltsgröße Kategorie'] = np.select(conditions, choices, default='Unbekannt')
    return df

df = categorize_household_size(df)

def update_bar_chart(factors, bins):
    plt.figure(figsize=(10, 5))
    df_temp = df.copy()
    for factor in factors:
        df_temp = df_temp.dropna(subset=[factor])
        if df_temp[factor].dtype == 'O':
            grouped = df_temp.groupby([factor, 'Gesundheitsstatus']).size().unstack()
        else:
            df_temp[factor] = pd.to_numeric(df_temp[factor], errors='coerce')
            df_temp[factor] = df_temp[factor].astype(float)
            df_temp[f'{factor}_binned'] = pd.cut(df_temp[factor], bins=bins, duplicates='drop')
            grouped = df_temp.groupby([f'{factor}_binned', 'Gesundheitsstatus']).size().unstack()
            # calculate percentage
            #grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
        grouped.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Risikofaktor')
    plt.ylabel('Anzahl')
    plt.title('Gestapeltes Balkendiagramm der Gesundheitszustände')
    plt.show()

def update_heatmap(x, y, bins):
    plt.figure(figsize=(8, 6))
    df_temp = df.copy().dropna(subset=[x, y])
    
    if df_temp[x].dtype == 'O' and df_temp[y].dtype == 'O':
        pivot = df_temp.pivot_table(index=y, columns=x, values='ID', aggfunc='count', fill_value=0)
    else:
        if df_temp[x].dtype != 'O':
            df_temp[x] = pd.cut(df_temp[x], bins=bins, duplicates='drop')
        if df_temp[y].dtype != 'O':
            df_temp[y] = pd.cut(df_temp[y], bins=bins, duplicates='drop')
        pivot = df_temp.pivot_table(index=y, columns=x, values='ID', aggfunc='count', fill_value=0)
    
    if pivot.empty:
        print("Keine Daten für die ausgewählten Variablen.")
        return
    
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt='d')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Heatmap mit der Anzahl der Krankheitsfälle von {x} vs {y}')
    plt.show()

def update_scatter_plot(alter_range, geschlecht, beruf, pumpe, haushaltsgröße_kategorie, rohes_gemüse, nächste_pumpe):
    filtered_df = df[(df['Alter'] >= alter_range[0]) & (df['Alter'] <= alter_range[1])]
    
    if geschlecht != 'Alle':
        filtered_df = filtered_df[filtered_df['Geschlecht'] == geschlecht]
    if beruf != 'Alle':
        filtered_df = filtered_df[filtered_df['Beruf'] == beruf]
    if haushaltsgröße_kategorie != 'Alle':
        filtered_df = filtered_df[filtered_df['Haushaltsgröße Kategorie'] == haushaltsgröße_kategorie]
    if rohes_gemüse != 'Alle':
        filtered_df = filtered_df[filtered_df['Rohkost-Konsum'] == rohes_gemüse]
    if nächste_pumpe != 'Alle':
        filtered_df = filtered_df[filtered_df['Nächstgelegene Pumpe'] == nächste_pumpe]
    
    plt.figure(figsize=(10, 5))
    for status in df['Gesundheitsstatus'].unique():
        subset = filtered_df[filtered_df['Gesundheitsstatus'] == status]
        plt.scatter(subset[f'Entfernung zu {pumpe}'], subset['Alter'], label=status, alpha=0.6)
    
    plt.xlabel(f'Distanz zu {pumpe}')
    plt.ylabel('Alter')
    plt.title(f'Cholera-Schweregrad vs. Distanz zu {pumpe}')
    plt.legend()
    plt.show()

display(widgets.interactive(update_bar_chart, 
                            factors=balkendiagramm_faktor_dropdown, 
                            bins=bin_slider)
                            )
display(widgets.interactive(update_heatmap, 
                            x=heatmap_x_dropdown, 
                            y=heatmap_y_dropdown, 
                            bins=bin_slider)
                            )
display(widgets.interactive(update_scatter_plot, 
                            alter_range=alter_slider, 
                            geschlecht=geschlecht_dropdown, 
                            beruf=beruf_dropdown, 
                            pumpe=pumpe_dropdown, 
                            haushaltsgröße_kategorie=haushaltsgröße_kategorie_dropdown, 
                            rohes_gemüse=rohes_gemüse_dropdown, 
                            nächste_pumpe=nächste_pumpe_dropdown)
                            )
