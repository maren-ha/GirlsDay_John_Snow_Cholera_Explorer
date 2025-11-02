#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import numpy as np

# Load dataset
df = pd.read_csv('../data/cholera_dataset.csv')

# Widget elements
age_slider = widgets.IntRangeSlider(
    value=[0, 80], min=0, max=80, step=1, description='Age Range:', continuous_update=False
)
gender_dropdown = widgets.Dropdown(
    options=['All', 'Male', 'Female'], value='All', description='Gender:'
)
occupation_dropdown = widgets.Dropdown(
    options=['All'] + list(df['Occupation'].dropna().unique()), value='All', description='Occupation:'
)
pump_dropdown = widgets.Dropdown(
    options=['Pump A', 'Pump B', 'Pump C', 'Pump D'], value='Pump A', description='Pump:'
)
household_size_category_dropdown = widgets.Dropdown(
    options=['All'] + ['1-2', '3-4', '5-6', '7+'], value='All', description='Household Size:'
)
raw_veggies_dropdown = widgets.Dropdown(
    options=['All'] + list(df['Raw Vegetable Consumption'].dropna().unique()), value='All', description='Raw Vegetables:'
)
nearest_pump_dropdown = widgets.Dropdown(
    options=['All'] + ['Pump A', 'Pump B', 'Pump C', 'Pump D'], value='All', description='Nearest Pump:'
)
bar_chart_factors_dropdown = widgets.SelectMultiple(
    options=df.columns.difference(['ID', 'Home Location X', 'Home Location Y', 'Health Status']),
    value=['Gender'], description='Bar Chart Factors:'
)
heatmap_x_dropdown = widgets.Dropdown(
    options=df.columns.difference(['ID', 'Home Location X', 'Home Location Y', 'Health Status']),
    value='Gender', description='Heatmap X:'
)
heatmap_y_dropdown = widgets.Dropdown(
    options=df.columns.difference(['ID', 'Home Location X', 'Home Location Y', 'Health Status']),
    value='Occupation', description='Heatmap Y:'
)
bin_slider = widgets.IntSlider(
    value=5, min=2, max=20, step=1, description='Bin Resolution:'
)

def categorize_household_size(df):
    conditions = [
        (df['Household Size'] <= 2),
        (df['Household Size'] <= 4),
        (df['Household Size'] <= 6),
        (df['Household Size'] > 6)
    ]
    choices = ['1-2', '3-4', '5-6', '7+']
    df['Household Size Category'] = np.select(conditions, choices, default='Unknown')
    return df

df = categorize_household_size(df)

def update_bar_chart(factors, bins):
    plt.figure(figsize=(10, 5))
    df_temp = df.copy()
    for factor in factors:
        df_temp = df_temp.dropna(subset=[factor])
        if df_temp[factor].dtype == 'O':
            grouped = df_temp.groupby([factor, 'Health Status']).size().unstack()
        else:
            df_temp[factor] = pd.to_numeric(df_temp[factor], errors='coerce')
            df_temp[factor] = df_temp[factor].astype(float)
            df_temp[f'{factor}_binned'] = pd.cut(df_temp[factor], bins=bins, duplicates='drop')
            grouped = df_temp.groupby([f'{factor}_binned', 'Health Status']).size().unstack()
            # calculate percentage
            #grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
        grouped.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Risk Factor')
    plt.ylabel('Count')
    plt.title('Stacked Bar Chart of Health Status by Risk Factors')
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
        print("No data available for the selected variables.")
        return
    
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt='d')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Heatmap of Disease Cases by {x} vs {y}')
    plt.show()

def update_scatter_plot(age_range, gender, occupation, pump, household_size_category, raw_vegetables, nearest_pump):
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

    if gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender]
    if occupation != 'All':
        filtered_df = filtered_df[filtered_df['Occupation'] == occupation]
    if household_size_category != 'All':
        filtered_df = filtered_df[filtered_df['Household Size Category'] == household_size_category]
    if raw_vegetables != 'All':
        filtered_df = filtered_df[filtered_df['Raw Vegetables'] == raw_vegetables]
    if nearest_pump != 'All':
        filtered_df = filtered_df[filtered_df['Nearest Pump'] == nearest_pump]
    
    plt.figure(figsize=(10, 5))
    for status in df['Health Status'].unique():
        subset = filtered_df[filtered_df['Health Status'] == status]
        plt.scatter(subset[f'Distance to {pump}'], subset['Age'], label=status, alpha=0.6)

    plt.xlabel(f'Distance to {pump}')
    plt.ylabel('Age')
    plt.title(f'Age vs. Distance to {pump}, colored by Health Status')
    plt.legend()
    plt.show()

display(widgets.interactive(update_bar_chart, 
                            factors=bar_chart_factors_dropdown, 
                            bins=bin_slider)
                            )
display(widgets.interactive(update_heatmap, 
                            x=heatmap_x_dropdown, 
                            y=heatmap_y_dropdown, 
                            bins=bin_slider)
                            )
display(widgets.interactive(update_scatter_plot, 
                            age_range=age_slider, 
                            gender=gender_dropdown, 
                            occupation=occupation_dropdown, 
                            pump=pump_dropdown, 
                            household_size_category=household_size_category_dropdown, 
                            raw_vegetables=raw_veggies_dropdown, 
                            nearest_pump=nearest_pump_dropdown)
                            )


# calculate a statistical test to see if the distance to Pump B is significantly different from Pump A

#from scipy import stats
# Filter the DataFrame for the two pumps
#pump_a_distances = df['Distance to Pump A'].dropna()
#pump_b_distances = df['Distance to Pump B'].dropna()
# Perform an independent t-test
#t_stat, p_value = stats.ttest_ind(pump_a_distances, pump_b_distances)
#print(f"T-statistic: {t_stat}, P-value: {p_value}")