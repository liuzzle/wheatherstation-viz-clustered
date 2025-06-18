# INSPO:
# k-means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# agglomerative clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# 2x2 grid layout: https://community.plotly.com/t/display-graphs-side-by-side-in-a-2x2-grid-2-columns-and-2-rows/50782
# https://github.com/Moh-Ozzi/dash-interactive-filtering-/blob/main/app.py

# Imports
from dash import Dash, html, dcc, Output, Input
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('meteo_wop_10min.csv')
print("Size before dropping NaN values:", len(df))

# The dataset contains multiple NaN values. Removing them
df = df.dropna(subset=['T10m', 'RH10m', 'p10m', 'p_NN10m'])
print("Size after dropping", len(df))

# Reset the index and add it as a column
df = df.reset_index()  
# Add the index as a new column
df['point_id'] = df.index 

features = ['T10m', 'RH10m', 'p10m', 'p_NN10m']
# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Perform k-means 
kmeans = KMeans(n_clusters=3)
df['kmeans_cluster'] = kmeans.fit_predict(scaled_features)

# Perform Agglomerative Clustering 
agg_cluster = AgglomerativeClustering(n_clusters=3)
df['agg_cluster'] = agg_cluster.fit_predict(scaled_features)

print(df) # to check the clustering results

# Create a Dash app
app = Dash(__name__)
app.title = "Weather Station Davos Wolfgang Dataset"

var_options = [
    {'label': 'Temperature (°C)', 'value': 'T10m'},
    {'label': 'Relative Humidity (%)', 'value': 'RH10m'},
    {'label': 'Pressure (hPa) at 10 m above ground', 'value': 'p10m'},
    {'label': 'Pressure (hPa) at 10 m above sea level', 'value': 'p_NN10m'}]

# 2x2 grid by creating a div for each scatter plot
app.layout = html.Div([
    html.H1("Clustering Comparison: K-Means vs Agglomerative Clustering"),
    # first dropdown
    html.Div([
        html.Div([
            "Select variable for first scatter plot (y-axis)",
            dcc.Dropdown(
                id='dropdown-1',
                options=var_options,
                value='T10m',
                placeholder='Select Y-axis variable for first plot',
            )
        ], style={'display': 'inline-block'}),
    ]),
    # first row of scatter plots
    html.Div([
        dcc.Graph(id='kmeans-1', selectedData={'points': []}),
        dcc.Graph(id='agg-1', selectedData={'points': []}),
        ], style={'display': 'flex'}),
    # second dropdown
    html.Div([
        html.Div([
        "Select variable for second scatter plot (y-axis)",
        dcc.Dropdown(
            id='dropdown-2',
            options=var_options,
            value='RH10m',
            placeholder='Select Y-axis variable for second plot'
        )
        ], style={'display': 'inline-block'}),
    ]),
    # second row of scatter plots
    html.Div([
        dcc.Graph(id='kmeans-2'),
        dcc.Graph(id='agg-2'),
    ], style={'display': 'flex'}),
])  

# Callback to update the scatter plots based on selected variables and data
@app.callback(
    Output('kmeans-1', 'figure'),
    Output('agg-1', 'figure'),
    Output('kmeans-2', 'figure'),
    Output('agg-2', 'figure'),
    Input('dropdown-1', 'value'),
    Input('dropdown-2', 'value'),
    Input('kmeans-1', 'selectedData')
)

# update the scatter plots based on selected variables and data
def update_graph(var1, var2, selectedData):
    # Check if any points are selected
    if selectedData and 'points' in selectedData and len(selectedData['points']) > 0:
        # Get the selected points
        selected_indices = [p['pointIndex'] for p in selectedData['points']]
        # Create a new DataFrame with the selected points
        selected_df = df.iloc[selected_indices].copy()
        # Create a new DataFrame with the rest of the points
        rest_df = df.drop(selected_df.index)

        # Scale the selected data
        scaled_sel = scaler.transform(selected_df[features])
        # Reclustering: Perform clustering on the selected data
        selected_df['kmeans_cluster'] = KMeans(n_clusters=3).fit_predict(scaled_sel)
        selected_df['agg_cluster'] = AgglomerativeClustering(n_clusters=3).fit_predict(scaled_sel)

        # Assign -1 to the rest of the data
        rest_df['kmeans_cluster'] = -1
        rest_df['agg_cluster'] = -1
        # Concatenate the selected and rest data
        df_plot = pd.concat([selected_df, rest_df])
    # If no points are selected, use the original DataFrame
    else:
        df_plot = df.copy()

    # get the label for the selected variable
    def get_label(value):
        for option in var_options:
            if option['value'] == value:
                return option['label']

    # create scatter plot
    def scatter_fig(y_col, cluster_col, title):
        # Convert cluster column to string and set -1 to grey
        df_plot[cluster_col] = df_plot[cluster_col].astype(str)
        df_plot.loc[df_plot[cluster_col] == '-1', cluster_col] = 'grey'
        return px.scatter(
            df_plot, x='point_id', y=y_col, color=cluster_col,
            title=title, custom_data=['point_id'], color_discrete_map={'grey': 'lightgrey'},
        ).update_traces(marker=dict(size=5)).update_layout(dragmode='lasso')
        
    return (
        scatter_fig(var1, 'kmeans_cluster', f'K-Means: {get_label(var1)}'),
        scatter_fig(var1, 'agg_cluster', f'Agglomerative: {get_label(var1)}'),
        scatter_fig(var2, 'kmeans_cluster', f'K-Means: {get_label(var2)}'),
        scatter_fig(var2, 'agg_cluster', f'Agglomerative: {get_label(var2)}')
    )

# Run the app   
if __name__ == '__main__':
    app.run(debug=True, port=5000)