import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc




# -----------------------------
# App Initialization
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# -----------------------------
# Load Data
# -----------------------------
#csv_file = "cluster_data.csv"  # <-- replace with your actual CSV
cluster = pd.read_csv(r"https://github.com/gohu00/Green_swan/blob/cae87dfb8dba0a52f977f4b5ba7cfe77e9b6f556/cluster.csv?raw=true")

all_features = [
    'BLI_scaled',
    'carbon_intensity',
    'IMF-Adapted Readiness score_scaled',
    'Vulnerability score_scaled',
    'Biocapacity',
    'Renewable_value_scaled',
    'Mineral_value_scaled',
    "GCP_scaled",
    "Sovereign risk"
]

cluster_names = {
    0: "Emerging Economies",
    1: "Industrialized Economies",
    2: "Extractive Economies",
    3: "Vulnerable Economies but nature rich"
}

color_map = {
    "Industrialized Economies": "#1f77b4",
    "Extractive Economies": "#d62728",
    "Emerging Economies": "#2ca02c",
    "Vulnerable Economies but nature rich": "#ff7f0e"
}

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Country Clustering Dashboard"), width=12)
    ], className="mt-3 mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Features"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': f, 'value': f} for f in all_features],
                value=all_features,
                multi=True
            )
        ], width=6),
        dbc.Col([
            html.Label("Select Number of Clusters"),
            dcc.Slider(
                id='cluster-slider',
                min=2, max=6,
                step=1, value=4,
                marks={i: str(i) for i in range(2, 7)}
            )
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='cluster-graph'), width=12)
    ])
], fluid=True)

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output('cluster-graph', 'figure'),
    Input('cluster-slider', 'value'),
    Input('feature-dropdown', 'value')
)
def update_clusters(n_clusters, selected_features):
    # Handle case where no feature is selected
    if len(selected_features) < 2:
        return px.scatter(title="Select at least 2 features")
    
    X = cluster[selected_features].dropna()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster.loc[X.index, 'cluster'] = kmeans.fit_predict(X)
    
    components = PCA(n_components=2).fit_transform(X)
    cluster.loc[X.index, 'PC1'] = components[:, 0]
    cluster.loc[X.index, 'PC2'] = components[:, 1]
    
    cluster['cluster_name'] = cluster['cluster'].map(lambda c: cluster_names.get(c, f"Cluster {c}"))
    
    cluster['hover'] = (
        "ISO: " + cluster['ISO'] +
        "<br>Cluster: " + cluster['cluster_name'] +
        "<br>Patents: " + cluster['Filed Patents'].astype(str) +
        "<br>Readiness: " + cluster['IMF-Adapted Readiness score_scaled'].round(2).astype(str) +
        "<br>Vulnerability: " + cluster['Vulnerability score_scaled'].round(2).astype(str) +
        "<br>Debt Sustainability: " + cluster['Debt_sustainability'].round(2).astype(str) +
        "<br>Biocapacity: " + cluster['Biocapacity'].round(2).astype(str) +
        "<br>Carbon Intensity: " + cluster['carbon_intensity'].round(2).astype(str) +
        "<br>Renewable Capital: " + cluster['Renewable_value_scaled'].round(2).astype(str) +
        "<br>Mineral Capital: " + cluster['Mineral_value_scaled'].round(2).astype(str)
    )
    
    fig = px.scatter(
        cluster.loc[X.index],
        x='PC1',
        y='PC2',
        color='cluster_name',
        text='ISO',
        hover_name='ISO',
        hover_data={'PC1': False, 'PC2': False, 'cluster_name': False, 'hover': True},
        color_discrete_map=color_map,
        size_max=15
    )
    
    fig.update_traces(textposition='top center', textfont=dict(size=9))
    fig.update_layout(
        title="Country Clustering by Selected Indicators",
        title_font_size=20,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Cluster Group",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='LightGray',
            borderwidth=1
        )
    )
    
    return fig

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
