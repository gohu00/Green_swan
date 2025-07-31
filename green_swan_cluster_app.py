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

bubble = pd.read_csv(r"https://github.com/gohu00/Green_swan/blob/main/bubble_filling.csv?raw=true")
cluster = cluster.merge(bubble, on="ISO", how="left")

# -----------------------------
# Dimensions and variable groups
# -----------------------------
dimensions = {
    "Macro Stability": ["Sovereign risk", "Debt_sustainability"],
    "Nature": ["Biocapacity", "Renewable_value_scaled", "Mineral_value_scaled"],
    "Green Competitiveness": ["BLI_scaled", "Filed Patents", "GCP_scaled"],
    "Climate Adaptation and vulnerability": [
        "IMF-Adapted Readiness score_scaled",
        "Vulnerability score_scaled",
        "carbon_intensity"
    ]
}

# Variable definitions for the Data Explorer
variable_definitions = {
    'BLI_scaled': 'Better Life Index, scaled score of wellbeing',
    'carbon_intensity': 'CO2 emissions per unit of GDP',
    'IMF-Adapted Readiness score_scaled': 'Country readiness to implement policies, scaled',
    'Vulnerability score_scaled': 'Economic vulnerability index, scaled',
    'Biocapacity': 'Biological capacity of the country (gha)',
    'Renewable_value_scaled': 'Value of renewable resources, scaled',
    'Mineral_value_scaled': 'Value of mineral resources, scaled',
    'GCP_scaled': 'Green competitiveness potential, scaled',
    'Sovereign risk': 'Country sovereign debt risk',
    'Debt_sustainability': 'Assessment of a country’s ability to sustain its debt',
    'Filed Patents': 'Number of patents filed'
}

# Cluster naming and colors
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

# Example lists (adjust as needed)
OECD_members = ["USA", "CAN", "FRA", "DEU", "JPN", "AUS", "GBR", "ESP", "ITA", "KOR"]  # ... add all OECD
BRICS_members = ["BRA", "RUS", "IND", "CHN", "ZAF"]
BRICS_plus_members = BRICS_members + ["EGY", "SAU", "ARE", "IRN", "ARG", "ETH"]

def get_country_group(iso):
    if iso in OECD_members:
        return "OECD"
    elif iso in BRICS_plus_members:
        return "BRICS+"
    elif iso in BRICS_members:
        return "BRICS"
    return "Other"

# Add this column to your dataframe
cluster["Group"] = cluster["ISO"].apply(get_country_group)

# Pre-compute PCA components for clustering
all_features = sum(dimensions.values(), [])

# Compute PCA for all data
X_full = cluster[all_features].dropna()
pca_full = PCA(n_components=2).fit(X_full)
components_full = pca_full.transform(cluster[all_features].fillna(0))  # Fill NAs for full coverage

cluster['PC1'] = components_full[:, 0]
cluster['PC2'] = components_full[:, 1]


# -----------------------------
# App Initialization
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Country Clustering Dashboard",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    dcc.Tabs(id="tabs", value='tab-cluster', children=[
        dcc.Tab(label='Clustering Visualization', value='tab-cluster'),
        dcc.Tab(label='Data Explorer', value='tab-data')
    ]),
    html.Div(id='tabs-content')
], fluid=True)

# -----------------------------
# Tabs content callback
# -----------------------------
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-cluster':
        return dbc.Container([
            # --- Feature selection ---
            dbc.Row([
                dbc.Col([
                    html.Label("Macro Stability"),
                    dcc.Dropdown(
                        id='macro-dropdown',
                        options=[{'label': v, 'value': v} for v in dimensions["Macro Stability"]],
                        value=dimensions["Macro Stability"],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Nature"),
                    dcc.Dropdown(
                        id='nature-dropdown',
                        options=[{'label': v, 'value': v} for v in dimensions["Nature"]],
                        value=dimensions["Nature"],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Green Competitiveness"),
                    dcc.Dropdown(
                        id='green-dropdown',
                        options=[{'label': v, 'value': v} for v in dimensions["Green Competitiveness"]],
                        value=dimensions["Green Competitiveness"],
                        multi=True
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Climate Adaptation & Vulnerability"),
                    dcc.Dropdown(
                        id='climate-dropdown',
                        options=[{'label': v, 'value': v} for v in dimensions["Climate Adaptation and vulnerability"]],
                        value=dimensions["Climate Adaptation and vulnerability"],
                        multi=True
                    )
                ], width=3)
            ], className="mb-4"),

            # --- Visualization mode and filters ---
            dbc.Row([
                dbc.Col([
                    html.Label("Visualization Mode"),
                    dcc.RadioItems(
                        id='viz-mode',
                        options=[
                            {'label': 'Highlight Groups', 'value': 'highlight'},
                            {'label': 'Bubble Size', 'value': 'bubble'}
                        ],
                        value='highlight',
                        inline=True
                    )
                ], width=6),

                dbc.Col([
                    html.Label("Group Filter"),
                    dcc.Dropdown(
                        id='group-filter',
                        options=[
                            {'label': 'All', 'value': 'All'},
                            {'label': 'OECD', 'value': 'OECD'},
                            {'label': 'BRICS', 'value': 'BRICS'},
                            {'label': 'BRICS+', 'value': 'BRICS+'}
                        ],
                        value='All',
                        clearable=False
                    )
                ], width=3),

                dbc.Col([
                    html.Label("Bubble Size Variable"),
                    dcc.Dropdown(
                        id='bubble-variable',
                        options=[
                            {'label': 'CO₂ emissions', 'value': 'CO2_per_capita'},
                            {'label': 'Finance needs', 'value': 'Needs'}
                        ],
                        value='CO2_per_capita',
                        clearable=False
                    )
                ], width=3)
            ], className="mb-4"),

            # --- Cluster slider ---
            dbc.Row([
                dbc.Col([
                    html.Label("Select Number of Clusters"),
                    dcc.Slider(
                        id='cluster-slider',
                        min=2, max=6,
                        step=1, value=4,
                        marks={i: str(i) for i in range(2, 7)}
                    )
                ], width=12)
            ], className="mb-4"),

            # --- Graph output ---
            dbc.Row([
                dbc.Col(dcc.Graph(id='cluster-graph'), width=12)
            ])
        ])

    elif tab == 'tab-data':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Variable"),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': col, 'value': col} for col in variable_definitions.keys()],
                        value=list(variable_definitions.keys())[0],
                        clearable=False
                    ),
                    html.Br(),
                    dash_table.DataTable(
                        id='variable-table',
                        columns=[
                            {"name": "ISO", "id": "ISO"},
                            {"name": "Value", "id": "Value"}
                        ],
                        page_size=15,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
                    )
                ], width=5),
                dbc.Col([
                    dcc.Graph(id='variable-map')
                ], width=7)
            ])
        ])
# -----------------------------
# Cluster graph callback
# -----------------------------
@app.callback(
    Output('cluster-graph', 'figure'),
    Input('cluster-slider', 'value'),
    Input('macro-dropdown', 'value'),
    Input('nature-dropdown', 'value'),
    Input('green-dropdown', 'value'),
    Input('climate-dropdown', 'value'),
    Input('viz-mode', 'value'),
    Input('group-filter', 'value'),
    Input('bubble-variable', 'value')
)
def update_clusters(n_clusters, macro_vars, nature_vars, green_vars, climate_vars, viz_mode, group_filter, bubble_var):
    selected_features = (macro_vars or []) + (nature_vars or []) + (green_vars or []) + (climate_vars or [])
    if len(selected_features) < 2:
        return px.scatter(title="Select at least 2 features")

    df = cluster.copy()
    X = df[selected_features].dropna()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[X.index, 'cluster'] = kmeans.fit_predict(X)
    df['cluster_name'] = df['cluster'].map(lambda c: cluster_names.get(c, f"Cluster {c}"))

    if viz_mode == 'highlight':
        # --- Highlight Mode ---
        df['opacity'] = 1.0
        if group_filter != 'All':
            df['opacity'] = df['Group'].apply(lambda g: 1.0 if g == group_filter else 0.2)
        df['marker_size'] = 6

        fig = px.scatter(
            df, x='PC1', y='PC2',
            color='cluster_name',
            text='ISO',
            hover_name='ISO',
            color_discrete_map=color_map
        )
        fig.update_traces(
            marker=dict(
                size=df['marker_size'],
                opacity=df['opacity']
            )
        )

    else:
        # --- Bubble Mode ---
        df['marker_size'] = df[bubble_var].fillna(0.1)  # avoid zeros
        fig = px.scatter(
            df, x='PC1', y='PC2',
            color='cluster_name',
            text='ISO',
            hover_name='ISO',
            size='marker_size',
            color_discrete_map=color_map,
            size_max=25
        )

    fig.update_traces(textposition='top center', textfont=dict(size=9))
    fig.update_layout(
        title=f"Country Clustering ({'Highlight' if viz_mode == 'highlight' else 'Bubble'} mode)",
        title_font_size=20,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Cluster Group",
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='LightGray', borderwidth=1)
    )

    return fig




# -----------------------------
# Data explorer callback
# -----------------------------
@app.callback(
    [Output('variable-table', 'data'),
     Output('variable-map', 'figure')],
    Input('variable-dropdown', 'value')
)
def update_data_explorer(selected_variable):
    table_data = cluster[['ISO', selected_variable]].rename(columns={selected_variable: "Value"})
    table_records = table_data.to_dict('records')

    fig = px.choropleth(
        cluster,
        locations="ISO",
        color=selected_variable,
        hover_name="ISO",
        color_continuous_scale="turbid",
        title=f"{selected_variable} (2023)",
        labels={selected_variable: selected_variable}
    )

    fig.update_geos(showcountries=True)
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        plot_bgcolor='white'
    )

    return table_records, fig

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
