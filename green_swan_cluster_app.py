import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import itertools
# -----------------------------
# Load Data
# -----------------------------
#csv_file = "cluster_data.csv"  # <-- replace with your actual CSV
cluster = pd.read_csv(r"https://github.com/gohu00/Green_swan/blob/e136e9dcd7d1e453807e846fda408b451fce0c70/cluster.csv?raw=true")

bubble = pd.read_csv(r"https://github.com/gohu00/Green_swan/blob/main/bubble_filling.csv?raw=true")
cluster = cluster.merge(bubble, on="ISO", how="left")
ipd_data = pd.read_csv(r"https://github.com/gohu00/Green_swan/blob/bd0cc5e018f2f402681e7f6dad0fdb23f5c775ed/IdealpointsJuly2025.csv?raw=true")
# -----------------------------
# Dimensions and variable groups
# -----------------------------
dimensions = {
    "Macro Stability": ["Sovereign risk", "Debt_sustainability",'TDG', 'External_Debt_GDP', 'Debt_Export_GDP',
       'Debt_Service_export', 'Debt_Budget_Revenue', 'Short_term_on_gov_debt',
       'Short_term_on_reserves', 'debt_service_UNCTAD',
       'interest_rate_GDP_growth_differential'],
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
    'BLI_scaled': 'Brown Lock-in Index',
    'carbon_intensity': 'CO2 emissions per unit of GDP',
    'IMF-Adapted Readiness score_scaled': 'Country readiness to implement adaptation policies, scaled',
    'Vulnerability score_scaled': 'Economic vulnerability index, scaled',
    'Biocapacity': 'Biological capacity of the country (gha)',
    'Renewable_value_scaled': 'Value of renewable resources, scaled',
    'Mineral_value_scaled': 'Value of mineral resources, scaled',
    'GCP_scaled': 'Green competitiveness potential, scaled',
    'Sovereign risk': 'Country sovereign debt risk',
    'Debt_sustainability': 'Assessment of a country’s ability to sustain its debt',
    'Filed Patents': 'Number of patents filed',
    'TDG': 'Total public and private debt as a percentage of GDP',
    'External_Debt_GDP': 'External debt stock relative to GDP',
    'Debt_Export_GDP': 'Debt stock relative to total exports',
    'Debt_Service_export': 'Annual debt service payments relative to total exports',
    'Debt_Budget_Revenue': 'Debt stock relative to total government budget revenue',
    'Short_term_on_gov_debt': 'Share of short-term debt in total government external debt',
    'Short_term_on_reserves': 'Short-term debt stock relative to foreign exchange reserves',
    'debt_service_UNCTAD': 'Debt service obligations relative to government revenue (UNCTAD definition)',
    'interest_rate_GDP_growth_differential': 'Difference between interest rate on debt and GDP growth rate'
}

variable_labels = {
    "Sovereign risk": "Sovereign Risk (Macro)",
    "Debt_sustainability": "Debt Sustainability",
    "Biocapacity": "Biocapacity Index",
    "Renewable_value_scaled": "Renewable Resource Value",
    "Mineral_value_scaled": "Mineral Resource Value",
    "BLI_scaled": "Better Life Index",
    "Filed Patents": "Number of Filed Green Patents",
    "GCP_scaled": "Green Competitiveness Index",
    "IMF-Adapted Readiness score_scaled": "Climate Readiness Score",
    "Vulnerability score_scaled": "Climate Vulnerability Score",
    "carbon_intensity": "Carbon Intensity",
    'TDG': 'Total Debt/GDP', 
    'External_Debt_GDP': 'External Debt/GDP' , 
    'Debt_Export_GDP': 'Debt/Exports ',
    'Debt_Service_export': 'Debt Service/Exports', 
    'Debt_Budget_Revenue': 'Debt/Budget Revenue', 
    'Short_term_on_gov_debt': "Short-term debt/External Debt",
    'Short_term_on_reserves': "Short-term debt/Reserves ", 
    'debt_service_UNCTAD': "Debt Service/Revenue",
    'interest_rate_GDP_growth_differential': 'Interest Rate/GDP Growth Differential'
}

# Cluster naming and colors
# Cluster naming and colors
cluster_names = {
    0: "Group 1",
    1: "Group 2",
    2: "Group 3",
    3: "Group 4",
    4: "Group 5",
    5: "Group 6"
}

color_map = {
    "Group 1": "#1f77b4",
    "Group 2": "#d62728",
    "Group 3": "#2ca02c",
    "Group 4": "#ff7f0e",
    "Group 5": "#9467bd",
    "Group 6": "#8c564b"
}

# Example lists (adjust as needed)
OECD_members = [
    "AUS",  # Australia
    "AUT",  # Austria
    "BEL",  # Belgium
    "CAN",  # Canada
    "CHL",  # Chile
    "COL",  # Colombia
    "CZE",  # Czech Republic
    "DNK",  # Denmark
    "EST",  # Estonia
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "GRC",  # Greece
    "HUN",  # Hungary
    "ISL",  # Iceland
    "IRL",  # Ireland
    "ISR",  # Israel
    "ITA",  # Italy
    "JPN",  # Japan
    "KOR",  # South Korea
    "LVA",  # Latvia
    "LTU",  # Lithuania
    "LUX",  # Luxembourg
    "MEX",  # Mexico
    "NLD",  # Netherlands
    "NZL",  # New Zealand
    "NOR",  # Norway
    "POL",  # Poland
    "PRT",  # Portugal
    "SVK",  # Slovakia
    "SVN",  # Slovenia
    "ESP",  # Spain
    "SWE",  # Sweden
    "CHE",  # Switzerland
    "TUR",  # Türkiye
    "GBR",  # United Kingdom
    "USA"   # United States
]
BRICS_members = ["BRA", "RUS", "IND", "CHN", "ZAF"]
BRICS_plus_members = BRICS_members + ["EGY", "SAU", "ARE", "IRN", "ARG", "ETH"]
coalition_iso_alpha3 = [
    "AND", "ARG", "AUT", "AUS", "AZE", "BHS", "BHR", "BGD", "BRB", "BEL", "BWA", "BRA", "BFA", "CPV", "KHM", "CMR",
    "CAN", "CHL", "COL", "CRI", "CIV", "HRV", "CYP", "DNK", "DJI", "DOM", "ECU", "EGY", "GNQ", "EST", "SWZ", "ETH",
    "FJI", "FIN", "FRA", "DEU", "GHA", "GRC", "GTM", "HND", "HUN", "ISL", "IDN", "IRQ", "IRL", "ITA", "JAM", "JPN",
    "KAZ", "KEN", "KGZ", "LVA", "LBR", "LTU", "LUX", "MDG", "MYS", "MDV", "MHL", "MEX", "MCO", "MNE", "MAR", "MOZ",
    "NAM", "NLD", "NZL", "NGA", "MKD", "NOR", "PAK", "PAN", "PRY", "PER", "PHL", "POL", "PRT", "COG", "KOR", "RWA",
    "SRB", "SYC", "SLE", "SGP", "SVK", "ESP", "LKA", "SWE", "CHE", "TGO", "TON", "TUR", "UGA", "UKR", "GBR", "URY",
    "UZB", "ZMB"
]
g7_iso3 = ["CAN", "FRA", "DEU", "ITA", "JPN", "GBR", "USA"]
cop_30_mofs = ['BRA', 'FRA', 'MAR', 'FJI', 'COL', 'IND', 'KEN', 'POL', 'ESP', 'ARE',
 'GBR', 'CAN', 'AZE', 'BRB', 'CHL', 'CHN', 'EGY', 'GHA', 'IDN', 'PHL',
 'SAU', 'UGA', 'ZAF']

ngfs_iso_alpha3 = [
    "ALB", "DZA", "AGO", "ARG", "ARM", "AUS", "AUT", "AZE", "BHS", "BHR", "BGD", "BLR", "BEL", "BEN", "BTN", "BOL",
    "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "CPV", "KHM", "CMR", "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM",
    "COG", "CRI", "HRV", "CYP", "CZE", "DNK", "DJI", "DOM", "ECU", "EGY", "SLV", "GNQ", "EST", "SWZ", "ETH", "FJI",
    "FIN", "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", "GUY", "HTI", "HND", "HUN", "ISL",
    "IND", "IDN", "IRN", "IRQ", "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT", "KGZ", "LAO",
    "LVA", "LBN", "LSO", "LBR", "LTU", "LUX", "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX",
    "MDA", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "MKD", "NOR", "OMN",
    "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "RWA", "KNA", "LCA", "VCT",
    "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "ZAF", "KOR", "SSD", "ESP",
    "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TWN", "TJK", "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR",
    "TKM", "UGA", "UKR", "ARE", "GBR", "USA", "UZB", "VUT", "VEN", "VNM", "YEM", "ZMB", "ZWE"
]

boga_iso_alpha3 = [
    "DNK",  # Denmark (co-founder)
    "CRI",  # Costa Rica (co-founder)
    "FRA",  # France
    "SWE",  # Sweden
    "IRL",  # Ireland
    "ESP",  # Spain
    "POR",  # Portugal
    "ITA",  # Italy
    "FIJ",  # Fiji
    "GRD",  # Grenada
    "SLV",  # El Salvador
    "COL",  # Colombia
    "NZL",  # New Zealand
    "QUE"   # Quebec (subnational member)
]
ppca_iso_alpha3 = [
    "AUT", "BEL", "CAN", "CHL", "COL", "CYP", "CZE", "DNK", "ECU", "ESP", "EST", "FJI", "FIN", "FRA",
    "DEU", "GRD", "HUN", "IRL", "ISR", "ITA", "JPN", "LVA", "LTU", "LUX", "MEX", "MDA", "MNG", "MNE",
    "MAR", "NZL", "NOR", "POR", "SVK", "SVN", "SGP", "ZAF", "KOR", "SWE", "CHE", "UKR", "GBR", "USA",
    "URY"
]

ff_npt_iso_alpha3 = [
    "VUT",  # Vanuatu
    "TUV",  # Tuvalu
    "TON",  # Tonga
    "FJI",  # Fiji
    "SLB",  # Solomon Islands
    "NIU",  # Niue
    "ATG",  # Antigua and Barbuda
    "TLS",  # Timor-Leste
    "PLW",  # Palau
    "COL",  # Colombia
    "WSM",  # Samoa
    "NRU",  # Nauru
    "MHL"   # Marshall Islands
]

port_vila_call_iso_alpha3 = [
    "VUT",  # Vanuatu
    "TUV",  # Tuvalu
    "TON",  # Tonga
    "FJI",  # Fiji
    "NIU",  # Niue
    "SLB",  # Solomon Islands
    "PNG"   # Papua New Guinea
]

cnc_iso_alpha3 = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "CAN",  # Canada
    "CHL",  # Chile
    "COL",  # Colombia
    "DEN",  # Denmark
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "ISL",  # Iceland
    "IRL",  # Ireland
    "ITA",  # Italy
    "LUX",  # Luxembourg
    "MEX",  # Mexico
    "NLD",  # Netherlands
    "NZL",  # New Zealand
    "NOR",  # Norway
    "POR",  # Portugal
    "ESP",  # Spain
    "SWE",  # Sweden
    "UKR",  # Ukraine
    "GBR"   # United Kingdom
]

coffis_iso_alpha3 = [
    "ATG",  # Antigua and Barbuda
    "AUT",  # Austria
    "BEL",  # Belgium
    "CAN",  # Canada
    "COL",  # Colombia
    "CRI",  # Costa Rica
    "DNK",  # Denmark
    "FIN",  # Finland
    "FRA",  # France
    "IRL",  # Ireland
    "LUX",  # Luxembourg
    "MHL",  # Marshall Islands
    "NLD",  # Netherlands
    "NZL",  # New Zealand
    "ESP",  # Spain
    "CHE",  # Switzerland
    "GBR"   # United Kingdom
]
gcpa_finance_mission_iso_alpha3 = [
    "AUS",  # Australia
    "BRB",  # Barbados
    "CAN",  # Canada
    "CHL",  # Chile
    "COL",  # Colombia
    "FRA",  # France
    "DEU",  # Germany
    "MAR",  # Morocco
    "NOR",  # Norway
    "TZA",  # Tanzania
    "GBR",  # United Kingdom
    "BRA"   # Brazil
]
sids_energy_transition_iso_alpha3 = [
    "ATG",  # Antigua and Barbuda
    "BHS",  # Bahamas
    "BRB",  # Barbados
    "BLZ",  # Belize
    "CPV",  # Cabo Verde
    "COM",  # Comoros
    "CUB",  # Cuba
    "DMA",  # Dominica
    "DOM",  # Dominican Republic
    "FJI",  # Fiji
    "GRD",  # Grenada
    "GUY",  # Guyana
    "HTI",  # Haiti
    "JAM",  # Jamaica
    "KIR",  # Kiribati
    "MDV",  # Maldives
    "MHL",  # Marshall Islands
    "MUS",  # Mauritius
    "FSM",  # Micronesia (Federated States of)
    "NRU",  # Nauru
    "NIU",  # Niue
    "PLW",  # Palau
    "PNG",  # Papua New Guinea
    "WSM",  # Samoa
    "STP",  # São Tomé and Príncipe
    "SYC",  # Seychelles
    "SLB",  # Solomon Islands
    "LCA",  # Saint Lucia
    "VCT",  # Saint Vincent and the Grenadines
    "SUR",  # Suriname
    "TLS",  # Timor-Leste
    "TON",  # Tonga
    "TTO",  # Trinidad and Tobago
    "TUV",  # Tuvalu
    "VUT"   # Vanuatu
]

v20_iso_alpha3 = [
    "AFG",  # Afghanistan
    "BGD",  # Bangladesh
    "BRB",  # Barbados
    "BTN",  # Bhutan
    "CRI",  # Costa Rica
    "ETH",  # Ethiopia
    "GHA",  # Ghana
    "KEN",  # Kenya
    "KIR",  # Kiribati
    "MDG",  # Madagascar
    "MDV",  # Maldives
    "NPL",  # Nepal
    "PHL",  # Philippines
    "RWA",  # Rwanda
    "LCA",  # Saint Lucia
    "TZA",  # Tanzania
    "TLS",  # Timor-Leste
    "TUV",  # Tuvalu
    "VUT",  # Vanuatu
    "VNM"   # Vietnam
]

eu_iso_alpha3 = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "BGR",  # Bulgaria
    "HRV",  # Croatia
    "CYP",  # Cyprus
    "CZE",  # Czechia
    "DNK",  # Denmark
    "EST",  # Estonia
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "GRC",  # Greece
    "HUN",  # Hungary
    "IRL",  # Ireland
    "ITA",  # Italy
    "LVA",  # Latvia
    "LTU",  # Lithuania
    "LUX",  # Luxembourg
    "MLT",  # Malta
    "NLD",  # Netherlands
    "POL",  # Poland
    "PRT",  # Portugal
    "ROU",  # Romania
    "SVK",  # Slovakia
    "SVN",  # Slovenia
    "ESP",  # Spain
    "SWE"   # Sweden
]

african_union_iso_alpha3 = [
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM", "COG", "CIV", "DJI",
    "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY",
    "MDG", "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC",
    "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA", "ZMB", "ZWE"
]

celac_iso_alpha3 = [
    "ARG",  # Argentina
    "ATG",  # Antigua and Barbuda
    "BHS",  # Bahamas
    "BRB",  # Barbados
    "BLZ",  # Belize
    "BOL",  # Bolivia
    "BRA",  # Brazil
    "CHL",  # Chile
    "COL",  # Colombia
    "CRI",  # Costa Rica
    "CUB",  # Cuba
    "DMA",  # Dominica
    "DOM",  # Dominican Republic
    "ECU",  # Ecuador
    "SLV",  # El Salvador
    "GRD",  # Grenada
    "GTM",  # Guatemala
    "GUY",  # Guyana
    "HTI",  # Haiti
    "HND",  # Honduras
    "JAM",  # Jamaica
    "MEX",  # Mexico
    "NIC",  # Nicaragua
    "PAN",  # Panama
    "PRY",  # Paraguay
    "PER",  # Peru
    "KNA",  # Saint Kitts and Nevis
    "LCA",  # Saint Lucia
    "VCT",  # Saint Vincent and the Grenadines
    "SUR",  # Suriname
    "TTO",  # Trinidad and Tobago
    "URY",  # Uruguay
    "VEN"   # Venezuela
]

asean_iso_alpha3 = [
    "BRN",  # Brunei Darussalam
    "KHM",  # Cambodia
    "IDN",  # Indonesia
    "LAO",  # Lao People's Democratic Republic
    "MYS",  # Malaysia
    "MMR",  # Myanmar
    "PHL",  # Philippines
    "SGP",  # Singapore
    "THA",  # Thailand
    "VNM"   # Vietnam
]

commonwealth_iso_alpha3 = [
    "ATG", "AUS", "BHS", "BGD", "BRB", "BLZ", "BWA", "BRN", "CMR", "CAN", "CYP", "DMA", 
    "FJI", "GMB", "GHA", "GRD", "GUY", "IND", "JAM", "KEN", "KIR", "LSO", "MWI", "MYS", 
    "MDV", "MLT", "MUS", "MOZ", "NAM", "NRU", "NZL", "NGA", "PAK", "PNG", "RWA", "KNA", 
    "LCA", "VCT", "WSM", "SYC", "SLE", "SGP", "SLB", "ZAF", "LKA", "SWZ", "TZA", "TON", 
    "TTO", "UGA", "GBR", "VUT", "ZMB"
]
sco_iso_alpha3 = [
    "CHN",  # China
    "RUS",  # Russia
    "KAZ",  # Kazakhstan
    "KGZ",  # Kyrgyzstan
    "TJK",  # Tajikistan
    "UZB",  # Uzbekistan
    "IND",  # India
    "PAK",  # Pakistan
    "IRN"   # Iran
]

cis_iso_alpha3 = [
    "ARM",  # Armenia
    "AZE",  # Azerbaijan
    "BLR",  # Belarus
    "KAZ",  # Kazakhstan
    "KGZ",  # Kyrgyzstan
    "MDA",  # Moldova
    "RUS",  # Russia
    "TJK",  # Tajikistan
    "UZB"   # Uzbekistan
]
# Create separate boolean columns for each group
cluster['is_OECD'] = cluster['ISO'].isin(OECD_members)
cluster['is_BRICS'] = cluster['ISO'].isin(BRICS_members)
cluster['is_BRICS_plus'] = cluster['ISO'].isin(BRICS_plus_members)
cluster['is_G7'] = cluster['ISO'].isin(g7_iso3)
cluster['is_Coalition'] = cluster['ISO'].isin(coalition_iso_alpha3)
cluster['is_COP30'] = cluster['ISO'].isin(cop_30_mofs)
cluster['is_NGFS'] = cluster['ISO'].isin(ngfs_iso_alpha3)
cluster['is_BOGA'] = cluster['ISO'].isin(boga_iso_alpha3)
cluster['is_PPCA'] = cluster['ISO'].isin(ppca_iso_alpha3)
cluster['is_FF_NPT'] = cluster['ISO'].isin(ff_npt_iso_alpha3)
cluster['is_Port_Vila'] = cluster['ISO'].isin(port_vila_call_iso_alpha3)
cluster['is_CNC'] = cluster['ISO'].isin(cnc_iso_alpha3)
cluster['is_COFFS'] = cluster['ISO'].isin(coffis_iso_alpha3)
cluster['is_GCPA'] = cluster['ISO'].isin(gcpa_finance_mission_iso_alpha3)
cluster['is_SIDS'] = cluster['ISO'].isin(sids_energy_transition_iso_alpha3)
cluster['is_V20'] = cluster['ISO'].isin(v20_iso_alpha3)
cluster['is_EU'] = cluster['ISO'].isin(eu_iso_alpha3)
cluster['is_AU'] = cluster['ISO'].isin(african_union_iso_alpha3)
cluster['is_CELAC'] = cluster['ISO'].isin(celac_iso_alpha3)
cluster['is_ASEAN'] = cluster['ISO'].isin(asean_iso_alpha3)
cluster['is_SCO'] = cluster['ISO'].isin(sco_iso_alpha3)
cluster['is_CIS'] = cluster['ISO'].isin(cis_iso_alpha3)
cluster['is_Commonwealth'] = cluster['ISO'].isin(commonwealth_iso_alpha3)


group_filter_options = [
    {'label': 'All', 'value': 'All'},
    {'label': 'OECD', 'value': 'OECD'},
    {'label': 'BRICS', 'value': 'BRICS'},
    {'label': 'BRICS+', 'value': 'BRICS+'},
    {'label': 'G7', 'value': 'G7'},
    {'label': 'Coalition of Finance Ministers', 'value': 'Coalition'},
    {'label': 'COP30 Finance Ministers', 'value': 'COP30'},
    {'label': 'NGFS', 'value': 'NGFS'},
    {'label': 'Beyond Oil & Gas Alliance (BOGA)', 'value': 'BOGA'},
    {'label': 'Powering Past Coal Alliance (PPCA)', 'value': 'PPCA'},
    {'label': 'Fossil Fuel Non-Proliferation Treaty', 'value': 'FF_NPT'},
    {'label': 'Port Vila Call to Action', 'value': 'Port_Vila'},
    {'label': 'Carbon Neutrality Coalition', 'value': 'CNC'},
    {'label': 'Coalition Against Fossil Fuel Subsidies', 'value': 'COFFS'},
    {'label': 'Global Clean Power Alliance', 'value': 'GCPA'},
    {'label': 'SIDS Renewable Energy Transition', 'value': 'SIDS'},
    {'label': 'Vulnerable 20 (V20)', 'value': 'V20'},
    {'label': 'European Union', 'value': 'EU'},
    {'label': 'African Union', 'value': 'AU'},
    {'label': 'CELAC', 'value': 'CELAC'},
    {'label': 'ASEAN', 'value': 'ASEAN'},
    {'label': 'Commonwealth of Nations', 'value': 'Commonwealth'},
    {'label': 'Shanghai Cooperation Organisation (SCO)', 'value': 'SCO'},
    {'label': 'Commonwealth of Independent States (CIS)', 'value': 'CIS'}
]



# -------------------------
# Function to compute avg distance
# -------------------------
def average_distance(df, iso_list):
    subset = df[df['iso3c'].isin(iso_list)]
    values = subset['idealpointfp'].values
    pairs = list(itertools.combinations(values, 2))
    distances = [abs(a - b) for a, b in pairs]
    return sum(distances) / len(distances) if distances else None
# -------------------------
# Define a function to retrieve the list dynamically
# -------------------------

def get_iso_list(club_name):
    # Map group values to variable names
    group_map = {
        "OECD": "OECD_members",
        "BRICS": "BRICS_members",
        "BRICS+": "BRICS_plus_members",
        "G7": "g7_iso3",
        "Coalition": "coalition_iso_alpha3",
        "COP30": "cop_30_mofs",
        "NGFS": "ngfs_iso_alpha3",
        "BOGA": "boga_iso_alpha3",
        "PPCA": "ppca_iso_alpha3",
        "FF_NPT": "ff_npt_iso_alpha3",
        "Port_Vila": "port_vila_call_iso_alpha3",
        "CNC": "cnc_iso_alpha3",
        "COFFS": "coffis_iso_alpha3",
        "GCPA": "gcpa_finance_mission_iso_alpha3",
        "SIDS": "sids_energy_transition_iso_alpha3",
        "V20": "v20_iso_alpha3",
        "EU": "eu_iso_alpha3",
        "AU": "african_union_iso_alpha3",
        "CELAC": "celac_iso_alpha3",
        "ASEAN": "asean_iso_alpha3",
        "Commonwealth": "commonwealth_iso_alpha3",
        "SCO": "sco_iso_alpha3",
        "CIS": "cis_iso_alpha3"
    }
    return globals().get(group_map.get(club_name, ""), [])


# -----------------------------
# App Initialization
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# -----------------------------
# Layout
# -----------------------------


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

        # ------------------- CLUSTERING TAB -------------------
        dcc.Tab(label='Clustering Visualization', value='tab-cluster', children=[
            dbc.Container([
                # --- Feature selection ---
                dbc.Row([
                    dbc.Col([
                        html.Label("Macro Stability"),
                        dcc.Dropdown(
                            id='macro-dropdown',
                            options=[
                                {'label': variable_labels[v], 'value': v} 
                                for v in dimensions["Macro Stability"]
                            ],
                            value=["Sovereign risk"],
                            multi=True
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Nature"),
                        dcc.Dropdown(
                            id='nature-dropdown',
                            options=[
                                {'label': variable_labels[v], 'value': v} 
                                for v in dimensions["Nature"]
                            ],
                            value=["Biocapacity", "Renewable_value_scaled", "Mineral_value_scaled"],
                            multi=True
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Green Competitiveness"),
                        dcc.Dropdown(
                            id='green-dropdown',
                            options=[
                                {'label': variable_labels[v], 'value': v} 
                                for v in dimensions["Green Competitiveness"]
                            ],
                            value=["BLI_scaled", "GCP_scaled"],
                            multi=True
                        )
                    ], width=3),
                    dbc.Col([
                        html.Label("Climate Adaptation & Vulnerability"),
                        dcc.Dropdown(
                            id='climate-dropdown',
                            options=[
                                {'label': variable_labels[v], 'value': v} 
                                for v in dimensions["Climate Adaptation and vulnerability"]
                            ],
                            value=["IMF-Adapted Readiness score_scaled","Vulnerability score_scaled"],
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
                            id='group-filter-cluster',
                            options=group_filter_options,
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
                                {'label': 'Climate Finance needs', 'value': 'Needs'}
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
        ]),

        # ------------------- DATA EXPLORER TAB -------------------
        dcc.Tab(label='Data Explorer', value='tab-data', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Variable"),
                        dcc.Dropdown(
                            id='variable-dropdown',
                            options=[
                                {
                                    'label': variable_labels.get(col, col),
                                    'value': col
                                }
                                for col in variable_definitions.keys()
                            ],
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
        ]),

        # ------------------- COMPARE CLIMATE CLUB TAB -------------------
        dcc.Tab(label='Compare Climate Club', value='tab-ClimateClub', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Label("Group Filter"),
                        dcc.Dropdown(
                            id='group-filter-club',
                            options=group_filter_options,
                            value='All',
                            clearable=False
                        ),
                        html.Div(id="club-output", style={"marginTop": "20px"})
                    ])
                ])
            ])
        ]),

        # ------------------- CLIMATE CLUB CREATOR TAB -------------------
        dcc.Tab(label='Climate Club Creator', value='tab-ClimateClubCreator', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Label("Group Filter"),
                        dcc.Dropdown(
                            id='group-filter-creator',
                            options=group_filter_options,
                            value='All',
                            clearable=False
                        ),
                        html.Div(id="club-output-creator", style={"marginTop": "20px"})
                    ])
                ])
            ])
        ])
    ])
], fluid=True)

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
    Input('group-filter-cluster', 'value'),
    Input('bubble-variable', 'value')
)
def update_clusters(n_clusters, macro_vars, nature_vars, green_vars, climate_vars, viz_mode, group_filter, bubble_var):
    selected_features = (macro_vars or []) + (nature_vars or []) + (green_vars or []) + (climate_vars or [])
    if len(selected_features) < 2:
        return px.scatter(title="Select at least 2 features")

    df = cluster.copy()

    # PCA
    X = df[selected_features].fillna(0)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    df['cluster_name'] = df['cluster'].map(lambda c: cluster_names.get(c, f"Cluster {c}"))

    # --------------------------
    # ✅ Dynamic Group Selection
    # --------------------------
    if group_filter == 'All':
        df['is_selected'] = True
    else:
        col_name = f"is_{group_filter.replace('+', 'plus')}"  # handles BRICS+
        if col_name in df.columns:
            df['is_selected'] = df[col_name]
        else:
            df['is_selected'] = False

    if viz_mode == 'highlight':
        df['opacity'] = df['is_selected'].map({True: 1.0, False: 0.2})
        df['marker_size'] = df['is_selected'].map({True: 10, False: 3})
        #df['border_width'] = df['is_selected'].map({True: 3, False: 0.5})

        fig = go.Figure()
        for cluster_name, group in df.groupby('cluster_name'):
            fig.add_trace(go.Scatter(
                x=group['PC1'],
                y=group['PC2'],
                mode='markers+text',
                text=group['ISO'],
                name=cluster_name,
                marker=dict(
                    size=group['marker_size'],
                    color=color_map.get(cluster_name, 'gray'),
                    opacity=group['opacity'],
                    #line=dict(width=group['border_width'], color='black')
                ),
                textposition='top center'
            ))
    else:  # Bubble mode
        df['marker_size'] = df[bubble_var].fillna(0.1)
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

    #print(df[['ISO', 'is_selected'] + [col for col in df.columns if col.startswith('is_')]].head(40))

    return fig







# -----------------------------
# Data explorer callback
# -----------------------------
@app.callback(
    Output('variable-table', 'data'),
    Output('variable-map', 'figure'),
    Input('variable-dropdown', 'value')
)
def update_data_explorer(selected_variable):
    df = cluster.copy()

    # Check variable existence
    if selected_variable not in df.columns:
        fig = px.choropleth(title="Variable not found")
        return [], fig

    # Ensure numeric or fallback
    if not pd.api.types.is_numeric_dtype(df[selected_variable]):
        df[selected_variable] = pd.to_numeric(df[selected_variable], errors='coerce')

    df[selected_variable] = df[selected_variable].fillna(0)

    # Table
    table_data = df[['ISO', selected_variable]].rename(columns={selected_variable: "Value"})
    table_records = table_data.to_dict('records')

    # Choropleth
    fig = px.choropleth(
        df,
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
# Compare Climate Club callback
# -----------------------------
@app.callback(
    Output("club-output", "children"),
    Input("group-filter-club", "value")
)
def update_average_distance(club_name):
    iso_list = get_iso_list(club_name)
    avg_dist = average_distance(ipd_data, iso_list)
    
    if avg_dist is None:
        return f"No data available for {club_name}."
    else:
        return f"Average distance for {club_name}: {avg_dist:.3f}"



# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)



