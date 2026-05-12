"""
Simulateur du Taux NEET par Commune — Sénégal
Application Streamlit — Modèle de Régression Linéaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import copy
import re
import unicodedata
from urllib.request import urlopen
import plotly.graph_objects as go
import plotly.express as px

# ─── CONFIG PAGE ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulateur NEET — Communes",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #0f172a;
}

.stApp label,
.stApp .stCheckbox,
.stApp .stCheckbox *,
.stApp .stCheckbox label,
.stApp .stCheckbox div[role="checkbox"],
.stApp div[role="checkbox"] + div,
.stApp [data-testid="stCheckbox"],
.stApp .css-10trblm,
.stApp .css-1yj0el2 {
    color: #000000 !important;
}

/* Fond principal */
.stApp {
    background: #eff6ff;
    color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"] * {
    color: #111827 !important;
}
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem !important;
    color: #475569 !important;
}

/* Titres */
h1 { font-family: 'Fraunces', serif !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; }

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.09);
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    color: #0f172a;
}
.metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 6px;
}

/* Gauge badge */
.badge-low    { color: #059669; }
.badge-medium { color: #d97706; }
.badge-high   { color: #dc2626; }
.badge-critic { color: #7f1d1d; }

/* Section headers */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #1d4ed8;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #1d4ed8;
    padding-left: 12px;
    margin: 20px 0 14px 0;
}

/* Factor pills */
.factor-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px;
}
.pill-red   { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.pill-green { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 24px 0;
}

/* Group accordion */
.group-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 18px;
    margin-bottom: 4px;
    padding: 4px 0;
    border-bottom: 1px solid #e2e8f0;
    font-weight: 700;
}

/* Tab style */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #475569 !important;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* Slider overrides */
.stSlider > div { padding: 0 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL PARAMS ───────────────────────────────────────────────────────
@st.cache_data
def load_model():
    with open("model_params.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_excel("Modele_NEET_Communes.xlsx", sheet_name="Données")
    return df

def normalize_commune_name(value):
    """Normalise les noms pour faire correspondre Excel et GeoJSON."""
    text = str(value or "").upper().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def iter_geojson_points(geometry):
    if not geometry:
        return
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        for ring in coords:
            for lon, lat, *rest in ring:
                yield lon, lat
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for lon, lat, *rest in ring:
                    yield lon, lat

def feature_collection_bounds(features):
    points = [
        point
        for feature in features
        for point in iter_geojson_points(feature.get("geometry"))
    ]
    if not points:
        return None

    lons, lats = zip(*points)
    return {
        "min_lon": min(lons),
        "max_lon": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats),
        "center": {"lon": (min(lons) + max(lons)) / 2, "lat": (min(lats) + max(lats)) / 2},
    }

def zoom_from_bounds(bounds):
    if not bounds:
        return 5.6

    lon_span = max(bounds["max_lon"] - bounds["min_lon"], 0.01)
    lat_span = max(bounds["max_lat"] - bounds["min_lat"], 0.01)
    span = max(lon_span, lat_span)

    if span <= 0.04:
        return 12
    if span <= 0.08:
        return 11
    if span <= 0.16:
        return 10
    if span <= 0.35:
        return 9
    if span <= 0.7:
        return 8
    return 7

@st.cache_data(show_spinner=False)
def load_communes_geojson():
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SEN_4.json"
    try:
        with urlopen(url, timeout=20) as response:
            geojson = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    # On garde le GeoJSON original intact dans le cache et on ajoute une clé de jointure.
    geojson = copy.deepcopy(geojson)
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        commune_name = (
            props.get("NAME_4")
            or props.get("NAME_3")
            or props.get("NAME_2")
            or props.get("NAME_1")
            or ""
        )
        props["_app_commune_key"] = normalize_commune_name(commune_name)
        props["_app_commune_name"] = commune_name
    return geojson

params = load_model()
df_data = load_data()
df_data["_commune_key"] = df_data["Commune_Nom"].apply(normalize_commune_name)

FEATURES       = params["features"]
COEF           = np.array(params["coefficients"])
INTERCEPT      = params["intercept"]
SCALER_MEAN    = np.array(params["scaler_mean"])
SCALER_SCALE   = np.array(params["scaler_scale"])
FEAT_STATS     = {f: {"min": df_data[f].min(), "max": df_data[f].max(), "mean": df_data[f].mean()} for f in FEATURES}

TARGET = "E1_Taux_NEET"
neet_mean = df_data[TARGET].mean()

# ─── PREDICTION FUNCTION ─────────────────────────────────────────────────────
def predict_neet(values_dict):
    x = np.array([values_dict[f] for f in FEATURES])
    x_sc = (x - SCALER_MEAN) / SCALER_SCALE
    pred = float(np.dot(x_sc, COEF) + INTERCEPT)
    return max(0.0, min(1.0, pred))

def neet_label(v):
    if v < 0.25: return "Faible",   "#059669", "badge-low"
    if v < 0.45: return "Modéré",   "#d97706", "badge-medium"
    if v < 0.65: return "Élevé",    "#dc2626", "badge-high"
    return            "Critique",   "#7f1d1d", "badge-critic"

# ─── FEATURE GROUPS ──────────────────────────────────────────────────────────
GROUPS = {
    "Démographie": {
        "color": "#6366f1",
        "features": ["D1_Taux_Feminite_15_35", "D4_Mariage_Precoce",
                     "D5_Taux_Orphelins", "D6_Taux_Urbanisation"]
    },
    "Éducation": {
        "color": "#0891b2",
        "features": ["S1_Taux_Descolarisation", "S2_Taux_NonScolarisation",
                     "S3_Taux_Achevement_BFEM", "S4_Taux_Analphabetisme",
                     "S5_Taux_Formation_Pro", "S6_Part_Coranique_Exclusif",
                     "S7_Taux_Handicap"]
    },
    "Emploi et Économie": {
        "color": "#ea580c",
        "features": ["E2_Taux_Chomage", "E3_Taux_Informalite",
                     "E4_Part_Emploi_Agricole", "E5_Pauvrete_Alim_Conjoncturelle",
                     "E6_Pauvrete_Alim_Structurelle", "E7_Transferts_Diaspora",
                     "E8_Inactivite_Feminine"]
    },
    "Technologie": {
        "color": "#0284c7",
        "features": ["T1_Acces_Internet", "T2_Possession_Mobile", "T3_Possession_Ordinateur"]
    },
    "Habitat": {
        "color": "#7c3aed",
        "features": ["H1_Location_Precaire", "H2_Acces_Eau_Potable",
                     "H3_Acces_Toilettes_Adequates", "H4_Surpeuplement",
                     "H5_Indice_Equipement_Moyen"]
    },
    "Migration": {
        "color": "#b45309",
        "features": ["M1_Migration_Recente_1an", "M3_Emig_Pour_Travail", "M4_Emig_Pour_Etudes"]
    },
    "Ménage": {
        "color": "#15803d",
        "features": ["R1_Taille_Moyenne_Menages", "R3_Fecondite_Precoce"]
    },
}

LABELS = {
    "D1_Taux_Feminite_15_35":       "Féminité 15-35 ans",
    "D4_Mariage_Precoce":           "Mariage précoce",
    "D5_Taux_Orphelins":            "Taux d'orphelins",
    "D6_Taux_Urbanisation":         "Urbanisation",
    "S1_Taux_Descolarisation":      "Déscolarisation",
    "S2_Taux_NonScolarisation":     "Non-scolarisation",
    "S3_Taux_Achevement_BFEM":      "Achèvement BFEM",
    "S4_Taux_Analphabetisme":       "Analphabétisme",
    "S5_Taux_Formation_Pro":        "Formation pro",
    "S6_Part_Coranique_Exclusif":   "Coranique exclusif",
    "S7_Taux_Handicap":             "Handicap",
    "E2_Taux_Chomage":              "Chômage",
    "E3_Taux_Informalite":          "Informalité",
    "E4_Part_Emploi_Agricole":      "Emploi agricole",
    "E5_Pauvrete_Alim_Conjoncturelle": "Pauvreté alim. conjoncturelle",
    "E6_Pauvrete_Alim_Structurelle":   "Pauvreté alim. structurelle",
    "E7_Transferts_Diaspora":       "Transferts diaspora",
    "E8_Inactivite_Feminine":       "Inactivité féminine",
    "T1_Acces_Internet":            "Accès internet",
    "T2_Possession_Mobile":         "Possession mobile",
    "T3_Possession_Ordinateur":     "Possession ordinateur",
    "H1_Location_Precaire":         "Location précaire",
    "H2_Acces_Eau_Potable":         "Accès eau potable",
    "H3_Acces_Toilettes_Adequates": "Toilettes adéquates",
    "H4_Surpeuplement":             "Surpeuplement",
    "H5_Indice_Equipement_Moyen":   "Indice équipement moyen",
    "M1_Migration_Recente_1an":     "Migration récente (1 an)",
    "M3_Emig_Pour_Travail":         "Émigration pour travail",
    "M4_Emig_Pour_Etudes":          "Émigration pour études",
    "R1_Taille_Moyenne_Menages":    "Taille moy. ménages",
    "R3_Fecondite_Precoce":         "Fécondité précoce",
}

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="font-family:'Fraunces',serif; font-size:2.4rem; font-weight:700;
                color: #1a2035; line-height:1.1; margin-bottom:0.4rem;">
        Simulateur NEET par Commune
    </div>
    <div style="color:#475569; font-size:0.9rem; letter-spacing:0.5px; font-family:'DM Sans',sans-serif;">
        Sénégal &nbsp;·&nbsp; 552 communes &nbsp;·&nbsp; Régression Linéaire &nbsp;·&nbsp; R² = 0.9595
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ─── SIDEBAR — SLIDERS ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Fraunces',serif; font-size:1.3rem; font-weight:700;
                color:#1a2035; margin-bottom:4px;">
        Paramètres
    </div>
    <div style="color:#475569; font-size:0.75rem; margin-bottom:16px; font-family:'DM Sans',sans-serif;">
        Ajustez les indicateurs pour simuler le taux NEET
    </div>
    """, unsafe_allow_html=True)

    # Préchargement depuis une commune réelle
    communes_list = ["— Manuel —"] + sorted(df_data["Commune_Nom"].tolist())
    selected_commune = st.selectbox("Charger une commune", communes_list)

    if selected_commune != "— Manuel —":
        row = df_data[df_data["Commune_Nom"] == selected_commune].iloc[0]
        defaults = {f: float(row[f]) for f in FEATURES}
    else:
        defaults = {f: float(FEAT_STATS[f]["mean"]) for f in FEATURES}

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    values = {}
    for group_name, group_info in GROUPS.items():
        st.markdown(f"""
        <div class="group-title" style="color:{group_info['color']};">
            {group_name}
        </div>""", unsafe_allow_html=True)

        for feat in group_info["features"]:
            mn  = FEAT_STATS[feat]["min"]
            mx  = FEAT_STATS[feat]["max"]
            avg = defaults[feat]

            if feat == "R1_Taille_Moyenne_Menages":
                val = st.slider(
                    LABELS[feat], min_value=round(mn, 1), max_value=round(mx, 1),
                    value=round(avg, 1), step=0.1, key=feat,
                    format="%.1f"
                )
            else:
                val = st.slider(
                    LABELS[feat], min_value=0.0, max_value=1.0,
                    value=round(min(max(avg, 0.0), 1.0), 3),
                    step=0.01, key=feat, format="%.2f"
                )
            values[feat] = val

# ─── PREDICTION ──────────────────────────────────────────────────────────────
neet_pred = predict_neet(values)
label, color, badge_cls = neet_label(neet_pred)

# Calcul contribution de chaque variable
x_arr    = np.array([values[f] for f in FEATURES])
x_sc_arr = (x_arr - SCALER_MEAN) / SCALER_SCALE
contributions = x_sc_arr * COEF

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Résultat et Analyse",
    "Carte des communes",
    "Leviers d'action",
    "Comparaison",
])

# Couleurs pour les graphiques (thème clair)
BG_COLOR   = "#ffffff"
GRID_COLOR = "#e2e8f0"
TICK_COLOR = "#334155"
TEXT_COLOR = "#000000"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Résultat
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_gauge, col_metrics = st.columns([1, 2], gap="large")

    with col_gauge:
        # Gauge Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=neet_pred * 100,
            number={"suffix": "%", "font": {"size": 42, "color": color,
                                             "family": "Fraunces"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": TICK_COLOR, "tickfont": {"color": TEXT_COLOR}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "#d1fae5"},
                    {"range": [25, 45], "color": "#fef9c3"},
                    {"range": [45, 65], "color": "#fee2e2"},
                    {"range": [65, 100],"color": "#fca5a5"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": neet_pred * 100
                },
            },
            title={"text": f"<b>Taux NEET Prédit</b><br><span style='color:{color};font-size:1rem'>{label}</span>",
                   "font": {"color": "#475569", "size": 14}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            height=280,
            margin=dict(t=60, b=10, l=20, r=20),
            font=dict(color="#000000"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Context vs moyenne nationale
        delta     = neet_pred - neet_mean
        delta_pct = delta * 100
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Vs Moyenne nationale</div>
            <div class="metric-value" style="color:{'#dc2626' if delta>0 else '#059669'}; font-size:1.8rem;">
                {'▲' if delta>0 else '▼'} {abs(delta_pct):.1f} pts
            </div>
            <div class="metric-sub">Moyenne : {neet_mean*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="section-header">Top contributions au taux NEET</div>', unsafe_allow_html=True)

        contrib_df = pd.DataFrame({
            "feature": FEATURES,
            "label":   [LABELS[f] for f in FEATURES],
            "contrib": contributions,
        }).sort_values("contrib", key=abs, ascending=False).head(12)

        colors_bar = ["#ef4444" if v > 0 else "#10b981" for v in contrib_df["contrib"]]

        fig_bar = go.Figure(go.Bar(
            x=contrib_df["contrib"],
            y=contrib_df["label"],
            orientation="h",
            marker_color=colors_bar,
            marker_line_width=0,
            text=[f"{v:+.4f}" for v in contrib_df["contrib"]],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=10),
        ))
        fig_bar.update_layout(
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            height=380,
            margin=dict(t=10, b=10, l=10, r=80),
            font=dict(color="#000000"),
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=True,
                       zerolinecolor="#cbd5e1", color="#000000", tickfont=dict(color="#000000", size=9)),
            yaxis=dict(color="#000000", tickfont=dict(color="#000000", size=10)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Bottom row : indicateurs clés ────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Indicateurs clés de la simulation</div>', unsafe_allow_html=True)

    key_feats = [
        ("S2_Taux_NonScolarisation", "Non-scolarisation"),
        ("E2_Taux_Chomage",          "Chômage"),
        ("E8_Inactivite_Feminine",   "Inactivité féminine"),
        ("S3_Taux_Achevement_BFEM", "Achèvement BFEM"),
        ("D4_Mariage_Precoce",       "Mariage précoce"),
    ]
    cols = st.columns(5)
    for col, (feat, lbl) in zip(cols, key_feats):
        v = values[feat]
        nat_mean = FEAT_STATS[feat]["mean"]
        d = v - nat_mean
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="font-size:1.6rem; color:#2563eb;">
                {v*100:.1f}%
            </div>
            <div class="metric-sub" style="color:{'#dc2626' if d>0 else '#059669'};">
                {'▲' if d>0 else '▼'} {abs(d*100):.1f} pts vs moy.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Carte des communes
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Recherche et carte des communes</div>', unsafe_allow_html=True)

    default_commune = selected_commune if selected_commune != "— Manuel —" else "DAKAR-PLATEAU"
    if default_commune not in df_data["Commune_Nom"].values:
        default_commune = df_data["Commune_Nom"].iloc[0]

    commune_options = sorted(df_data["Commune_Nom"].tolist())
    total_communes = len(df_data)
    active_commune = st.session_state.get("active_map_commune", default_commune)
    if active_commune not in commune_options:
        active_commune = default_commune
    st.session_state["active_map_commune"] = active_commune
    if st.session_state.get("map_commune_search") != active_commune:
        st.session_state["map_commune_search"] = active_commune

    selected_map_commune = st.selectbox(
        "Rechercher une commune",
        commune_options,
        index=commune_options.index(active_commune),
        key="map_commune_search",
    )
    if selected_map_commune != st.session_state["active_map_commune"]:
        st.session_state["active_map_commune"] = selected_map_commune

    col_map, col_info = st.columns([2, 1], gap="large")

    with col_map:
        geojson = load_communes_geojson()
        map_df = df_data.copy()
        map_df["NEET_pct"] = map_df[TARGET] * 100

        if geojson:
            geo_keys = {
                feature.get("properties", {}).get("_app_commune_key")
                for feature in geojson.get("features", [])
            }
            matched_count = int(map_df["_commune_key"].isin(geo_keys).sum())
            selected_key = normalize_commune_name(selected_map_commune)
            selected_geo = [
                feature for feature in geojson.get("features", [])
                if feature.get("properties", {}).get("_app_commune_key") == selected_key
            ]
            selected_bounds = feature_collection_bounds(selected_geo)
            map_center = selected_bounds["center"] if selected_bounds else {"lat": 14.4974, "lon": -14.4524}
            map_zoom = zoom_from_bounds(selected_bounds)

            fig_map = px.choropleth_mapbox(
                map_df,
                geojson=geojson,
                locations="_commune_key",
                featureidkey="properties._app_commune_key",
                color="NEET_pct",
                hover_name="Commune_Nom",
                custom_data=["Commune_Nom"],
                hover_data={
                    "_commune_key": False,
                    "NEET_pct": ":.1f",
                    "E2_Taux_Chomage": ":.1%",
                    "S2_Taux_NonScolarisation": ":.1%",
                    "E8_Inactivite_Feminine": ":.1%",
                },
                color_continuous_scale=["#d1fae5", "#fef9c3", "#fca5a5", "#991b1b"],
                range_color=(0, max(70, float(map_df["NEET_pct"].max()))),
                mapbox_style="carto-positron",
                center=map_center,
                zoom=map_zoom,
                opacity=0.72,
            )
            fig_map.update_traces(
                marker_line_width=0.35,
                marker_line_color="#ffffff",
                hovertemplate="<b>%{hovertext}</b><br>Taux NEET : %{z:.1f}%<extra></extra>",
            )
            fig_map.update_layout(
                height=560,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor=BG_COLOR,
                coloraxis_colorbar=dict(
                    title=dict(text="NEET (%)", font=dict(color="#000000")),
                    tickfont=dict(color="#000000"),
                ),
            )

            if selected_geo:
                fig_map.add_trace(go.Choroplethmapbox(
                    geojson={"type": "FeatureCollection", "features": selected_geo},
                    locations=[selected_key],
                    z=[1],
                    featureidkey="properties._app_commune_key",
                    colorscale=[[0, "#2563eb"], [1, "#2563eb"]],
                    showscale=False,
                    marker_opacity=0.18,
                    marker_line_width=3,
                    marker_line_color="#1d4ed8",
                    hoverinfo="skip",
                ))

            map_event = None
            try:
                map_event = st.plotly_chart(
                    fig_map,
                    use_container_width=True,
                    key="communes_neet_map",
                    on_select="rerun",
                    selection_mode="points",
                )
            except TypeError:
                st.plotly_chart(fig_map, use_container_width=True)

            if map_event:
                if isinstance(map_event, dict):
                    selected_points = map_event.get("selection", {}).get("points", [])
                else:
                    selected_points = getattr(getattr(map_event, "selection", None), "points", [])
                if selected_points:
                    point_data = selected_points[0]
                    customdata = point_data.get("customdata", [None])
                    clicked_commune = customdata[0] if isinstance(customdata, list) else customdata
                    clicked_commune = clicked_commune or point_data.get("location")
                    if clicked_commune in commune_options:
                        st.session_state["active_map_commune"] = clicked_commune
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()

            st.caption(f"Contours GADM niveau 4 chargés automatiquement. Correspondances trouvées : {matched_count}/{total_communes} communes.")
        else:
            st.warning(
                "La carte géographique n'a pas pu être chargée. Vérifiez la connexion internet, "
                "puis relancez l'application pour récupérer les contours des communes."
            )

    with col_info:
        selected_map_commune = st.session_state["active_map_commune"]
        selected_row = df_data[df_data["Commune_Nom"] == selected_map_commune].iloc[0]
        selected_actual_neet = float(selected_row[TARGET])
        selected_pred_values = {f: float(selected_row[f]) for f in FEATURES}
        selected_pred_neet = predict_neet(selected_pred_values)
        selected_label, selected_color, _ = neet_label(selected_actual_neet)
        rank = int(df_data[TARGET].rank(method="min", ascending=False).loc[selected_row.name])

        st.markdown(f"""
        <div class="metric-card" style="border-color:{selected_color};">
            <div class="metric-label">Commune sélectionnée</div>
            <div class="metric-value" style="font-size:1.55rem; color:#0f172a;">
                {selected_map_commune}
            </div>
            <div class="metric-sub" style="color:{selected_color};">
                {selected_label} · Rang {rank}/{total_communes}
            </div>
        </div>
        """, unsafe_allow_html=True)

        indicator_cards = [
            ("Taux NEET observé", selected_actual_neet, selected_actual_neet - neet_mean),
            ("NEET prédit modèle", selected_pred_neet, selected_pred_neet - neet_mean),
            ("Chômage", float(selected_row["E2_Taux_Chomage"]), float(selected_row["E2_Taux_Chomage"]) - FEAT_STATS["E2_Taux_Chomage"]["mean"]),
            ("Non-scolarisation", float(selected_row["S2_Taux_NonScolarisation"]), float(selected_row["S2_Taux_NonScolarisation"]) - FEAT_STATS["S2_Taux_NonScolarisation"]["mean"]),
            ("Inactivité féminine", float(selected_row["E8_Inactivite_Feminine"]), float(selected_row["E8_Inactivite_Feminine"]) - FEAT_STATS["E8_Inactivite_Feminine"]["mean"]),
            ("Accès internet", float(selected_row["T1_Acces_Internet"]), float(selected_row["T1_Acces_Internet"]) - FEAT_STATS["T1_Acces_Internet"]["mean"]),
        ]

        for title, value, delta_value in indicator_cards:
            st.markdown(f"""
            <div class="metric-card" style="margin-top:12px; padding:18px 20px;">
                <div class="metric-label">{title}</div>
                <div class="metric-value" style="font-size:1.6rem; color:#2563eb;">
                    {value*100:.1f}%
                </div>
                <div class="metric-sub" style="color:{'#dc2626' if delta_value>0 else '#059669'};">
                    {'▲' if delta_value>0 else '▼'} {abs(delta_value)*100:.1f} pts vs moy.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">Profil rapide</div>', unsafe_allow_html=True)
        profile_feats = [
            "S2_Taux_NonScolarisation",
            "E2_Taux_Chomage",
            "S3_Taux_Achevement_BFEM",
            "D4_Mariage_Precoce",
            "T1_Acces_Internet",
        ]
        profile_df = pd.DataFrame({
            "Indicateur": [LABELS[f] for f in profile_feats],
            "Commune": [float(selected_row[f]) * 100 for f in profile_feats],
            "Moyenne": [FEAT_STATS[f]["mean"] * 100 for f in profile_feats],
        })
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Bar(
            x=profile_df["Commune"],
            y=profile_df["Indicateur"],
            orientation="h",
            marker_color="#2563eb",
            name=selected_map_commune,
        ))
        fig_profile.add_trace(go.Bar(
            x=profile_df["Moyenne"],
            y=profile_df["Indicateur"],
            orientation="h",
            marker_color="#cbd5e1",
            name="Moyenne",
        ))
        fig_profile.update_layout(
            barmode="group",
            height=260,
            margin=dict(t=10, b=20, l=10, r=10),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            font=dict(color="#000000"),
            xaxis=dict(title="%", gridcolor=GRID_COLOR, color="#000000"),
            yaxis=dict(color="#000000", tickfont=dict(size=9)),
            legend=dict(orientation="h", y=-0.25, font=dict(color="#000000")),
        )
        st.plotly_chart(fig_profile, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Leviers d'action
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Simulation de scénarios d\'intervention</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#475569; font-size:0.85rem; margin-bottom:20px; font-family:'DM Sans',sans-serif;">
        Définissez un objectif de réduction du NEET et découvrez l'impact des leviers d'action.
    </div>
    """, unsafe_allow_html=True)

    col_scen, col_res = st.columns([1, 1], gap="large")

    with col_scen:
        target_neet = st.slider(
            "Objectif NEET cible (%)",
            min_value=5, max_value=int(neet_pred * 100),
            value=max(5, int(neet_pred * 100) - 10),
            step=1
        ) / 100

        reduction = neet_pred - target_neet
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px;">
            <div class="metric-label">Réduction à atteindre</div>
            <div class="metric-value" style="color:#d97706; font-size:2rem;">
                -{reduction*100:.1f} pts
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Leviers sélectionnables
        st.markdown("**Choisissez les leviers à activer :**")
        leviers = {
            "S2_Taux_NonScolarisation": ("Réduire non-scolarisation",   -1),
            "E2_Taux_Chomage":          ("Réduire chômage",              -1),
            "S5_Taux_Formation_Pro":    ("Augmenter formation pro",       +1),
            "S3_Taux_Achevement_BFEM": ("Augmenter achèvement BFEM",    +1),
            "D4_Mariage_Precoce":       ("Réduire mariage précoce",      -1),
            "E8_Inactivite_Feminine":   ("Réduire inactivité féminine",  -1),
            "T2_Possession_Mobile":     ("Augmenter possession mobile",   +1),
        }
        selected_leviers = {}
        for feat, (label_l, direction) in leviers.items():
            if st.checkbox(label_l, key=f"levier_{feat}"):
                selected_leviers[feat] = direction

    with col_res:
        if selected_leviers:
            scenario_values = dict(values)
            step_size = 0.05
            max_steps = 20
            history   = [neet_pred]
            step_neet  = neet_pred

            for step in range(max_steps):
                for feat, direction in selected_leviers.items():
                    current = scenario_values[feat]
                    if feat == "R1_Taille_Moyenne_Menages":
                        scenario_values[feat] = max(1.0, min(20.0, current + direction * 0.5))
                    else:
                        scenario_values[feat] = max(0.0, min(1.0, current + direction * step_size))
                new_pred = predict_neet(scenario_values)
                history.append(new_pred)
                if new_pred <= target_neet:
                    break

            final_neet = history[-1]
            label_f, color_f, _ = neet_label(final_neet)

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                y=[v * 100 for v in history],
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=6, color="#2563eb"),
                name="Trajectoire NEET",
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.06)"
            ))
            fig_traj.add_hline(
                y=target_neet * 100,
                line_dash="dash",
                line_color="#059669",
                annotation_text=f"Objectif {target_neet*100:.1f}%",
                annotation_font_color="#059669"
            )
            fig_traj.update_layout(
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=BG_COLOR,
                height=260,
                margin=dict(t=20, b=30, l=50, r=20),
                font=dict(color="#000000"),
                xaxis=dict(title="Étapes d'intervention", color="#000000", gridcolor=GRID_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(title="Taux NEET (%)", color="#000000", gridcolor=GRID_COLOR, tickfont=dict(color="#000000")),
                showlegend=False,
            )
            st.plotly_chart(fig_traj, use_container_width=True)

            achieved = final_neet <= target_neet
            border_color = "#bbf7d0" if achieved else "#fecaca"
            status_label = "Objectif atteint" if achieved else "Objectif non atteint"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{border_color};">
                <div class="metric-label">{status_label}</div>
                <div class="metric-value" style="color:{color_f}; font-size:1.8rem;">
                    {final_neet*100:.1f}% — {label_f}
                </div>
                <div class="metric-sub">
                    Gain : -{(neet_pred - final_neet)*100:.1f} pts en {len(history)-1} étapes
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:16px;">Changements appliqués</div>', unsafe_allow_html=True)
            for feat, direction in selected_leviers.items():
                old_val = values[feat]
                new_val = scenario_values[feat]
                delta_v = new_val - old_val
                pill_cls = "pill-red" if direction < 0 else "pill-green"
                arrow    = "▼" if delta_v < 0 else "▲"
                st.markdown(f"""
                <span class="factor-pill {pill_cls}">
                    {arrow} {LABELS[feat]} : {old_val*100:.1f}% — {new_val*100:.1f}%
                </span>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#cbd5e1; padding:60px 20px; font-size:1rem; font-family:'DM Sans',sans-serif;">
                Sélectionnez au moins un levier d'action
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Comparaison avec d'autres communes
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Comparaison avec d\'autres communes</div>', unsafe_allow_html=True)

    communes_compare = st.multiselect(
        "Sélectionnez des communes à comparer",
        sorted(df_data["Commune_Nom"].tolist()),
        default=["DAKAR-PLATEAU", "ZIGUINCHOR", "SAINT-LOUIS"] if all(
            c in df_data["Commune_Nom"].values
            for c in ["DAKAR-PLATEAU", "ZIGUINCHOR", "SAINT-LOUIS"]
        ) else []
    )

    sim_row = {"Commune": "Ma Simulation", "NEET": neet_pred}
    rows = [sim_row]
    for c in communes_compare:
        row = df_data[df_data["Commune_Nom"] == c].iloc[0]
        rows.append({"Commune": c, "NEET": float(row[TARGET])})

    df_comp = pd.DataFrame(rows).sort_values("NEET", ascending=True)
    colors_comp = ["#2563eb" if r["Commune"] == "Ma Simulation" else "#94a3b8"
                   for _, r in df_comp.iterrows()]

    fig_comp = go.Figure(go.Bar(
        x=df_comp["NEET"] * 100,
        y=df_comp["Commune"],
        orientation="h",
        marker_color=colors_comp,
        text=[f"{v*100:.1f}%" for v in df_comp["NEET"]],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11),
    ))
    fig_comp.add_vline(
        x=neet_mean * 100,
        line_dash="dot", line_color="#d97706",
        annotation_text=f"Moy. nationale {neet_mean*100:.1f}%",
        annotation_font_color="#d97706",
        annotation_position="top right"
    )
    fig_comp.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=max(280, len(rows) * 55),
        margin=dict(t=20, b=20, l=20, r=80),
        font=dict(color="#000000"),
        xaxis=dict(title="Taux NEET (%)", color="#000000", gridcolor=GRID_COLOR,
                   tickfont=dict(color="#000000"), range=[0, max(df_comp["NEET"].max() * 110, 30)]),
        yaxis=dict(color="#000000", tickfont=dict(color="#000000", size=12)),
        legend=dict(bgcolor="#f8fafc", font=dict(color="#000000"), bordercolor="#e2e8f0", borderwidth=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    if communes_compare:
        st.markdown('<div class="section-header">Profil multi-indicateurs</div>', unsafe_allow_html=True)
        radar_feats = [
            "S2_Taux_NonScolarisation", "E2_Taux_Chomage",
            "S3_Taux_Achevement_BFEM", "D4_Mariage_Precoce",
            "T2_Possession_Mobile", "H2_Acces_Eau_Potable",
            "E8_Inactivite_Feminine"
        ]
        radar_labels = [LABELS[f] for f in radar_feats]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[values[f] * 100 for f in radar_feats] + [values[radar_feats[0]] * 100],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Ma Simulation",
            line=dict(color="#2563eb", width=2),
            fillcolor="rgba(37,99,235,0.1)",
        ))
        palette = ["#d97706", "#059669", "#7c3aed", "#dc2626"]
        for idx, c in enumerate(communes_compare[:4]):
            row = df_data[df_data["Commune_Nom"] == c].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[float(row[f]) * 100 for f in radar_feats] + [float(row[radar_feats[0]]) * 100],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=c,
                line=dict(color=palette[idx % 4], width=2),
                fillcolor=f"rgba(0,0,0,0.03)",
            ))
        fig_radar.update_layout(
            font=dict(color="#000000"),
            polar=dict(
                bgcolor="#f8fafc",
                radialaxis=dict(visible=True, range=[0, 100], color=TICK_COLOR,
                                gridcolor="#e2e8f0", tickfont=dict(color="#000000")),
                angularaxis=dict(color="#000000", gridcolor="#e2e8f0"),
            ),
            paper_bgcolor=BG_COLOR,
            legend=dict(bgcolor="#f8fafc", font=dict(color="#000000"), bordercolor="#e2e8f0", borderwidth=1),
            height=420,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.75rem; padding:8px 0 16px 0;
            font-family:'DM Sans',sans-serif;">
    Modèle de Régression Linéaire &nbsp;·&nbsp; R² = 0.9595 &nbsp;·&nbsp; RMSE = 0.0273 &nbsp;·&nbsp; 552 communes &nbsp;·&nbsp; Sénégal
</div>
""", unsafe_allow_html=True)
