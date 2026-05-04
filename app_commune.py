"""
Simulateur du Taux NEET par Commune — Sénégal
Application Streamlit — Modèle de Régression Linéaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

# ─── CONFIG PAGE ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulateur NEET — Communes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Fond principal */
.stApp {
    background: #0b1120;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1f2a3d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem !important;
    color: #8899b0 !important;
}

/* Titres */
h1 { font-family: 'Syne', sans-serif !important; }
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #141e30 0%, #1a2744 100%);
    border: 1px solid #263558;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label {
    font-size: 0.78rem;
    color: #6b7fa3;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #5a6a85;
    margin-top: 6px;
}

/* Gauge badge */
.badge-low    { color: #22d3a5; }
.badge-medium { color: #f59e0b; }
.badge-high   { color: #ef4444; }
.badge-critic { color: #9b1c1c; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #3b82f6;
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
.pill-red   { background: #3b1219; color: #f87171; border: 1px solid #7f1d1d; }
.pill-green { background: #052e16; color: #4ade80; border: 1px solid #14532d; }

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d4270, transparent);
    margin: 24px 0;
}

/* Group accordion */
.group-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4b6090;
    margin-top: 18px;
    margin-bottom: 4px;
    padding: 4px 0;
    border-bottom: 1px solid #1f2a3d;
}

/* Tab style */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7fa3 !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #93c5fd !important;
}

/* Slider overrides */
.stSlider > div { padding: 0 !important; }
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

params = load_model()
df_data = load_data()

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
    if v < 0.25: return "Faible", "#22d3a5", "badge-low"
    if v < 0.45: return "Modéré", "#f59e0b", "badge-medium"
    if v < 0.65: return "Élevé",  "#ef4444", "badge-high"
    return "Critique", "#9b1c1c", "badge-critic"

# ─── FEATURE GROUPS ──────────────────────────────────────────────────────────
GROUPS = {
    "🧑‍👧 Démographie": {
        "color": "#818cf8",
        "features": ["D1_Taux_Feminite_15_35", "D4_Mariage_Precoce",
                     "D5_Taux_Orphelins", "D6_Taux_Urbanisation"]
    },
    "📚 Éducation": {
        "color": "#34d399",
        "features": ["S1_Taux_Descolarisation", "S2_Taux_NonScolarisation",
                     "S3_Taux_Achevement_BFEM", "S4_Taux_Analphabetisme",
                     "S5_Taux_Formation_Pro", "S6_Part_Coranique_Exclusif",
                     "S7_Taux_Handicap"]
    },
    "💼 Emploi & Économie": {
        "color": "#fb923c",
        "features": ["E2_Taux_Chomage", "E3_Taux_Informalite",
                     "E4_Part_Emploi_Agricole", "E5_Pauvrete_Alim_Conjoncturelle",
                     "E6_Pauvrete_Alim_Structurelle", "E7_Transferts_Diaspora",
                     "E8_Inactivite_Feminine"]
    },
    "📱 Technologie": {
        "color": "#38bdf8",
        "features": ["T1_Acces_Internet", "T2_Possession_Mobile", "T3_Possession_Ordinateur"]
    },
    "🏠 Habitat": {
        "color": "#e879f9",
        "features": ["H1_Location_Precaire", "H2_Acces_Eau_Potable",
                     "H3_Acces_Toilettes_Adequates", "H4_Surpeuplement",
                     "H5_Indice_Equipement_Moyen"]
    },
    "✈️ Migration": {
        "color": "#fbbf24",
        "features": ["M1_Migration_Recente_1an", "M3_Emig_Pour_Travail", "M4_Emig_Pour_Etudes"]
    },
    "👨‍👩‍👧 Ménage": {
        "color": "#a3e635",
        "features": ["R1_Taille_Moyenne_Menages", "R3_Fecondite_Precoce"]
    },
}

LABELS = {
    "D1_Taux_Feminite_15_35":       "Féminité 15-35 ans",
    "D4_Mariage_Precoce":           "Mariage précoce",
    "D5_Taux_Orphelins":            "Taux d'orphelins",
    "D6_Taux_Urbanisation":         "Urbanisation",
    "S1_Taux_Descolarisation":      "Déscolarisation",
    "S2_Taux_NonScolarisation":     "Non-scolarisation ⚡",
    "S3_Taux_Achevement_BFEM":      "Achèvement BFEM",
    "S4_Taux_Analphabetisme":       "Analphabétisme",
    "S5_Taux_Formation_Pro":        "Formation pro",
    "S6_Part_Coranique_Exclusif":   "Coranique exclusif",
    "S7_Taux_Handicap":             "Handicap",
    "E2_Taux_Chomage":              "Chômage ⚡",
    "E3_Taux_Informalite":          "Informalité",
    "E4_Part_Emploi_Agricole":      "Emploi agricole",
    "E5_Pauvrete_Alim_Conjoncturelle": "Pauvreté alim. conjoncturelle",
    "E6_Pauvrete_Alim_Structurelle":   "Pauvreté alim. structurelle",
    "E7_Transferts_Diaspora":       "Transferts diaspora",
    "E8_Inactivite_Feminine":       "Inactivité féminine ⚡",
    "T1_Acces_Internet":            "Accès internet",
    "T2_Possession_Mobile":         "Possession mobile",
    "T3_Possession_Ordinateur":     "Possession ordinateur",
    "H1_Location_Precaire":         "Location précaire",
    "H2_Acces_Eau_Potable":         "Accès eau potable",
    "H3_Acces_Toilettes_Adequates": "Toilettes adéquates",
    "H4_Surpeuplement":             "Surpeuplement",
    "H5_Indice_Equipement_Moyen":   "Indice équipement moyen",
    "M1_Migration_Recente_1an":     "Migration récente (1an)",
    "M3_Emig_Pour_Travail":         "Émigration pour travail",
    "M4_Emig_Pour_Etudes":          "Émigration pour études",
    "R1_Taille_Moyenne_Menages":    "Taille moy. ménages",
    "R3_Fecondite_Precoce":         "Fécondité précoce",
}

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
                background: linear-gradient(135deg, #60a5fa, #a78bfa);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                line-height:1.1; margin-bottom:0.4rem;">
        Simulateur NEET par Commune
    </div>
    <div style="color:#4b6090; font-size:0.9rem; letter-spacing:1px;">
        Sénégal · 552 communes · Régression Linéaire · R² = 0.9595
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ─── SIDEBAR — SLIDERS ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
                color:#60a5fa; margin-bottom:4px;">
        ⚙️ Paramètres
    </div>
    <div style="color:#4b6090; font-size:0.75rem; margin-bottom:16px;">
        Ajustez les indicateurs pour simuler le taux NEET
    </div>
    """, unsafe_allow_html=True)

    # Préchargement depuis une commune réelle
    communes_list = ["— Manuel —"] + sorted(df_data["Commune_Nom"].tolist())
    selected_commune = st.selectbox("📍 Charger une commune", communes_list)

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

            # Special: R1_Taille_Moyenne_Menages is not 0-1
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
contributions = x_sc_arr * COEF  # contribution de chaque feature au score

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Résultat & Analyse", "🎯 Leviers d'action", "🗺️ Comparaison"])

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
                                             "family": "Syne"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#2d3f5e", "tickfont": {"color": "#4b6090"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#0f1929",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],  "color": "#052e16"},
                    {"range": [25, 45], "color": "#1c2702"},
                    {"range": [45, 65], "color": "#2d1a00"},
                    {"range": [65, 100],"color": "#2d0a0a"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": neet_pred * 100
                },
            },
            title={"text": f"<b>Taux NEET Prédit</b><br><span style='color:{color};font-size:1rem'>{label}</span>",
                   "font": {"color": "#8899b0", "size": 14}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0f1929",
            plot_bgcolor="#0f1929",
            height=280,
            margin=dict(t=60, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Context vs moyenne nationale
        delta     = neet_pred - neet_mean
        delta_pct = delta * 100
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">Vs Moyenne nationale</div>
            <div class="metric-value" style="color:{'#ef4444' if delta>0 else '#22d3a5'}; font-size:1.8rem;">
                {'▲' if delta>0 else '▼'} {abs(delta_pct):.1f} pts
            </div>
            <div class="metric-sub">Moyenne : {neet_mean*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="section-header">Top contributions au taux NEET</div>', unsafe_allow_html=True)

        # Waterfall chart des contributions
        contrib_df = pd.DataFrame({
            "feature": FEATURES,
            "label":   [LABELS[f] for f in FEATURES],
            "contrib": contributions,
        }).sort_values("contrib", key=abs, ascending=False).head(12)

        colors_bar = ["#ef4444" if v > 0 else "#22d3a5" for v in contrib_df["contrib"]]

        fig_bar = go.Figure(go.Bar(
            x=contrib_df["contrib"],
            y=contrib_df["label"],
            orientation="h",
            marker_color=colors_bar,
            marker_line_width=0,
            text=[f"{v:+.4f}" for v in contrib_df["contrib"]],
            textposition="outside",
            textfont=dict(color="#6b7fa3", size=10),
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            height=380,
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis=dict(showgrid=True, gridcolor="#1a2540", zeroline=True,
                       zerolinecolor="#2d4270", color="#4b6090", tickfont=dict(size=9)),
            yaxis=dict(color="#8899b0", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Bottom row : indicateurs clés ────────────────────────────────────────
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Indicateurs clés de la simulation</div>', unsafe_allow_html=True)

    key_feats = [
        ("S2_Taux_NonScolarisation", "Non-scolarisation", "📚"),
        ("E2_Taux_Chomage",          "Chômage",           "💼"),
        ("E8_Inactivite_Feminine",   "Inactivité féminine", "👩"),
        ("S3_Taux_Achevement_BFEM", "Achèvement BFEM",   "🎓"),
        ("D4_Mariage_Precoce",       "Mariage précoce",   "💍"),
    ]
    cols = st.columns(5)
    for col, (feat, lbl, icon) in zip(cols, key_feats):
        v = values[feat]
        nat_mean = FEAT_STATS[feat]["mean"]
        d = v - nat_mean
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {lbl}</div>
            <div class="metric-value" style="font-size:1.6rem; color:#60a5fa;">
                {v*100:.1f}%
            </div>
            <div class="metric-sub" style="color:{'#ef4444' if d>0 else '#22d3a5'};">
                {'▲' if d>0 else '▼'} {abs(d*100):.1f}pts vs moy.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Leviers d'action
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Simulation de scénarios d\'intervention</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#4b6090; font-size:0.85rem; margin-bottom:20px;">
        Définissez un objectif de réduction du NEET et découvrez l'impact des leviers d'action.
    </div>
    """, unsafe_allow_html=True)

    col_scen, col_res = st.columns([1, 1], gap="large")

    with col_scen:
        target_neet = st.slider(
            "🎯 Objectif NEET cible (%)",
            min_value=5, max_value=int(neet_pred * 100),
            value=max(5, int(neet_pred * 100) - 10),
            step=1
        ) / 100

        reduction = neet_pred - target_neet
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px;">
            <div class="metric-label">Réduction à atteindre</div>
            <div class="metric-value" style="color:#f59e0b; font-size:2rem;">
                -{reduction*100:.1f} pts
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Leviers sélectionnables
        st.markdown("**Choisissez les leviers à activer :**")
        leviers = {
            "S2_Taux_NonScolarisation": ("📚 Réduire non-scolarisation", -1),
            "E2_Taux_Chomage":          ("💼 Réduire chômage",           -1),
            "S5_Taux_Formation_Pro":    ("🎓 Augmenter formation pro",    +1),
            "S3_Taux_Achevement_BFEM": ("📖 Augmenter achèvement BFEM", +1),
            "D4_Mariage_Precoce":       ("💍 Réduire mariage précoce",   -1),
            "E8_Inactivite_Feminine":   ("👩 Réduire inactivité féminine", -1),
            "T2_Possession_Mobile":     ("📱 Augmenter possession mobile", +1),
        }
        selected_leviers = {}
        for feat, (label_l, direction) in leviers.items():
            if st.checkbox(label_l, key=f"levier_{feat}"):
                selected_leviers[feat] = direction

    with col_res:
        if selected_leviers:
            # Simulation : modifier les valeurs par pas de 5%
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

            # Line chart de la trajectoire
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                y=[v * 100 for v in history],
                mode="lines+markers",
                line=dict(color="#60a5fa", width=3),
                marker=dict(size=6, color="#60a5fa"),
                name="Trajectoire NEET",
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.08)"
            ))
            fig_traj.add_hline(
                y=target_neet * 100,
                line_dash="dash",
                line_color="#22d3a5",
                annotation_text=f"Objectif {target_neet*100:.1f}%",
                annotation_font_color="#22d3a5"
            )
            fig_traj.update_layout(
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                height=260,
                margin=dict(t=20, b=30, l=50, r=20),
                xaxis=dict(title="Étapes d'intervention", color="#4b6090", gridcolor="#1a2540"),
                yaxis=dict(title="Taux NEET (%)", color="#4b6090", gridcolor="#1a2540"),
                showlegend=False,
            )
            st.plotly_chart(fig_traj, use_container_width=True)

            # Résultat
            achieved = final_neet <= target_neet
            st.markdown(f"""
            <div class="metric-card" style="border-color:{'#166534' if achieved else '#7f1d1d'};">
                <div class="metric-label">{'✅ Objectif atteint' if achieved else '⚠️ Objectif non atteint'}</div>
                <div class="metric-value" style="color:{color_f}; font-size:1.8rem;">
                    {final_neet*100:.1f}% → {label_f}
                </div>
                <div class="metric-sub">
                    Gain : -{(neet_pred - final_neet)*100:.1f} pts en {len(history)-1} étapes
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Détail des changements
            st.markdown('<div class="section-header" style="margin-top:16px;">Changements appliqués</div>', unsafe_allow_html=True)
            for feat, direction in selected_leviers.items():
                old_val = values[feat]
                new_val = scenario_values[feat]
                delta_v = new_val - old_val
                pill_cls = "pill-red" if direction < 0 else "pill-green"
                arrow    = "▼" if delta_v < 0 else "▲"
                st.markdown(f"""
                <span class="factor-pill {pill_cls}">
                    {arrow} {LABELS[feat].replace(' ⚡','')} : {old_val*100:.1f}% → {new_val*100:.1f}%
                </span>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; color:#2d4270; padding:60px 20px; font-size:1rem;">
                ← Sélectionnez au moins un levier d'action
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Comparaison avec d'autres communes
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Comparaison avec d\'autres communes</div>', unsafe_allow_html=True)

    communes_compare = st.multiselect(
        "Sélectionnez des communes à comparer",
        sorted(df_data["Commune_Nom"].tolist()),
        default=["DAKAR-PLATEAU", "ZIGUINCHOR", "SAINT-LOUIS"] if all(
            c in df_data["Commune_Nom"].values
            for c in ["DAKAR-PLATEAU", "ZIGUINCHOR", "SAINT-LOUIS"]
        ) else []
    )

    # Simulation actuelle
    sim_row = {"Commune": "🔵 Ma Simulation", "NEET": neet_pred}
    rows = [sim_row]
    for c in communes_compare:
        row = df_data[df_data["Commune_Nom"] == c].iloc[0]
        rows.append({"Commune": c, "NEET": float(row[TARGET])})

    df_comp = pd.DataFrame(rows).sort_values("NEET", ascending=True)
    colors_comp = ["#3b82f6" if r["Commune"].startswith("🔵") else "#6b7fa3"
                   for _, r in df_comp.iterrows()]

    fig_comp = go.Figure(go.Bar(
        x=df_comp["NEET"] * 100,
        y=df_comp["Commune"],
        orientation="h",
        marker_color=colors_comp,
        text=[f"{v*100:.1f}%" for v in df_comp["NEET"]],
        textposition="outside",
        textfont=dict(color="#8899b0", size=11),
    ))
    fig_comp.add_vline(
        x=neet_mean * 100,
        line_dash="dot", line_color="#f59e0b",
        annotation_text=f"Moy. nationale {neet_mean*100:.1f}%",
        annotation_font_color="#f59e0b",
        annotation_position="top right"
    )
    fig_comp.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        height=max(280, len(rows) * 55),
        margin=dict(t=20, b=20, l=20, r=80),
        xaxis=dict(title="Taux NEET (%)", color="#4b6090", gridcolor="#1a2540",
                   range=[0, max(df_comp["NEET"].max() * 110, 30)]),
        yaxis=dict(color="#c9d1e0", tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    if communes_compare:
        # Radar chart multi-indicateurs
        st.markdown('<div class="section-header">Profil multi-indicateurs</div>', unsafe_allow_html=True)
        radar_feats = [
            "S2_Taux_NonScolarisation", "E2_Taux_Chomage",
            "S3_Taux_Achevement_BFEM", "D4_Mariage_Precoce",
            "T2_Possession_Mobile", "H2_Acces_Eau_Potable",
            "E8_Inactivite_Feminine"
        ]
        radar_labels = [LABELS[f].replace(" ⚡", "") for f in radar_feats]

        fig_radar = go.Figure()
        # Simulation actuelle
        fig_radar.add_trace(go.Scatterpolar(
            r=[values[f] * 100 for f in radar_feats] + [values[radar_feats[0]] * 100],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Ma Simulation",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.15)",
        ))
        palette = ["#f59e0b", "#22d3a5", "#e879f9", "#fb923c"]
        for idx, c in enumerate(communes_compare[:4]):
            row = df_data[df_data["Commune_Nom"] == c].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[float(row[f]) * 100 for f in radar_feats] + [float(row[radar_feats[0]]) * 100],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=c,
                line=dict(color=palette[idx % 4], width=2),
                fillcolor=f"rgba(0,0,0,0.05)",
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0f1929",
                radialaxis=dict(visible=True, range=[0, 100], color="#2d4270",
                                gridcolor="#1a2540", tickfont=dict(color="#4b6090")),
                angularaxis=dict(color="#4b6090", gridcolor="#1a2540"),
            ),
            paper_bgcolor="#0b1120",
            legend=dict(bgcolor="#111827", font=dict(color="#8899b0")),
            height=420,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2d4270; font-size:0.75rem; padding:8px 0 16px 0;">
    Modèle de Régression Linéaire · R² = 0.9595 · RMSE = 0.0273 · 552 communes · Sénégal
</div>
""", unsafe_allow_html=True)