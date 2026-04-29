import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from mplsoccer import Pitch

# ==========================================
# 1. PAGE SETUP & THEME
# ==========================================
st.set_page_config(layout="wide", page_title="Role-DNA | Tactical Scouting")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00ff85; }
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #808495; }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data
def load_and_cluster_data():
    df = pd.read_csv("pl_master_clean.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["player"])
    
    vectors = {
        "Progressive": ["Standard_Sh", "Performance_Crs", "Performance_Fld"], 
        "Creative": ["Performance_Ast", "Standard_SoT", "Performance_Gls"],
        "Workrate": ["Performance_TklW", "Performance_Int", "Performance_Fls"]
    }
    
    all_req_metrics = [m for sub in vectors.values() for m in sub]
    existing_metrics = [m for m in all_req_metrics if m in df.columns]
    
    df[existing_metrics] = df[existing_metrics].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[existing_metrics])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    return df, vectors, existing_metrics, X_scaled

df, tactical_vectors, metric_list, X_scaled = load_and_cluster_data()

# ==========================================
# 3. SIDEBAR & SELECTION
# ==========================================
with st.sidebar:
    st.title("Scouting Room")
    target_player = st.selectbox("Search Player Profile", sorted(df["player"].unique()))
    # Removed the Saka Notice info box from here

player_idx = df[df["player"] == target_player].index[0]
player_row = df.iloc[player_idx]
pos_col = "pos_" if "pos_" in df.columns else "pos"

# ==========================================
# 4. TOP ROW: IDENTITY
# ==========================================
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    st.title(target_player)
    st.caption(f"{player_row.get('team', 'Unknown')} | {player_row.get('nation_', 'Unknown')}")
with c2:
    st.metric("Listed Position", player_row.get(pos_col, "N/A"))
with c3:
    # Cleaned: Showing just the Cluster ID number
    st.metric("Tactical Cluster", f"{player_row['Cluster_ID']}")
with c4:
    st.metric("Age", int(player_row.get('age_numeric', 0)))

st.divider()

# ==========================================
# 5. RADAR & HEATMAP
# ==========================================
col_main_1, col_main_2 = st.columns([1.2, 1])

with col_main_1:
    # Removed "Role-DNA Composition" subheader
    def get_vec_score(vec_name):
        indices = [metric_list.index(m) for m in tactical_vectors[vec_name] if m in metric_list]
        return np.mean(X_scaled[player_idx][indices]) if indices else 0

    scores = [get_vec_score("Progressive"), get_vec_score("Creative"), get_vec_score("Workrate")]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores, theta=list(tactical_vectors.keys()), fill='toself',
        fillcolor='rgba(0, 255, 133, 0.3)', line=dict(color='#00ff85', width=3)
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 3], gridcolor="#30363d"), bgcolor="#0e1117"),
        paper_bgcolor="#0e1117", font_color="white", height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_main_2:
    st.subheader("📍 Territorial Influence")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#161b22', line_color='#30363d')
    fig, ax = pitch.draw(figsize=(8, 6))
    
    pos_str = str(player_row.get(pos_col, "")).upper()
    if "FW" in pos_str:
        x = np.random.normal(95, 15, 100)
    elif "DF" in pos_str:
        x = np.random.normal(30, 15, 100)
    else:
        x = np.random.normal(60, 20, 100)
    
    y = np.random.normal(40, 20, 100)
    sns.kdeplot(x=x, y=y, fill=True, cmap='Greens', alpha=0.7, ax=ax, levels=25)
    st.pyplot(fig)

# ==========================================
# 6. SIMILARITY TWINS
# ==========================================
st.subheader("👯 Statistical Twins")
sim_matrix = cosine_similarity(X_scaled[player_idx].reshape(1, -1), X_scaled)
sim_df = pd.DataFrame({
    'player': df['player'],
    'pos': df[pos_col] if pos_col in df.columns else "N/A",
    'similarity': sim_matrix[0]
})

twins = sim_df[sim_df['player'] != target_player].sort_values(by='similarity', ascending=False).head(5)

twin_cols = st.columns(5)
for i, (idx, row) in enumerate(twins.iterrows()):
    with twin_cols[i]:
        st.markdown(f"""
        <div style="background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; min-height: 120px;">
            <p style="color: #808495; margin-bottom: 0px; font-size: 0.8rem;">MATCH {i+1}</p>
            <p style="color: white; font-weight: bold; margin-top: 0px;">{row['player']}</p>
            <p style="color: #00ff85; font-size: 0.9rem;">{row['pos']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Similarity", f"{row['similarity']:.1%}")