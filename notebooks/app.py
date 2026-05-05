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
# ROLE CLASSIFICATION
# ==========================================
def classify_role(row):
    pos_raw = str(row.get("pos_", row.get("pos", ""))).upper()

    sh90 = row.get("Standard_Sh/90", 0)
    gls = row.get("Performance_Gls", 0)
    ast = row.get("Performance_Ast", 0)
    crs = row.get("Performance_Crs", 0)
    tkl = row.get("Performance_TklW", 0)
    inter = row.get("Performance_Int", 0)

    # =========================================
    # NORMALIZE POSITION (IMPORTANT FIX)
    # =========================================
    if "GK" in pos_raw:
        pos = "GK"
    elif "ST" in pos_raw or "FW" in pos_raw:
        pos = "ST"
    elif "LW" in pos_raw:
        pos = "LW"
    elif "RW" in pos_raw:
        pos = "RW"
    elif "CAM" in pos_raw or "AM" in pos_raw:
        pos = "CAM"
    elif "CM" in pos_raw:
        pos = "CM"
    elif "CDM" in pos_raw:
        pos = "CDM"
    elif "LB" in pos_raw:
        pos = "LB"
    elif "RB" in pos_raw:
        pos = "RB"
    elif "CB" in pos_raw:
        pos = "CB"
    else:
        pos = "CM"

    # =========================================
    # GOALKEEPERS
    # =========================================
    if pos == "GK":
        if tkl > 3:
            return "Sweeper Keeper"
        elif ast > 1:
            return "Ball-Playing Keeper"
        return "Goalkeeper"

    # =========================================
    # STRIKERS
    # =========================================
    if pos == "ST":
        if gls > 10:
            return "Poacher"
        elif ast > 4:
            return "False 9"
        elif sh90 > 2:
            return "Advanced Forward"
        return "Target Forward"

    # =========================================
    # WIDE FORWARDS
    # =========================================
    if pos in ["LW", "RW"]:
        if crs > 35:
            return "Winger"
        elif ast > 5:
            return "Wide Playmaker"
        return "Inside Forward"

    # =========================================
    # CAM
    # =========================================
    if pos == "CAM":
        if gls > 8:
            return "Shadow Striker"
        elif ast > 6:
            return "Classic 10"
        elif sh90 > 1.8:
            return "Playmaker"
        return "Half Winger"

    # =========================================
    # CENTRAL MIDFIELD
    # =========================================
    if pos == "CM":
        if tkl > 18 and inter > 8:
            return "Box-To-Box"
        elif ast > 5:
            return "Deep-Lying Playmaker"
        elif sh90 > 1.5:
            return "Playmaker"
        return "Holding"

    # =========================================
    # CDM
    # =========================================
    if pos == "CDM":
        if tkl > 22:
            return "Holding"
        elif ast > 4:
            return "Deep-Lying Playmaker"
        elif gls > 2:
            return "Box Crasher"
        return "Centre Half"

    # =========================================
    # FULLBACKS
    # =========================================
    if pos in ["LB", "RB"]:
        if crs > 25:
            return "Wingback"
        elif ast > 3:
            return "Attacking Wingback"
        elif tkl > 15:
            return "Fullback"
        return "Inverted Wingback"

    # =========================================
    # CB
    # =========================================
    if pos == "CB":
        if inter > 12:
            return "Stopper"
        elif crs > 8:
            return "Ball-Playing Defender"
        return "Defender"

    return "Hybrid"


# ==========================================
# PAGE SETUP
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
# DATA ENGINE
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
    df["Cluster_ID"] = kmeans.fit_predict(X_scaled)

    df["Role"] = df.apply(classify_role, axis=1)

    return df, vectors, existing_metrics, X_scaled


df, tactical_vectors, metric_list, X_scaled = load_and_cluster_data()


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("Scouting Room")
    target_player = st.selectbox("Search Player Profile", sorted(df["player"].unique()))


player_idx = df[df["player"] == target_player].index[0]
player_row = df.iloc[player_idx]

pos_col = "pos_" if "pos_" in df.columns else "pos"


# ==========================================
# HEADER
# ==========================================
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    st.title(target_player)
    st.caption(f"{player_row.get('team', 'Unknown')} | {player_row.get('nation_', 'Unknown')}")

with c2:
    st.metric("Listed Position", player_row.get(pos_col, "N/A"))

with c3:
    st.metric("Tactical Role", player_row["Role"])

with c4:
    age_val = player_row.get("age_numeric", None)
    st.metric("Age", int(age_val) if pd.notnull(age_val) else "N/A")


st.divider()


# ==========================================
# ROLE BREAKDOWN
# ==========================================
st.subheader("🧠 Tactical Role Breakdown")

sh90 = player_row.get("Standard_Sh/90", 0)
gls = player_row.get("Performance_Gls", 0)
ast = player_row.get("Performance_Ast", 0)
crs = player_row.get("Performance_Crs", 0)
tkl = player_row.get("Performance_TklW", 0)
inter = player_row.get("Performance_Int", 0)

st.markdown(f"""
### {target_player} → **{player_row['Role']}**

- ⚽ Shots/90: **{sh90:.2f}**
- 🎯 Goals: **{gls}**
- 🎨 Assists: **{ast}**
- 📦 Crosses: **{crs}**
- 🛡️ Tackles Won: **{tkl}**
- 🔍 Interceptions: **{inter}**
""")


# ==========================================
# RADAR + HEATMAP
# ==========================================
col1, col2 = st.columns([1.2, 1])

with col1:

    def get_vec_score(vec_name):
        indices = [metric_list.index(m) for m in tactical_vectors[vec_name] if m in metric_list]
        if not indices:
            return 0
        return np.mean(X_scaled[player_idx][indices])

    scores = [
        get_vec_score("Progressive"),
        get_vec_score("Creative"),
        get_vec_score("Workrate")
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores,
        theta=list(tactical_vectors.keys()),
        fill='toself',
        line=dict(color='#00ff85', width=3)
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 3])),
        paper_bgcolor="#0e1117",
        font_color="white",
        height=450
    )

    st.plotly_chart(fig_radar, use_container_width=True)


with col2:
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

    sns.kdeplot(x=x, y=y, fill=True, cmap="Greens", alpha=0.7, ax=ax, levels=25)

    st.pyplot(fig)


# ==========================================
# SIMILARITY
# ==========================================
st.subheader("👯 Statistical Twins")

sim_matrix = cosine_similarity(X_scaled[player_idx].reshape(1, -1), X_scaled)

pos_data = df[pos_col] if pos_col in df.columns else ["N/A"] * len(df)

sim_df = pd.DataFrame({
    "player": df["player"],
    "pos": pos_data,
    "similarity": sim_matrix[0]
})

twins = sim_df[sim_df["player"] != target_player] \
    .sort_values("similarity", ascending=False) \
    .head(5)

cols = st.columns(5)

for i, (_, row) in enumerate(twins.iterrows()):
    with cols[i]:
        st.markdown(f"""
        <div style="background:#161b22;padding:15px;border-radius:10px;border:1px solid #30363d;">
            <p style="color:#808495;font-size:0.8rem;">MATCH {i+1}</p>
            <p style="color:white;font-weight:bold;">{row['player']}</p>
            <p style="color:#00ff85;">{row['pos']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Similarity", f"{row['similarity']:.1%}")