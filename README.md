# ⚽ Advanced Soccer Tactical Profiling & Role Analysis

## 📌 Project Overview
**Analytical Question:** *How can we use granular event-based metrics to mathematically define a player's tactical role regardless of their listed position?*

Traditional football scouting often relies on rigid positions (e.g., "Midfielder"). This project moves beyond those labels by using high-dimensional event data to identify a player's **"Role-DNA."** By analyzing where players touch the ball and what they do with it, we can distinguish a "Deep-Lying Playmaker" from a "Box-to-Box Engine," even if both are listed as Center Mids.

---

## 📊 Dataset & Sources
- **Source:** [FBref (via StatsBomb data)](https://fbref.com/en/comps/9/Premier-League-Stats)
- **Scope:** 2025-2026 Big Five European Leagues
- **Key Metrics:**
  - **Shooting:** xG, Shot-Creating Actions (SCA).
  - **Passing:** Progressive Passes, Passes into Final Third, Completion %.
  - **Possession:** Touches (Def/Mid/Att 1/3), Progressive Carries, Successful Take-ons.
  - **Defensive:** Tackles/Interceptions in specific zones.

---

## 🛠️ Methodology

### 1. Spatial Heat Mapping
Using touch-density data across the defensive, middle, and attacking thirds to visualize "roaming" patterns. This provides a spatial foundation for role classification.

### 2. Role Clustering (Unsupervised Learning)
Applying algorithms like **K-Nearest Neighbors (KNN)** or **K-Means Clustering** to a multidimensional vector of metrics:
- **The "Progressive" Vector:** Progressive Carries + Progressive Passes.
- **The "Creative" Vector:** SCA + Key Passes + Passes into Penalty Area.
- **The "Workrate" Vector:** Recoveries + Touches in Middle Third.

### 3. Benchmarking
Once roles are identified, players are benchmarked against their **tactical peers**.
- *Example:* Instead of comparing a fullback to all other fullbacks, they are compared specifically to other "Inverted Playmaking Fullbacks."

---

## 📈 Planned Outputs

| Feature | Description |
| :--- | :--- |
| **Role-DNA Dashboard** | A visual breakdown of a player's style (e.g., 80% Playmaker, 20% Ball-Winner). |
| **Spatial Heatmaps** | Visualizing touch-density across the pitch to validate tactical roles. |
| **Peer Similarity Scores** | Identifying "statistical twins" across different leagues for scouting. |

---

## 🚀 Technical Requirements
- **Language:** Python
- **Key Libraries:** - `Pandas` / `Tidyverse` (Data Wrangling)
    - `Scikit-learn` (Clustering & KNN)
    - `Matplotlib` / `Plotly` / `MPLsoccer` (Visualization)
    - `worldfootballR` (Data Extraction)

---