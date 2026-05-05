# ⚽ Advanced Soccer Tactical Profiling & Role Analysis

## 📌 Project Overview
**Analytical Question:**  
How can we use granular event-based metrics to mathematically define a player's tactical role regardless of their listed position?

Traditional football scouting relies on rigid positional labels (e.g., “Midfielder”). This project moves beyond those labels by building a **Hybrid Tactical Intelligence System** that defines player identity through data. Instead of relying on position tags, we construct a player's **Role-DNA** using performance metrics, clustering, and similarity matching.

This allows us to distinguish archetypes such as a *Deep-Lying Playmaker* vs a *Box-to-Box Engine*, even when both are listed as central midfielders.

---

## 📊 Dataset & Sources
- **Source:** FBref (StatsBomb-derived event data)
- **Scope:** 2025–2026 Big Five European Leagues
- **Format:** Cleaned CSV dataset (`pl_master_clean.csv`)

### Key Metrics
- **Shooting:** Goals, Shots/90
- **Passing:** Assists, Progressive Passes, Key Passes, Shots on Target
- **Possession:** Crosses, Fouls, Dribbles, Touch-related proxies
- **Defensive:** Tackles Won, Interceptions

---

## 🛠️ Methodology

### 1. Hybrid Role Classification System (Rule-Based + Data-Driven)
Each player is first assigned a **baseline tactical role** using rule-based heuristics on key performance metrics:

- Finisher (high shots + goals)
- Wide Attacker (cross volume + attacking output)
- Creator (assists + crossing + chance creation)
- Ball Winner (tackles + interceptions)
- Defender (low attacking involvement)
- Hybrid (fallback category)

This provides interpretable baseline role labeling.

---

### 2. Role Clustering (Unsupervised Learning)
Players are then embedded into a standardized feature space using:

- StandardScaler normalization
- Feature groupings:
  - **Progressive Index:** Shots, Crosses, Fouls
  - **Creative Index:** Assists, Goals, Shooting Output
  - **Workrate Index:** Tackles Won, Interceptions

A **K-Means clustering model** is applied to identify natural groupings of player profiles.

> This step uncovers hidden tactical archetypes beyond traditional positions.

---

### 3. Similarity-Based Scouting (Statistical Twins)
To support scouting and comparison:

- Cosine similarity is computed across the full feature space
- Players are matched to their closest statistical peers
- “Statistical twins” are identified across leagues and positions

Example:
> A right-back is compared to other high-progression inverted fullbacks rather than all defenders.

---

### 4. Spatial Heat Mapping (Influence Approximation)
Player influence zones are visualized using pitch heatmaps via `mplsoccer` and kernel density estimation.

> Note: These are **statistical approximations of spatial behavior**, not real event-tracking coordinates.

---

## 📈 Outputs

| Feature | Description |
| :--- | :--- |
| Role-DNA Dashboard | Visual breakdown of player style composition |
| Spatial Heatmaps | Visual representation of player influence zones |
| Statistical Twins | Identifies closest player matches using cosine similarity |
| Tactical Clusters | Unsupervised grouping of player archetypes |

---

## 🚀 Technical Stack

- **Language:** Python  
- **Data Handling:** pandas, numpy  
- **Machine Learning:** scikit-learn  
  - K-Means Clustering  
  - StandardScaler  
  - Cosine Similarity  
- **Visualization:** plotly, matplotlib, seaborn, mplsoccer  

---

## 🧠 Methodological Clarifications

To ensure transparency and reproducibility, the following design decisions are reflected in this project:

- This project uses **K-Means clustering** for unsupervised role discovery rather than K-Nearest Neighbors (KNN).
- The dataset is based on **FBref-derived aggregated performance statistics** and does not include a custom scraping pipeline within this repository.
- Spatial heatmaps are **statistical approximations of player influence**, generated using kernel density estimation rather than raw event-tracking coordinates.
- Player roles are determined using a **hybrid system** combining rule-based classification, clustering outputs, and cosine similarity-based matching.

---

## 🧠 Summary
This project builds a data-driven scouting system that moves beyond positional labels and instead models players through a combination of:

- Interpretable rule-based classification  
- Unsupervised machine learning clustering  
- High-dimensional similarity matching  
- Spatial influence visualization