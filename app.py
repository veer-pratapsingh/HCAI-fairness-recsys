"""Streamlit Web App: HCAI Fairness Auditing & Mitigation Dashboard.

Run locally using:
    pip install streamlit matplotlib seaborn
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import paths
from src.data.splits import load_splits
from src.models.base import Context
from src.models.popularity import PopularityRecommender
from src.models.cf import CFRecommender
from src.mitigation.rerank import RerankingRecommender
from src.mitigation.calibration import CalibratedReranker

# Page Config
st.set_page_config(
    page_title="HCAI Recommender Fairness Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f5f7f9;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    splits = load_splits(write=False)
    vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    
    # Map item_idx -> activity_type and code
    idx_to_type = vocab.set_index('item_idx')['activity_type'].to_dict()
    idx_to_code = vocab.set_index('item_idx')['id_site'].to_dict()
    
    return splits, vocab, sequences, idx_to_type, idx_to_code


# 1. Load Data
splits, vocab, sequences, idx_to_type, idx_to_code = load_data()

# Page Header
st.title("🎓 HCAI: Fairness Auditing & Mitigation Dashboard")
st.markdown("An interactive dashboard to audit next-activity recommenders and evaluate fairness mitigations on the **OULAD** dataset.")

# Sidebar controls
st.sidebar.header("🛠️ Configurations & Controls")

# Select presentation
presentations = splits.groupby(['code_module', 'code_presentation']).size().index.tolist()
pres_labels = [f"{m} / {p}" for m, p in presentations]
selected_pres_idx = st.sidebar.selectbox(
    "1. Select Course Module / Presentation",
    range(len(pres_labels)),
    format_func=lambda x: pres_labels[x]
)
selected_m, selected_p = presentations[selected_pres_idx]

# Filter splits for presentation
pres_splits = splits[(splits['code_module'] == selected_m) & (splits['code_presentation'] == selected_p)].copy()

# Fit models for this presentation
@st.cache_resource
def get_fitted_base_models(module, presentation):
    # Filter splits
    train_df = splits[(splits['code_module'] == module) & (splits['code_presentation'] == presentation)]
    
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    
    cf_model = CFRecommender(factors=32, iterations=15)
    cf_model.fit(train_df)
    
    return pop_model, cf_model

pop_model, cf_model = get_fitted_base_models(selected_m, selected_p)

# Select student session
sessions = pres_splits['id_student'].tolist()
selected_student = st.sidebar.selectbox(
    "2. Select Student Session (ID)",
    sessions
)

student_row = pres_splits[pres_splits['id_student'] == selected_student].iloc[0]

# Display student demographics
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Student Profile")
st.sidebar.markdown(f"**Gender**: `{student_row['gender']}`")
st.sidebar.markdown(f"**Age Band**: `{student_row['age_band']}`")
st.sidebar.markdown(f"**Disability Status**: `{'Yes' if student_row['disability'] == 'Y' else 'No'}`")
st.sidebar.markdown(f"**Socioeconomic Decile (IMD)**: `{student_row['imd_band']}`")
st.sidebar.markdown(f"**Group (Binarized)**: `{student_row['imd_binary']}`")

# Model configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Recommender Model Settings")
model_choice = st.sidebar.radio(
    "3. Select Recommender Model",
    ["Popularity Baseline", "Collaborative Filtering (ALS)", "Group Rerank Recommender", "Calibrated MMR Reranker"]
)

alpha = 0.5
lambda_cal = 0.5

if model_choice == "Group Rerank Recommender":
    alpha = st.sidebar.slider("Group Affinity Weight (alpha)", 0.0, 1.0, 0.7, step=0.1)
    st.sidebar.caption("alpha=0 maps to CF, alpha=1 maps to group training frequencies.")
elif model_choice == "Calibrated MMR Reranker":
    lambda_cal = st.sidebar.slider("Calibration weight (lambda)", 0.0, 1.0, 0.5, step=0.1)
    st.sidebar.caption("lambda=0 maps to CF, lambda=1 maps to maximum category match.")

# Get recommendations
history = list(student_row['train_history'])
context = Context(code_module=selected_m, code_presentation=selected_p, id_student=selected_student)

if model_choice == "Popularity Baseline":
    recs = pop_model.recommend(history, k=10, context=context)
elif model_choice == "Collaborative Filtering (ALS)":
    recs = cf_model.recommend(history, k=10, context=context)
elif model_choice == "Group Rerank Recommender":
    rerank_model = RerankingRecommender(base=cf_model, alpha=alpha)
    rerank_model.fit(pres_splits)
    recs = rerank_model.recommend(history, k=10, context=context)
elif model_choice == "Calibrated MMR Reranker":
    cal_model = CalibratedReranker(base=cf_model, lambda_cal=lambda_cal)
    cal_model.fit(pres_splits)
    recs = cal_model.recommend(history, k=10, context=context)

# MAIN PANEL
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Clickstream History & Recommendations")
    
    # Render History
    history_types = [idx_to_type.get(item_idx, 'unknown') for item_idx in history]
    history_codes = [idx_to_code.get(item_idx, 0) for item_idx in history]
    
    hist_df = pd.DataFrame({
        'Seq Pos': range(1, len(history) + 1),
        'VLE Site Code': history_codes,
        'Activity Type': history_types
    })
    
    with st.expander(f"📖 View Full History ({len(history)} clicks completed)", expanded=True):
        st.dataframe(hist_df, use_container_width=True)
        
    # Render Recommendations
    rec_types = [idx_to_type.get(item_idx, 'unknown') for item_idx in recs]
    rec_codes = [idx_to_code.get(item_idx, 0) for item_idx in recs]
    
    rec_df = pd.DataFrame({
        'Rank': range(1, len(recs) + 1),
        'VLE Site Code': rec_codes,
        'Activity Type': rec_types
    })
    
    st.markdown("### 🎯 Recommended Activities (Top-10 Next-Clicks)")
    st.table(rec_df)
    
    target_idx = student_row['test_target']
    target_type = idx_to_type.get(target_idx, 'unknown')
    target_code = idx_to_code.get(target_idx, 0)
    
    st.info(f"🎯 **Student's Actual Next Click (Ground Truth)**: Site `{target_code}` (Type: `{target_type}`) " +
            f"— **{'Hit! 🎉' if target_idx in recs else 'Miss ❌'}** (Rank: {recs.index(target_idx)+1 if target_idx in recs else 'N/A'})")

with col2:
    st.subheader("📊 Audit & Calibration Panel")
    
    # 1. Activity Type Distribution in recommendations
    rec_counts = pd.Series(rec_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(rec_counts, labels=rec_counts.index, autopct='%1.0f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')
    ax.set_title("Recommended Types")
    st.pyplot(fig)
    
    # 2. Historical demographic target comparison
    st.markdown("### ⚖️ Group Engagement Benchmark")
    group_val = student_row['imd_binary']
    
    # Compute group engagement distribution in training
    pres_train_histories = []
    for h in pres_splits[pres_splits['imd_binary'] == group_val]['train_history']:
        pres_train_histories.extend(h)
    
    group_history_types = [idx_to_type.get(item_idx, 'unknown') for item_idx in pres_train_histories]
    group_distribution = pd.Series(group_history_types).value_counts(normalize=True)
    
    # Compute KL divergence
    rec_distribution = pd.Series(rec_types).value_counts(normalize=True)
    # Align indexes
    all_types = vocab['activity_type'].unique()
    q_target = np.array([group_distribution.get(t, 0.0) for t in all_types])
    p_rec = np.array([rec_distribution.get(t, 0.0) for t in all_types])
    # Add smoothing
    q_target = (q_target + 1e-5) / (q_target.sum() + len(all_types)*1e-5)
    p_rec = (p_rec + 1e-5) / (p_rec.sum() + len(all_types)*1e-5)
    kl_div = np.sum(p_rec * np.log(p_rec / q_target))
    
    st.markdown(f"**Active Demographic Group**: `{group_val}`")
    st.metric("KL Divergence (Calibration Error)", f"{kl_div:.4f}", help="Difference between recommended type distribution and historical group distribution. Lower is better calibrated.")
    
    # Visual comparison bar chart
    compare_df = pd.DataFrame({
        'Type': all_types,
        'Recommended (%)': [rec_distribution.get(t, 0.0)*100 for t in all_types],
        'Group Historical Average (%)': [group_distribution.get(t, 0.0)*100 for t in all_types]
    })
    compare_df = compare_df[(compare_df['Recommended (%)'] > 0) | (compare_df['Group Historical Average (%)'] > 0.05)]
    compare_df = compare_df.melt(id_vars='Type', var_name='Distribution', value_name='Percentage')
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=compare_df, y='Type', x='Percentage', hue='Distribution', ax=ax2, palette="muted")
    ax2.set_xlabel("Percentage (%)")
    ax2.set_ylabel("")
    ax2.set_title("Distribution Alignment")
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Developed as a companion demonstration for the **HCAI Course (OvGU)**. Designed by Akshat, Dhairithri, Veer, and Harshit.")
