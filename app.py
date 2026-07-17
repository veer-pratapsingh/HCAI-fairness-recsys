"""Streamlit dashboard: Fairness Auditing & Mitigation on OULAD (Group 17, HCAI @ OvGU).

The dashboard tells the paper's story in five tabs and reads every aggregate
number directly from the CSVs in `results/` (the same source of truth as the
paper), so the demo can never contradict the submitted numbers.

Run locally:
    pip install streamlit plotly
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import paths

# ---------------------------------------------------------------- constants

NAVY = "#1F2A5A"
ORANGE = "#E8641E"
GREY = "#B7BFCC"
LIGHT = "#EEF2F8"

MODEL_LABELS = {
    "popularity": "Popularity",
    "cf": "CF (ALS)",
    "sasrec": "SASRec",
    "llm_openai": "LLM (GPT-4o-mini)",
}

ATTR_LABELS = {
    "gender": "Gender",
    "age_binary": "Age (0-35 vs 35+)",
    "disability": "Disability",
    "imd_binary": "IMD (socioeconomic)",
}

# per_group CSVs use these attribute keys for the full (non-binary) breakdown
ATTR_PER_GROUP_KEY = {
    "gender": "gender",
    "age_binary": "age_band",
    "disability": "disability",
    "imd_binary": "imd_binary",
}

st.set_page_config(
    page_title="OULAD Fairness Audit — Group 17",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------- data access

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.RESULTS_DIR / name)


def pc(x: float) -> float:
    """fraction -> percent"""
    return 100.0 * float(x)


summary = load_csv("summary_table.csv").set_index("model")
gaps = load_csv("group_gap_significance.csv")
mcnemar = load_csv("mcnemar_significance.csv")
entropy = load_csv("group_predictability.csv")
llm_runs = load_csv("openai_llm_runs.csv")

# headline numbers (computed, not hard-coded)
POP_R = pc(summary.loc["popularity", "Recall@10_mean"])
CF_R = pc(summary.loc["cf", "Recall@10_mean"])
SAS_R = pc(summary.loc["sasrec", "Recall@10_mean"])
LLM_R = pc(llm_runs["Recall@10"].mean())
LLM_SD = pc(llm_runs["Recall@10"].std(ddof=1))

MC_CF_SAS = mcnemar[(mcnemar.label_a == "cf") & (mcnemar.label_b == "sasrec")].iloc[0]

SAS_IMD = gaps[(gaps.model == "sasrec") & (gaps.attribute == "imd_binary")].iloc[0]
SAS_AGE = gaps[(gaps.model == "sasrec") & (gaps.attribute == "age_binary")].iloc[0]

FAIR_ROW = summary.loc["sasrec_fair_lam1.0"]
RERANK_ROW = summary.loc["sasrec_rerank_a0.7"]
BASE_ROW = summary.loc["sasrec"]


def fig_layout(fig: go.Figure, title: str, ytitle: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY)),
        template="plotly_white",
        height=height,
        yaxis_title=ytitle,
        font=dict(family="Calibri, Arial", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=70, b=40),
    )
    return fig


# ---------------------------------------------------------------- header

st.title("⚖️ Who Does the Recommender Work For?")
st.markdown(
    f"**A fairness audit of next-activity recommenders on OULAD** &nbsp;·&nbsp; "
    f"Group 17 · HCAI @ OvGU &nbsp;·&nbsp; "
    f"<span style='color:{ORANGE}'>every number below is read live from "
    f"<code>results/</code> — the same files behind the paper</span>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 The Audit", "🎯 Accuracy", "⚖️ Fairness Audit", "🔧 Mitigation", "🧑‍🎓 Live Demo"]
)

# ================================================================= TAB 1
with tab1:
    st.subheader("One protocol, four models, four protected groups")
    st.markdown(
        "We predict each student's **next learning activity** (1 of 6,268) from their "
        "click history — 28,761 sessions, leave-last-out split — and audit **who** "
        "each model serves well. Chance level is **0.16%**; all models are compared "
        "on the *identical* hidden clicks."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Popularity — Recall@10", f"{POP_R:.2f}%", "the floor", delta_color="off")
    c2.metric("CF (ALS) — Recall@10", f"{CF_R:.2f}%", "best classical")
    c3.metric("SASRec — Recall@10", f"{SAS_R:.2f}%", "deep sequential", delta_color="off")
    c4.metric("LLM reranker — Recall@10", f"{LLM_R:.2f}% ± {LLM_SD:.2f}", "500 sessions only", delta_color="off")

    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"#### 1 · Simple ties deep")
        st.markdown(
            f"CF and SASRec are **statistically tied** "
            f"(McNemar p = {MC_CF_SAS.p_value:.3f}; CF alone won {int(MC_CF_SAS.a_only)} "
            f"sessions, SASRec alone {int(MC_CF_SAS.b_only)}). Modelling click *order* "
            f"buys nothing here — students follow standard paths. → *Tab 🎯*"
        )
    with f2:
        st.markdown(f"#### 2 · The bias runs backwards")
        st.markdown(
            f"The base models serve **socioeconomically disadvantaged students "
            f"better** than advantaged ones: gap **+{pc(SAS_IMD.gap):.2f} pts** "
            f"(95% CI [{pc(SAS_IMD.ci_low):.2f}, {pc(SAS_IMD.ci_high):.2f}], "
            f"Holm-significant). Older students do worse. A naive audit expects the "
            f"opposite. → *Tab ⚖️*"
        )
    with f3:
        st.markdown(f"#### 3 · The fixes level down")
        st.markdown(
            f"Both mitigations reach parity mainly by **dragging the favoured group "
            f"down**, not lifting anyone up — the gentler fix still costs "
            f"**{100*(RERANK_ROW['Recall@10_mean']/BASE_ROW['Recall@10_mean']-1):.1f}%** "
            f"of overall recall. In education that is the wrong kind of equality. → *Tab 🔧*"
        )

    st.info(
        "**The human-centred lesson:** a single fairness number can point the wrong "
        "way. Always read parity metrics *beside absolute per-group performance*, "
        "and keep a human in the loop before 'fixing' anything.",
        icon="🧭",
    )

# ================================================================= TAB 2
with tab2:
    st.subheader("Accuracy under one identical protocol")

    metric = st.radio(
        "Metric", ["Recall@10", "NDCG@10", "MRR"], horizontal=True, key="acc_metric"
    )

    rows = ["popularity", "cf", "sasrec"]
    means = [pc(summary.loc[m, f"{metric}_mean"]) for m in rows]
    err_lo = [pc(summary.loc[m, f"{metric}_mean"]) - pc(summary.loc[m, f"{metric}_ci_lower"]) for m in rows]
    err_hi = [pc(summary.loc[m, f"{metric}_ci_upper"]) - pc(summary.loc[m, f"{metric}_mean"]) for m in rows]

    # LLM from its own runs file (3 seeds, 500 sessions each)
    llm_mean = pc(llm_runs[metric].mean())
    llm_sd = pc(llm_runs[metric].std(ddof=1))

    labels = [MODEL_LABELS[m] for m in rows] + ["LLM (GPT-4o-mini)*"]
    colors = [GREY, ORANGE, NAVY, "#8892B0"]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=means + [llm_mean],
            marker_color=colors,
            error_y=dict(
                type="data",
                array=err_hi + [llm_sd],
                arrayminus=err_lo + [llm_sd],
                color="#444",
            ),
            text=[f"{v:.2f}%" for v in means + [llm_mean]],
            textposition="outside",
        )
    )
    fig.add_hline(y=100 * 10 / 6268, line_dash="dot", line_color="red",
                  annotation_text="chance (10/6268 ≈ 0.16%)", annotation_font_color="red")
    st.plotly_chart(fig_layout(fig, f"{metric} across the four families (5 seeds, 95% CI)", f"{metric} (%)"),
                    use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.success(
            f"**CF vs SASRec — statistically tied.** Session-level McNemar on all "
            f"{int(MC_CF_SAS.n_pairs):,} shared test questions: CF alone correct on "
            f"{int(MC_CF_SAS.a_only)}, SASRec alone on {int(MC_CF_SAS.b_only)}, "
            f"p = {MC_CF_SAS.p_value:.3f} → no significant difference. "
            f"**H1 (deep model wins) is refuted.**",
            icon="🤝",
        )
    with cb:
        st.warning(
            f"***LLM caveat:** evaluated on 500 sessions/seed (API cost), 3 seeds — "
            f"hence the wide ±{llm_sd:.2f} band. Its NDCG/MRR are *low*: it keeps the "
            f"hit in the top-10 but ranks it poorly (known reranker behaviour). We call "
            f"it 'similar to CF but noisier', never 'best'.",
            icon="🤖",
        )

# ================================================================= TAB 3
with tab3:
    st.subheader("Per-group audit: who is served well?")

    cm, cat = st.columns(2)
    model_sel = cm.selectbox("Model", ["sasrec", "cf", "popularity"],
                             format_func=lambda m: MODEL_LABELS[m])
    attr_sel = cat.selectbox("Protected attribute", list(ATTR_LABELS),
                             format_func=lambda a: ATTR_LABELS[a], index=3)

    per_group = load_csv(f"{model_sel}_per_group.csv")
    pg = per_group[(per_group.attribute == ATTR_PER_GROUP_KEY[attr_sel])
                   & (~per_group.group_value.str.contains("GAP"))]

    grow = gaps[(gaps.model == model_sel) & (gaps.attribute == attr_sel)].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        bar_colors = [ORANGE if g == grow.group_focus else NAVY for g in pg.group_value]
        fig = go.Figure(go.Bar(
            x=pg.group_value, y=[pc(v) for v in pg.recall],
            marker_color=bar_colors,
            text=[f"{pc(v):.2f}%" for v in pg.recall], textposition="outside",
        ))
        fig.add_hline(y=pc(summary.loc[model_sel, "Recall@10_mean"]), line_dash="dot",
                      line_color=GREY, annotation_text="overall average")
        st.plotly_chart(
            fig_layout(fig, f"Recall@10 per group — {MODEL_LABELS[model_sel]}, "
                            f"{ATTR_LABELS[attr_sel]}", "Recall@10 (%)"),
            use_container_width=True)
        st.caption("Orange = the group a conventional audit watches ('unprivileged'). "
                   "Small groups (55+, IMD-unknown) shown here but excluded from the "
                   "binary gap statistic — reported, never silently dropped.")

    with right:
        sig = bool(grow.significant_holm)
        st.metric(
            f"Gap: {grow.group_focus} − {grow.group_other}",
            f"{pc(grow.gap):+.2f} pts",
            f"95% CI [{pc(grow.ci_low):.2f}, {pc(grow.ci_high):.2f}]",
            delta_color="off",
        )
        if sig:
            st.markdown(f"**✅ Significant after Holm correction** (p_holm = {grow.p_holm:.2g})")
        else:
            st.markdown(f"**⬜ Not significant after Holm** (p_holm = {grow.p_holm:.2g})")

        if attr_sel == "imd_binary":
            st.error(
                "**The headline reversal:** disadvantaged students get *better* "
                "recommendations than advantaged ones — the opposite of the "
                "expected direction. Positive gap = 'unprivileged' group favoured.",
                icon="🔄",
            )
        if attr_sel == "age_binary":
            st.info("Age is the one gap in the *expected* direction: 35+ students "
                    "are served significantly worse.", icon="👵")

        with st.expander("Did 'predictability' explain it? (entropy check)"):
            ent = entropy[entropy.attribute.isin([attr_sel, ATTR_PER_GROUP_KEY[attr_sel]])]
            for _, r in ent.iterrows():
                st.markdown(f"- **{r['group']}**: {r.mean_cond_entropy_bits:.2f} bits")
            st.caption(
                "Lower entropy = more predictable click paths. The hypothesis "
                "('predictability privilege') fits disability, but IMD and age groups "
                "are near-identical — so it does **not** explain the significant "
                "reversals. Reported honestly as a partial explanation (descriptive only)."
            )

# ================================================================= TAB 4
with tab4:
    st.subheader("What does 'fixing' fairness cost — and who pays?")

    lam_models = [m for m in summary.index if m.startswith("sasrec_fair_lam")]
    a_models = [m for m in summary.index if m.startswith("sasrec_rerank_a")]

    def frontier_trace(models: list[str], name: str, color: str, prefix: str, strip: str):
        return go.Scatter(
            x=[summary.loc[m, "imd_EOD_mean"] for m in models],
            y=[pc(summary.loc[m, "Recall@10_mean"]) for m in models],
            mode="lines+markers+text",
            text=[prefix + m.replace(strip, "") for m in models],
            textposition="top center",
            name=name, marker=dict(size=10, color=color),
        )

    fig = go.Figure()
    fig.add_trace(frontier_trace(sorted(lam_models), "Fix 1 — fairness loss (retrain)",
                                 NAVY, "λ=", "sasrec_fair_lam"))
    fig.add_trace(frontier_trace(sorted(a_models), "Fix 2 — post-hoc rerank (cheap)",
                                 ORANGE, "α=", "sasrec_rerank_a"))
    fig.add_annotation(x=summary.loc["sasrec", "imd_EOD_mean"],
                       y=pc(summary.loc["sasrec", "Recall@10_mean"]),
                       text="base SASRec", showarrow=True, arrowhead=2)
    fig.update_xaxes(title="IMD unfairness — EOD (→ 0 = parity)")
    st.plotly_chart(fig_layout(fig, "Fairness–accuracy trade-off (IMD, 5 seeds)",
                               "Recall@10 (%)", 480), use_container_width=True)
    st.caption("Every step toward parity (left) moves down in accuracy. Reranking (α) "
               "gives the gentler slope → the better trade-off. H2 (<5% cost) only "
               "partially supported: even α=0.7 costs ~13% relative recall.")

    st.markdown("#### …and *who* pays: levelling down (per-group view, IMD)")
    conditions = {
        "Base SASRec": "sasrec_per_group.csv",
        "Fix 1 · fair-loss λ=1.0": "sasrec_fair_lam1.0_per_group.csv",
        "Fix 2 · rerank α=0.7": "sasrec_rerank_a0.7_per_group.csv",
    }
    disadv, adv = [], []
    for f in conditions.values():
        d = load_csv(f)
        d = d[d.attribute == "imd_binary"].set_index("group_value")
        disadv.append(pc(d.loc["disadvantaged", "recall"]))
        adv.append(pc(d.loc["advantaged", "recall"]))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Disadvantaged (favoured by base model)",
                          x=list(conditions), y=disadv, marker_color=ORANGE,
                          text=[f"{v:.2f}%" for v in disadv], textposition="outside"))
    fig2.add_trace(go.Bar(name="Advantaged", x=list(conditions), y=adv,
                          marker_color=NAVY,
                          text=[f"{v:.2f}%" for v in adv], textposition="outside"))
    st.plotly_chart(fig_layout(fig2, "Both groups get WORSE — parity by suppression, "
                                     "not by lifting", "Recall@10 (%)", 440),
                    use_container_width=True)
    st.error(
        "**Levelling down:** the gap narrows because the *favoured* group falls "
        "furthest — nobody is helped. In education this is ethically contested; a "
        "deployment decision like this must stay with a **human in the loop**.",
        icon="⚠️",
    )
    st.caption("Trade-off frontier: 5-seed means from summary_table.csv. Per-group "
               "bars: seed-0 per-group files (as in the paper, Sec. 6.2).")

# ================================================================= TAB 5
with tab5:
    st.subheader("Live demo: one real (anonymised) student")
    st.markdown(
        "Everything so far was aggregate evidence. This tab runs **real models "
        "live** on one student's actual click history — the human-in-the-loop view."
    )

    if not st.checkbox("Enable live demo (trains Popularity + CF for one course, ~15 s once, then cached)"):
        st.stop()

    from src.data.splits import load_splits
    from src.models.base import Context
    from src.models.cf import CFRecommender
    from src.models.popularity import PopularityRecommender
    from src.mitigation.rerank import RerankingRecommender

    @st.cache_data
    def load_seq_data():
        splits = load_splits(write=False)
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        return splits, vocab

    splits, vocab = load_seq_data()
    idx_to_type = vocab.set_index("item_idx")["activity_type"].to_dict()
    idx_to_site = vocab.set_index("item_idx")["id_site"].to_dict()

    pres_list = splits.groupby(["code_module", "code_presentation"]).size().index.tolist()
    sel = st.selectbox("Course presentation", pres_list, format_func=lambda t: f"{t[0]} / {t[1]}")
    pres_splits = splits[(splits.code_module == sel[0]) & (splits.code_presentation == sel[1])]

    @st.cache_resource
    def fit_models(module: str, presentation: str):
        train = splits[(splits.code_module == module) & (splits.code_presentation == presentation)]
        pop = PopularityRecommender().fit(train)
        cf = CFRecommender(factors=64, iterations=20, seed=0).fit(train)  # paper hyperparameters
        return pop, cf

    pop_model, cf_model = fit_models(*sel)

    student = st.selectbox("Student session", pres_splits.id_student.tolist())
    row = pres_splits[pres_splits.id_student == student].iloc[0]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Gender", row.gender)
    p2.metric("Age band", row.age_band)
    p3.metric("Disability", "Yes" if row.disability == "Y" else "No")
    p4.metric("IMD group", row.imd_binary)

    model_choice = st.radio(
        "Recommender",
        ["CF (ALS) — paper settings", "Popularity baseline", "Fix 2 — fairness rerank on top of CF"],
        horizontal=True,
    )
    history = list(row.test_input)  # everything except the hidden last click
    ctx = Context(code_module=sel[0], code_presentation=sel[1], id_student=int(student))

    if model_choice.startswith("CF"):
        recs = cf_model.recommend(history, k=10, context=ctx)
    elif model_choice.startswith("Popularity"):
        recs = pop_model.recommend(history, k=10, context=ctx)
    else:
        alpha = st.slider("α — fairness strength (0 = pure CF, 1 = pure group frequency)",
                          0.0, 1.0, 0.7, 0.1)
        rr = RerankingRecommender(base=cf_model, alpha=alpha, fair_attr="imd_binary")
        rr.fit(pres_splits)
        recs = rr.recommend(history, k=10, context=ctx)

    target = int(row.test_target)
    hit = target in recs

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown(f"**Click history** ({len(history)} visits — model input)")
        hist_df = pd.DataFrame({
            "#": range(1, len(history) + 1),
            "activity type": [idx_to_type.get(i, "?") for i in history],
            "VLE site": [idx_to_site.get(i, 0) for i in history],
        })
        st.dataframe(hist_df.tail(15), use_container_width=True, height=380)
        st.caption("Showing the last 15 visits; the model sees all of them.")
    with rcol:
        st.markdown("**Top-10 recommendation** (predicted next click)")
        rec_df = pd.DataFrame({
            "rank": range(1, len(recs) + 1),
            "activity type": [idx_to_type.get(i, "?") for i in recs],
            "VLE site": [idx_to_site.get(i, 0) for i in recs],
            "hidden next click?": ["🎯 YES" if i == target else "" for i in recs],
        })
        st.dataframe(rec_df, use_container_width=True, height=380)

    if hit:
        st.success(f"**HIT** — the student's real next click (site {idx_to_site.get(target)}, "
                   f"{idx_to_type.get(target, '?')}) is at rank {recs.index(target) + 1}. "
                   f"This is exactly what Recall@10 counts, once per 28,761 sessions.", icon="🎯")
    else:
        st.error(f"**MISS** — the real next click was site {idx_to_site.get(target)} "
                 f"({idx_to_type.get(target, '?')}), not in the top-10. With 6,268 candidates "
                 f"this is the common case — hence ~4% recall, ~26× better than chance.", icon="❌")

st.markdown("---")
st.caption("Group 17 · OvGU HCAI · Data: OULAD (CC-BY 4.0). Aggregate figures read from "
           "results/*.csv (5 seeds where available); live demo trains Popularity + CF "
           "with the paper's hyperparameters on the selected course presentation.")
