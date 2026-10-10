"""Streamlit dashboard for the AI Resume-Job Matcher."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.gap_analysis import analyze_gap
from src.parser import ResumeParseError, extract_text
from src.quality import score_resume
from src.recommender import recommend_roles
from src.scoring import SKILL_WEIGHT, TFIDF_WEIGHT, compute_match
from src.skills import SKILL_TO_CATEGORY, group_by_category

st.set_page_config(page_title="AI Resume-Job Matcher", page_icon="📄", layout="wide")


@st.cache_data(show_spinner=False)
def run_analysis(resume_text: str, job_text: str):
    """Cached so re-rendering the page does not recompute the pipeline."""
    match = compute_match(resume_text, job_text)
    gap = analyze_gap(match.resume_skills, match.job_skills)
    roles = recommend_roles(match.resume_skills)
    quality = score_resume(resume_text)
    return match, gap, roles, quality


def gauge(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"}},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def chips(skills, color: str) -> str:
    if not skills:
        return "_None_"
    return " ".join(
        f"<span style='background:{color};padding:3px 10px;border-radius:12px;"
        f"margin:2px;display:inline-block;font-size:0.85rem'>{s}</span>"
        for s in sorted(skills)
    )


st.title("📄 AI Resume-Job Matcher")
st.caption("Hybrid scoring: TF-IDF cosine similarity + rule-based skill matching")

with st.sidebar:
    st.header("How to use")
    st.markdown("1. Upload your resume (PDF, DOCX or TXT)\n2. Paste a job description\n3. Click **Analyze**")
    uploaded = st.file_uploader("Resume", type=["pdf", "docx", "txt"])
    st.markdown(f"**Weights:** TF-IDF {TFIDF_WEIGHT:.0%} · Skills {SKILL_WEIGHT:.0%}")

job_text = st.text_area("Paste the job description", height=220)

if st.button("Analyze", type="primary"):
    if uploaded is None or not job_text.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        try:
            resume_text = extract_text(uploaded, uploaded.name)
        except ResumeParseError as err:
            st.error(str(err))
            st.stop()
        st.session_state["results"] = run_analysis(resume_text, job_text)

if "results" in st.session_state:
    match, gap, roles, quality = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Skills", "Recommendations", "Role Fit", "Resume Quality"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(gauge(match.final_score * 100), use_container_width=True)
        c2.metric("TF-IDF similarity", f"{match.tfidf_score:.0%}")
        c3.metric("Skill match", f"{match.skill_score:.0%}")
        st.write(
            f"Final = {TFIDF_WEIGHT} × TF-IDF + {SKILL_WEIGHT} × skill match. "
            f"Matched {len(gap.matched)} of {len(match.job_skills)} skills the job asks for."
        )

    with tab2:
        a, b, c = st.columns(3)
        a.markdown("**✅ Matched**")
        a.markdown(chips(gap.matched, "#bbf7d0"), unsafe_allow_html=True)
        b.markdown("**❌ Missing**")
        b.markdown(chips(gap.missing, "#fecaca"), unsafe_allow_html=True)
        c.markdown("**➕ Extra (not in job)**")
        c.markdown(chips(gap.extra, "#e5e7eb"), unsafe_allow_html=True)

        rows = []
        for skill in match.job_skills:
            rows.append(
                {
                    "category": SKILL_TO_CATEGORY.get(skill, "Other"),
                    "status": "Matched" if skill in gap.matched else "Missing",
                }
            )
        if rows:
            df = pd.DataFrame(rows).value_counts().reset_index(name="count")
            fig = px.bar(df, x="category", y="count", color="status", barmode="stack",
                         color_discrete_map={"Matched": "#22c55e", "Missing": "#ef4444"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recognisable skills found in the job description.")

    with tab3:
        for rec in gap.recommendations:
            st.markdown(f"- {rec}")

    with tab4:
        df = pd.DataFrame([{"Role": r.role, "Fit %": round(r.score * 100, 1)} for r in roles])
        st.plotly_chart(px.bar(df, x="Fit %", y="Role", orientation="h", range_x=[0, 100]),
                        use_container_width=True)
        for r in roles:
            with st.expander(f"{r.role}: {r.score:.0%}"):
                st.write("Have:", ", ".join(r.matched) or "none")
                st.write("Still missing:", ", ".join(r.missing) or "none")

    with tab5:
        st.metric("Resume quality", f"{quality.score}/100")
        st.progress(quality.score / 100)
        left, right = st.columns(2)
        left.markdown("**Good**")
        for g in quality.good:
            left.markdown(f"- ✅ {g}")
        right.markdown("**Fix**")
        for f in quality.fix:
            right.markdown(f"- ⚠️ {f}")
