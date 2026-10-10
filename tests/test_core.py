"""Unit tests. Run with: pytest -q"""
import pytest

from src.gap_analysis import analyze_gap
from src.preprocess import preprocess, tokenize
from src.quality import score_resume
from src.recommender import recommend_roles
from src.scoring import compute_match, skill_match_score, tfidf_similarity
from src.skills import extract_skills


def test_skill_extraction_aliases_and_boundaries():
    skills = extract_skills("Built models with sklearn and JS; also C++ and Java.")
    assert "scikit-learn" in skills       # alias
    assert "javascript" in skills         # alias "js"
    assert "c++" in skills and "java" in skills
    # "java" must not be found inside "javascript"
    assert "java" not in extract_skills("I know javascript only")


def test_tech_tokens_survive_preprocessing():
    tokens = tokenize("Experienced in C++, node.js and scikit-learn")
    assert "c++" in tokens and "node.js" in tokens and "scikit-learn" in tokens


def test_identical_text_has_similarity_one():
    text = "python machine learning engineer with nlp experience"
    assert tfidf_similarity(text, text) == pytest.approx(1.0)


def test_skill_score_fraction():
    assert skill_match_score({"python", "sql"}, {"python", "sql", "docker", "aws"}) == 0.5
    assert skill_match_score({"python"}, set()) == 0.0


def test_hybrid_weights_and_gap():
    resume = "Python, pandas, scikit-learn, machine learning projects."
    job = "Looking for Python, machine learning, docker and aws skills."
    m = compute_match(resume, job)
    assert m.final_score == pytest.approx(0.4 * m.tfidf_score + 0.6 * m.skill_score)
    gap = analyze_gap(m.resume_skills, m.job_skills)
    assert {"docker", "aws"} <= gap.missing
    assert {"python", "machine learning"} <= gap.matched


def test_role_recommender_prefers_relevant_role():
    skills = {"sql", "excel", "tableau", "power bi", "pandas", "data cleaning"}
    assert recommend_roles(skills)[0].role == "Data Analyst"


def test_quality_score_in_range():
    q = score_resume("Education\nSkills: python\nProjects\nExperience\nBuilt things")
    assert 0 <= q.score <= 100
