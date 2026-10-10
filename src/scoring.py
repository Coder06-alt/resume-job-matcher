"""Hybrid scoring: TF-IDF cosine similarity + explicit skill matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import preprocess
from .skills import extract_skills

# Configurable weights (must sum to 1). Skills get more weight because they are
# explicit, reliable signals; TF-IDF adds broader context overlap.
TFIDF_WEIGHT: float = 0.4
SKILL_WEIGHT: float = 0.6


@dataclass
class MatchResult:
    tfidf_score: float      # 0..1
    skill_score: float      # 0..1
    final_score: float      # 0..1
    resume_skills: Set[str]
    job_skills: Set[str]


def tfidf_similarity(resume_text: str, job_text: str) -> float:
    """Cosine similarity between TF-IDF vectors (unigrams + bigrams)."""
    docs = [preprocess(resume_text), preprocess(job_text)]
    if not docs[0].strip() or not docs[1].strip():
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def skill_match_score(resume_skills: Set[str], job_skills: Set[str]) -> float:
    """Fraction of the job's required skills present in the resume."""
    if not job_skills:
        return 0.0
    return len(resume_skills & job_skills) / len(job_skills)


def compute_match(
    resume_text: str,
    job_text: str,
    tfidf_weight: float = TFIDF_WEIGHT,
    skill_weight: float = SKILL_WEIGHT,
) -> MatchResult:
    """Run the full hybrid scoring pipeline."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    tfidf_score = tfidf_similarity(resume_text, job_text)
    skill_score = skill_match_score(resume_skills, job_skills)
    final = tfidf_weight * tfidf_score + skill_weight * skill_score
    return MatchResult(tfidf_score, skill_score, final, resume_skills, job_skills)
