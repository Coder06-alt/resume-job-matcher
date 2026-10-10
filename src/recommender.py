"""Role recommendation from a structured skills taxonomy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

# Core skills that define each role. Canonical names from skills.TAXONOMY.
ROLE_SKILLS: Dict[str, Set[str]] = {
    "Data Scientist": {
        "python", "sql", "pandas", "numpy", "statistics", "machine learning",
        "scikit-learn", "regression", "classification", "data visualization",
        "matplotlib", "exploratory data analysis", "feature engineering",
        "hypothesis testing",
    },
    "ML Engineer": {
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "docker", "git", "mlops", "model deployment",
        "feature engineering", "rest api", "aws",
    },
    "Data Analyst": {
        "sql", "excel", "python", "pandas", "tableau", "power bi",
        "data visualization", "statistics", "exploratory data analysis",
        "data cleaning", "data analysis",
    },
    "AI Engineer": {
        "python", "deep learning", "nlp", "llm", "transformers", "pytorch",
        "rag", "prompt engineering", "langchain", "hugging face", "fastapi",
    },
    "NLP Engineer": {
        "python", "nlp", "tf-idf", "word embeddings", "spacy", "nltk",
        "transformers", "bert", "text classification", "sentiment analysis",
        "named entity recognition", "hugging face",
    },
    "Data Engineer": {
        "python", "sql", "spark", "kafka", "airflow", "etl", "hadoop",
        "aws", "docker", "big data", "postgresql",
    },
}


@dataclass
class RoleFit:
    role: str
    score: float            # 0..1 = overlap / role size
    matched: List[str]
    missing: List[str]


def recommend_roles(resume_skills: Set[str], top_n: int = 3) -> List[RoleFit]:
    """Rank roles by how many of the role's core skills the candidate has."""
    fits: List[RoleFit] = []
    for role, required in ROLE_SKILLS.items():
        matched = resume_skills & required
        missing = required - resume_skills
        fits.append(RoleFit(role, len(matched) / len(required), sorted(matched), sorted(missing)))
    fits.sort(key=lambda f: f.score, reverse=True)
    return fits[:top_n]
