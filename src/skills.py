"""Skill taxonomy and rule-based skill extraction.

TAXONOMY maps category -> {canonical skill: [aliases]}.
Extraction = phrase matching with word boundaries, so "java" does not match
inside "javascript" and "r" does not match inside "regression".
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Set

TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Programming Languages": {
        "python": ["py"],
        "java": [],
        "c++": ["cpp"],
        "c": [],
        "c#": ["csharp"],
        "javascript": ["js", "es6"],
        "typescript": ["ts"],
        "r": ["r programming"],
        "scala": [],
        "go": ["golang"],
        "rust": [],
        "sql": [],
        "bash": ["shell scripting"],
        "matlab": [],
        "kotlin": [],
    },
    "Machine Learning": {
        "machine learning": ["ml"],
        "deep learning": ["dl"],
        "supervised learning": [],
        "unsupervised learning": [],
        "reinforcement learning": ["rl"],
        "regression": ["linear regression"],
        "logistic regression": [],
        "classification": [],
        "clustering": ["k-means", "kmeans"],
        "random forest": [],
        "decision tree": ["decision trees"],
        "svm": ["support vector machine"],
        "xgboost": [],
        "lightgbm": [],
        "gradient boosting": [],
        "feature engineering": [],
        "model evaluation": [],
        "cross-validation": ["cross validation"],
        "hyperparameter tuning": [],
        "scikit-learn": ["sklearn", "scikit learn"],
        "anomaly detection": [],
        "recommender systems": ["recommendation systems", "recommender system"],
        "time series": ["time-series"],
        "a/b testing": ["ab testing"],
    },
    "Deep Learning Frameworks": {
        "tensorflow": [],
        "keras": [],
        "pytorch": ["torch"],
        "cnn": ["convolutional neural network", "convolutional neural networks"],
        "rnn": ["recurrent neural network"],
        "lstm": [],
        "transformers": ["transformer", "attention mechanism"],
        "bert": [],
        "gan": ["generative adversarial network"],
        "hugging face": ["huggingface"],
        "neural networks": ["neural network"],
        "onnx": [],
    },
    "NLP": {
        "nlp": ["natural language processing"],
        "tf-idf": ["tfidf", "tf idf"],
        "cosine similarity": [],
        "word embeddings": ["word2vec", "glove", "embeddings"],
        "spacy": [],
        "nltk": [],
        "text classification": [],
        "sentiment analysis": [],
        "named entity recognition": ["ner"],
        "topic modeling": ["lda"],
        "text processing": ["text preprocessing"],
        "tokenization": [],
        "llm": ["large language models", "large language model"],
        "rag": ["retrieval augmented generation"],
        "prompt engineering": [],
        "langchain": [],
        "sentence-transformers": ["sentence transformers"],
    },
    "Computer Vision": {
        "computer vision": ["cv"],
        "opencv": [],
        "image classification": [],
        "object detection": [],
        "image processing": [],
        "yolo": [],
    },
    "Data Analysis": {
        "pandas": [],
        "numpy": [],
        "scipy": [],
        "statistics": ["statistical modeling", "statistical analysis"],
        "exploratory data analysis": ["eda"],
        "data cleaning": ["data wrangling"],
        "data mining": [],
        "hypothesis testing": [],
        "probability": [],
        "linear algebra": [],
        "excel": [],
        "jupyter": ["jupyter notebook"],
        "data analysis": [],
    },
    "Data Visualization": {
        "matplotlib": [],
        "seaborn": [],
        "plotly": [],
        "tableau": [],
        "power bi": ["powerbi"],
        "data visualization": ["data visualisation"],
        "streamlit": [],
        "dash": [],
    },
    "Databases": {
        "mysql": [],
        "postgresql": ["postgres"],
        "mongodb": [],
        "redis": [],
        "sqlite": [],
        "elasticsearch": [],
        "nosql": [],
        "dbms": ["rdbms", "relational databases"],
        "cassandra": [],
        "neo4j": [],
    },
    "Big Data": {
        "spark": ["apache spark", "pyspark"],
        "hadoop": [],
        "mapreduce": ["map reduce"],
        "kafka": [],
        "hive": [],
        "airflow": [],
        "databricks": [],
        "etl": ["data pipelines", "data pipeline"],
        "big data": [],
        "dask": [],
    },
    "Cloud & DevOps": {
        "aws": ["amazon web services"],
        "azure": [],
        "gcp": ["google cloud"],
        "docker": [],
        "kubernetes": ["k8s"],
        "ci/cd": ["cicd"],
        "git": [],
        "github": [],
        "linux": ["unix"],
        "terraform": [],
        "jenkins": [],
    },
    "MLOps": {
        "mlops": [],
        "mlflow": [],
        "model deployment": [],
        "model monitoring": [],
        "kubeflow": [],
        "dvc": [],
        "feature store": [],
        "fastapi": [],
        "flask": [],
        "rest api": ["rest apis", "restful api"],
    },
    "Web & Software Engineering": {
        "react": ["react.js", "reactjs"],
        "node.js": ["nodejs", "node"],
        "express.js": ["express"],
        "next.js": ["nextjs"],
        "html": ["html5"],
        "css": ["css3"],
        "tailwind css": ["tailwind"],
        "django": [],
        "data structures": ["dsa", "data structures and algorithms"],
        "algorithms": [],
        "object-oriented programming": ["oop", "object oriented programming"],
        "system design": [],
        "operating systems": [],
        "computer networks": [],
        "distributed systems": [],
        "unit testing": ["pytest"],
    },
    "Trust & Safety": {
        "content moderation": [],
        "hate speech detection": ["toxicity detection"],
        "misinformation detection": [],
        "class imbalance": [],
        "fraud detection": [],
        "spam detection": [],
    },
}


def _build_alias_index() -> Dict[str, str]:
    """Flatten taxonomy into {surface form -> canonical skill}."""
    index: Dict[str, str] = {}
    for skills in TAXONOMY.values():
        for canonical, aliases in skills.items():
            index[canonical] = canonical
            for alias in aliases:
                index[alias] = canonical
    return index


ALIAS_INDEX: Dict[str, str] = _build_alias_index()
SKILL_TO_CATEGORY: Dict[str, str] = {
    skill: cat for cat, skills in TAXONOMY.items() for skill in skills
}
ALL_SKILLS: Set[str] = set(SKILL_TO_CATEGORY)


@lru_cache(maxsize=1)
def _compiled_pattern() -> re.Pattern:
    """One big regex of all surface forms, longest first so
    'machine learning' wins over 'learning' style partial matches."""
    forms = sorted(ALIAS_INDEX, key=len, reverse=True)
    escaped = [re.escape(f) for f in forms]
    # lookarounds act as word boundaries that also work for c++, c#, node.js
    return re.compile(r"(?<![a-z0-9+#.])(" + "|".join(escaped) + r")(?![a-z0-9+#]|\.[a-z0-9])")


def extract_skills(text: str) -> Set[str]:
    """Return the set of canonical skills mentioned in `text`."""
    lowered = text.lower()
    return {ALIAS_INDEX[m.group(1)] for m in _compiled_pattern().finditer(lowered)}


def group_by_category(skills: Set[str]) -> Dict[str, List[str]]:
    """Group a set of skills by taxonomy category (sorted)."""
    grouped: Dict[str, List[str]] = {}
    for skill in sorted(skills):
        grouped.setdefault(SKILL_TO_CATEGORY.get(skill, "Other"), []).append(skill)
    return grouped
