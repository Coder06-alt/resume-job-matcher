"""Heuristic resume-quality score (0-100) with a good/fix checklist."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .skills import extract_skills

SECTIONS = {
    "education": ["education"],
    "experience": ["experience", "internship", "work history"],
    "projects": ["projects", "project"],
    "skills": ["skills", "technical skills"],
}
ACTION_VERBS = {
    "developed", "built", "designed", "implemented", "led", "created", "improved",
    "deployed", "optimized", "analyzed", "mentored", "reduced", "increased",
    "engineered", "automated", "evaluated", "organized", "coordinated",
}
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|\+|x|k|m)?")


@dataclass
class QualityResult:
    score: int
    good: List[str] = field(default_factory=list)
    fix: List[str] = field(default_factory=list)


def score_resume(text: str) -> QualityResult:
    """Score resume structure and content. Max 100 points:
    sections 40, skills 15, action verbs 15, numbers 15, length 10, contact 5."""
    lowered = text.lower()
    result = QualityResult(score=0)
    points = 0

    # Sections: 10 each
    for name, keys in SECTIONS.items():
        if any(k in lowered for k in keys):
            points += 10
            result.good.append(f"Has a {name} section.")
        else:
            result.fix.append(f"Add a clear {name} section.")

    # Skills found: up to 15
    n_skills = len(extract_skills(text))
    points += min(n_skills, 15)
    if n_skills >= 10:
        result.good.append(f"Lists {n_skills} recognisable technical skills.")
    else:
        result.fix.append(f"Only {n_skills} recognisable skills found; list more relevant ones.")

    # Action verbs: up to 15 (3 points each)
    words = set(re.findall(r"[a-z]+", lowered))
    verbs = len(words & ACTION_VERBS)
    points += min(verbs * 3, 15)
    if verbs >= 5:
        result.good.append("Uses strong action verbs.")
    else:
        result.fix.append("Start bullets with action verbs (built, deployed, improved).")

    # Quantified achievements: up to 15 (3 points each)
    numbers = len(NUMBER_RE.findall(text))
    points += min(numbers * 3, 15)
    if numbers >= 5:
        result.good.append("Quantifies achievements with numbers.")
    else:
        result.fix.append("Add measurable results (%, counts, time saved).")

    # Length: 10 points if 250-900 words
    n_words = len(text.split())
    if 250 <= n_words <= 900:
        points += 10
        result.good.append(f"Good length ({n_words} words).")
    else:
        result.fix.append(f"Length is {n_words} words; aim for roughly 250-900.")

    # Contact info: 5 points
    if EMAIL_RE.search(text) and PHONE_RE.search(text):
        points += 5
        result.good.append("Contact info (email and phone) present.")
    else:
        result.fix.append("Include both email and phone number.")

    result.score = min(points, 100)
    return result
