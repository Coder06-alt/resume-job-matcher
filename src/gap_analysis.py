"""Skill-gap analysis and resume-improvement recommendations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .skills import group_by_category


@dataclass
class GapResult:
    matched: Set[str]
    missing: Set[str]
    extra: Set[str]
    recommendations: List[str]


def analyze_gap(resume_skills: Set[str], job_skills: Set[str]) -> GapResult:
    """Compare resume skills with job skills and build suggestions."""
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    extra = resume_skills - job_skills
    return GapResult(matched, missing, extra, build_recommendations(missing))


def build_recommendations(missing: Set[str]) -> List[str]:
    """Turn missing skills into actionable advice, grouped by category."""
    if not missing:
        return ["Great coverage: your resume mentions every skill found in the job description."]
    recs: List[str] = []
    grouped: Dict[str, List[str]] = group_by_category(missing)
    for category, skills in grouped.items():
        names = ", ".join(skills)
        recs.append(
            f"[{category}] Add these to your skills section only if you have used them: {names}."
        )
        recs.append(
            f"[{category}] Build or describe a project that demonstrates: {skills[0]}."
        )
    recs.append("Mirror the job description's wording where it is truthful (e.g. 'machine learning', not only 'ML').")
    return recs
