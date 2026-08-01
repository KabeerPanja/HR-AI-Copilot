import numpy as np

from config import FEATURE_NAMES
from src.text_processor import (
    compute_skill_match_score,
    extract_experience_years,
    compute_education_score,
    compute_text_similarity,
    compute_resume_quality_score,
    compute_resume_length_norm,
    compute_certifications_score,
    compute_projects_score,
    compute_matched_skills_count_norm,
    compute_missing_skills_count_norm,
    compute_resume_embedding_norm,
)


def build_feature_vector(resume_text: str, jd_text: str) -> dict:
    skill_score, matched_skills, missing_skills = compute_skill_match_score(resume_text, jd_text)
    experience_years = extract_experience_years(resume_text)
    education_score = compute_education_score(resume_text)
    similarity_score = compute_text_similarity(resume_text, jd_text)
    quality_score = compute_resume_quality_score(resume_text)
    resume_length_norm = compute_resume_length_norm(resume_text)
    certifications_score = compute_certifications_score(resume_text)
    projects_score = compute_projects_score(resume_text)
    matched_skills_count_norm = compute_matched_skills_count_norm(matched_skills)
    missing_skills_count_norm = compute_missing_skills_count_norm(missing_skills)
    resume_embedding_norm = compute_resume_embedding_norm(resume_text)

    experience_norm = min(experience_years / 15.0, 1.0)

    features = {
        "skill_match_score": round(float(skill_score), 3),
        "experience_years_norm": round(float(experience_norm), 3),
        "education_score": round(float(education_score), 3),
        "text_similarity_score": round(float(similarity_score), 3),
        "resume_length_norm": round(float(resume_length_norm), 3),
        "certifications_score": round(float(certifications_score), 3),
        "projects_score": round(float(projects_score), 3),
        "resume_quality_score": round(float(quality_score), 3),
        "matched_skills_count_norm": round(float(matched_skills_count_norm), 3),
        "missing_skills_count_norm": round(float(missing_skills_count_norm), 3),
        "resume_embedding_norm": round(float(resume_embedding_norm), 3),
    }

    meta = {
        "experience_years_raw": experience_years,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
    }

    return {"features": features, "meta": meta}


def feature_dict_to_array(features: dict) -> np.ndarray:
    """Feature dictionary ko ANN model ke expected order mein numpy array banata hai."""
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
