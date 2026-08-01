import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MASTER_SKILLS, EDUCATION_SCORES
from src.utils import clean_text

try:
    from sentence_transformers import SentenceTransformer
    _TRANSFORMER_MODEL = None
except ImportError:
    _TRANSFORMER_MODEL = None


def get_sentence_transformer():
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        try:
            _TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
    return _TRANSFORMER_MODEL


def extract_skills(text: str) -> set:
    cleaned = clean_text(text)
    found_skills = set()
    for skill in MASTER_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, cleaned):
            found_skills.add(skill)
    return found_skills


def compute_skill_match_score(resume_text: str, jd_text: str) -> tuple:
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if not jd_skills:
        score = min(len(resume_skills) / 10.0, 1.0)
        return score, resume_skills, set()

    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)

    score = len(matched) / len(jd_skills)
    return round(score, 3), matched, missing


def extract_experience_years(text: str) -> float:
    cleaned = clean_text(text)

    # Pattern 1: "X years" or "X+ years"
    direct_matches = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", cleaned)
    if direct_matches:
        years = [int(y) for y in direct_matches]
        return float(max(years))

    # Pattern 2: Date ranges like 2018-2023 or 2018 to present
    from datetime import datetime
    current_year = datetime.now().year
    
    range_matches = re.findall(
        r"\b((?:19|20)\d{2})\s*(?:-|to|–|and)\s*((?:19|20)\d{2}|present|current)\b", 
        cleaned
    )
    if range_matches:
        total_years = 0.0
        for start_str, end_str in range_matches:
            try:
                start_yr = int(start_str)
                if end_str in ["present", "current"]:
                    end_yr = current_year
                else:
                    end_yr = int(end_str)
                span = end_yr - start_yr
                if 0 <= span < 50:
                    total_years += span
            except ValueError:
                continue
        if total_years > 0:
            return min(float(total_years), 40.0)

    return 0.0


def compute_education_score(text: str) -> float:
    cleaned = clean_text(text)
    best_score = 0.2  # default (baseline)

    for keyword, score in EDUCATION_SCORES.items():
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, cleaned):
            best_score = max(best_score, score)

    return best_score


def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    model = get_sentence_transformer()
    if model is not None:
        try:
            embeddings = model.encode([resume_clean, jd_clean], convert_to_numpy=True)
            norm_res = np.linalg.norm(embeddings[0])
            norm_jd = np.linalg.norm(embeddings[1])
            if norm_res > 0 and norm_jd > 0:
                similarity = np.dot(embeddings[0], embeddings[1]) / (norm_res * norm_jd)
                return round(float(similarity), 3)
        except Exception:
            pass

    # TF-IDF Cosine Similarity Fallback
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 3)
    except ValueError:
        return 0.0


def compute_resume_length_norm(text: str) -> float:
    if not text.strip():
        return 0.0
    word_count = len(text.split())
    return round(min(word_count / 600.0, 1.0), 3)


def compute_certifications_score(text: str) -> float:
    cleaned = clean_text(text)
    cert_keywords = [
        "certified", "certification", "certificate", "aws certified",
        "pmp", "scrum master", "cissp", "comptia", "google cloud certified",
        "microsoft certified", "cisco certified",
    ]
    hits = sum(1 for keyword in cert_keywords if keyword in cleaned)
    return round(min(hits / 4.0, 1.0), 3)


def compute_projects_score(text: str) -> float:
    cleaned = clean_text(text)
    project_patterns = [
        r"\bprojects?\b",
        r"\bproject:\b",
        r"\bportfolio\b",
        r"\bpersonal project\b",
        r"\bcapstone\b",
    ]
    hits = sum(1 for pattern in project_patterns if re.search(pattern, cleaned))
    return round(min(hits / 3.0, 1.0), 3)


def compute_matched_skills_count_norm(matched_skills: set) -> float:
    return round(min(len(matched_skills) / 10.0, 1.0), 3)


def compute_missing_skills_count_norm(missing_skills: set) -> float:
    return round(min(len(missing_skills) / 10.0, 1.0), 3)


def compute_resume_embedding_norm(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned.strip():
        return 0.0

    model = get_sentence_transformer()
    if model is not None:
        try:
            embedding = model.encode([cleaned], convert_to_numpy=True)[0]
            norm = float(np.linalg.norm(embedding))
            return round(min(norm / 20.0, 1.0), 3)
        except Exception:
            pass

    return round(min(len(cleaned.split()) / 600.0, 1.0), 3)


def compute_resume_quality_score(text: str) -> float:
    if not text.strip():
        return 0.0

    word_count = len(text.split())
    length_score = min(word_count / 400.0, 1.0)  # ~400 words

    sections = ["experience", "education", "skills", "summary", "objective", "project"]
    cleaned = clean_text(text)
    section_hits = sum(1 for s in sections if s in cleaned)
    section_score = section_hits / len(sections)

    quality_score = (length_score * 0.5) + (section_score * 0.5)
    return round(quality_score, 3)
