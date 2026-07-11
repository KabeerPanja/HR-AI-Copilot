import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import MASTER_SKILLS, EDUCATION_SCORES
from src.utils import clean_text


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
    year_ranges = re.findall(r"(19|20)\d{2}\s*(?:-|to|–)\s*(?:(19|20)\d{2}|present|current)", cleaned)
    all_years = re.findall(r"\b(19|20)\d{2}\b", cleaned)

    if len(all_years) >= 2:
        full_years = [int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", cleaned)]
        if full_years:
            span = max(full_years) - min(full_years)
            return float(max(span, 0))

    return 0.0


def compute_education_score(text: str) -> float:
    cleaned = clean_text(text)
    best_score = 0.2  # default (baseline)

    for keyword, score in EDUCATION_SCORES.items():
        if keyword in cleaned:
            best_score = max(best_score, score)

    return best_score


def compute_text_similarity(resume_text: str, jd_text: str) -> float:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 3)
    except ValueError:
        return 0.0


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
