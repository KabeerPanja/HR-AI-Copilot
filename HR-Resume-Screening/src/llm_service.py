from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, GROQ_API_KEY, GROQ_MODEL

_client = None
_groq_client = None


def _get_client():
    global _client
    if _client is None and OPENAI_API_KEY:
        if OPENAI_BASE_URL.strip():
            _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL.strip())
        else:
            _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _groq_client


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    client = _get_client()

    if client is not None:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            groq_client = _get_groq_client()
            if groq_client is not None:
                try:
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.4,
                    )
                    return response.choices[0].message.content.strip()
                except Exception as g_exc:
                    return f"[LLM Error] Primary failed ({exc}), Groq fallback failed ({g_exc})"
            return f"[LLM Error] Response generate nahi ho saka: {exc}"

    groq_client = _get_groq_client()
    if groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as g_exc:
            return f"[LLM Error] Groq failed: {g_exc}"

    return "[LLM not configured]"


def summarize_resume(resume_text: str) -> str:
    """Resume ka concise professional summary generate karta hai (LLM reasoning task)."""
    system_prompt = (
        "You are an expert HR analyst. Summarize the candidate's resume in 3-4 concise "
        "bullet points covering their core expertise, experience level, and standout strengths. "
        "Be factual and professional. Do not invent information not present in the resume."
    )
    user_prompt = f"Resume text:\n\n{resume_text[:4000]}"
    return _call_llm(system_prompt, user_prompt, max_tokens=300)


def generate_explanation_narrative(
    candidate_name: str,
    fit_probability: float,
    confidence: float,
    top_contributions: list,
    matched_skills: list,
    missing_skills: list,
) -> str:
    system_prompt = (
        "You are an HR AI Co-Pilot assistant. Explain, in plain professional English, "
        "why the AI model gave this recommendation for a candidate. Reference the top "
        "contributing factors provided. Keep it to 4-5 sentences. Be balanced and objective, "
        "mentioning both strengths and gaps if present."
    )

    contributions_text = "\n".join(
        f"- {c['feature']}: value={c['value']}, impact={c['shap_contribution']}"
        for c in top_contributions[:5]
    )

    user_prompt = (
        f"Candidate: {candidate_name}\n"
        f"AI Fit Probability: {fit_probability:.2%}\n"
        f"Confidence Score: {confidence}%\n"
        f"Top contributing factors (SHAP values):\n{contributions_text}\n"
        f"Matched skills: {', '.join(matched_skills) if matched_skills else 'None found'}\n"
        f"Missing required skills: {', '.join(missing_skills) if missing_skills else 'None'}\n\n"
        "Write a short, clear explanation for the HR reviewer."
    )

    return _call_llm(system_prompt, user_prompt, max_tokens=250)


def generate_report_narrative(summary_stats: dict) -> str:
    """Poori batch screening ke liye ek executive summary narrative generate karta hai."""
    system_prompt = (
        "You are an HR analytics assistant writing an executive summary for a hiring "
        "manager. Be concise (5-6 sentences), professional, and data-driven."
    )
    user_prompt = (
        f"Total candidates screened: {summary_stats.get('total')}\n"
        f"AI Recommended: {summary_stats.get('recommended')}\n"
        f"AI Not Recommended: {summary_stats.get('not_recommended')}\n"
        f"HR Approved: {summary_stats.get('approved')}\n"
        f"HR Rejected: {summary_stats.get('rejected')}\n"
        f"HR Modified: {summary_stats.get('modified')}\n"
        f"Average AI confidence: {summary_stats.get('avg_confidence')}%\n\n"
        "Write an executive summary for this hiring batch."
    )
    return _call_llm(system_prompt, user_prompt, max_tokens=350)


def analyze_candidate_resume(resume_text: str, jd_text: str) -> dict:
    """
    LLM ko call kar ke resume ka complete analysis (summary, strengths, weaknesses,
    missing skills description, aur interview questions) extract karta hai.
    Saves API calls by getting all of it in a single structured JSON response.
    """
    import json
    
    system_prompt = (
        "You are an expert HR analyst. Analyze the candidate's resume relative to the provided Job Description.\n"
        "You must respond with a valid JSON object containing exactly the following keys:\n"
        "- 'summary': A concise 3-4 bullet point summary (using markdown bullets) of the candidate's background.\n"
        "- 'strengths': 3 key strengths of the candidate as bullet points (using markdown bullets).\n"
        "- 'weaknesses': 2-3 weaknesses or areas of improvement as bullet points (using markdown bullets).\n"
        "- 'missing_skills_explanation': A brief narrative explanation (2-3 sentences) of why some required skills are missing and their importance.\n"
        "- 'interview_questions': 3-4 tailored technical or situational interview questions for this candidate.\n\n"
        "Keep the content highly professional, factual, and based only on the provided text. Return ONLY raw valid JSON."
    )
    
    user_prompt = (
        f"Job Description:\n{jd_text[:2000]}\n\n"
        f"Candidate Resume:\n{resume_text[:4000]}"
    )
    
    client = _get_client()
    groq_client = _get_groq_client()
    
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.3,
            )
            result_text = response.choices[0].message.content.strip()
            return json.loads(result_text)
        except Exception as exc:
            if groq_client is not None:
                try:
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=600,
                        temperature=0.3,
                    )
                    result_text = response.choices[0].message.content.strip()
                    return json.loads(result_text)
                except Exception:
                    pass
            try:
                summary = summarize_resume(resume_text)
                return {
                    "summary": summary,
                    "strengths": "- Relevant professional background\n- Core technical competencies aligned with field",
                    "weaknesses": "- Missing some specific skill matching criteria from Job Description",
                    "missing_skills_explanation": "The candidate has gaps in some tools/technologies mentioned in the Job Description.",
                    "interview_questions": "1. Can you describe a challenging project you worked on recently?\n2. How do you adapt to learning new tech stacks quickly?",
                }
            except Exception:
                return {
                    "summary": f"[Error generating summary: {exc}]",
                    "strengths": "[Error generating strengths]",
                    "weaknesses": "[Error generating weaknesses]",
                    "missing_skills_explanation": "[Error generating missing skills explanation]",
                    "interview_questions": "[Error generating interview questions]",
                }
                
    if groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.3,
            )
            result_text = response.choices[0].message.content.strip()
            return json.loads(result_text)
        except Exception:
            pass
            
    try:
        summary = summarize_resume(resume_text)
        return {
            "summary": summary,
            "strengths": "- Relevant professional background\n- Core technical competencies aligned with field",
            "weaknesses": "- Missing some specific skill matching criteria from Job Description",
            "missing_skills_explanation": "The candidate has gaps in some tools/technologies mentioned in the Job Description.",
            "interview_questions": "1. Can you describe a challenging project you worked on recently?\n2. How do you adapt to learning new tech stacks quickly?",
        }
    except Exception as exc:
        return {
            "summary": f"[Error generating summary: {exc}]",
            "strengths": "[Error generating strengths]",
            "weaknesses": "[Error generating weaknesses]",
            "missing_skills_explanation": "[Error generating missing skills explanation]",
            "interview_questions": "[Error generating interview questions]",
        }
