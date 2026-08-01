"""
llm_service.py
---------------
IMPORTANT (handbook compliance): Is project mein LLM sirf REASONING,
SUMMARIZATION, EXPLANATION aur REPORT TEXT generate karne ke liye use
hota hai. Candidate ka "fit score" predict karne ka kaam sirf ANN
(models/ann_model.py) karta hai. LLM kabhi bhi primary predictive
engine ke taur par use nahi hota.
"""

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL

_client = None


def _get_client():
    """OpenAI/OpenRouter client ko lazy initialize karta hai."""
    global _client

    if _client is None and OPENAI_API_KEY:
        if OPENAI_BASE_URL:
            _client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )
        else:
            _client = OpenAI(api_key=OPENAI_API_KEY)

    return _client


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    """Generic LLM call wrapper with error handling."""

    client = _get_client()

    if client is None:
        return (
            "[LLM not configured] Please configure OPENAI_API_KEY in "
            "Streamlit Secrets or your .env file."
        )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        return f"[LLM Error] {exc}"


def summarize_resume(resume_text: str) -> str:
    """Generate a concise professional resume summary."""

    system_prompt = (
        "You are an expert HR analyst. Summarize the candidate's resume in "
        "3-4 concise bullet points covering their core expertise, experience "
        "level, and standout strengths. Be factual and professional. "
        "Do not invent information."
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
    """
    Converts SHAP feature contributions into a natural-language explanation.
    """

    system_prompt = (
        "You are an HR AI Co-Pilot assistant. Explain in professional English "
        "why the AI model gave this recommendation. Mention the strongest "
        "positive and negative factors, matched skills, missing skills, and "
        "keep the explanation to 4-5 sentences."
    )

    contributions_text = "\n".join(
        f"- {c['feature']}: value={c['value']}, impact={c['shap_contribution']}"
        for c in top_contributions[:5]
    )

    user_prompt = (
        f"Candidate: {candidate_name}\n"
        f"AI Fit Probability: {fit_probability:.2%}\n"
        f"Confidence Score: {confidence}%\n\n"
        f"Top SHAP Contributions:\n{contributions_text}\n\n"
        f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
        f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}"
    )

    return _call_llm(system_prompt, user_prompt, max_tokens=250)


def generate_report_narrative(summary_stats: dict) -> str:
    """Generate executive summary for screening report."""

    system_prompt = (
        "You are an HR analytics assistant. Write a concise executive summary "
        "for a hiring manager in 5-6 professional sentences."
    )

    user_prompt = (
        f"Total candidates screened: {summary_stats.get('total')}\n"
        f"AI Recommended: {summary_stats.get('recommended')}\n"
        f"AI Not Recommended: {summary_stats.get('not_recommended')}\n"
        f"HR Approved: {summary_stats.get('approved')}\n"
        f"HR Rejected: {summary_stats.get('rejected')}\n"
        f"HR Modified: {summary_stats.get('modified')}\n"
        f"Average AI Confidence: {summary_stats.get('avg_confidence')}%"
    )

    return _call_llm(system_prompt, user_prompt, max_tokens=350)
