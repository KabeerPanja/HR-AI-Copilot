# AI Co-Pilot for HR Resume Screening

A complete, hackathon-ready **AI Co-Pilot** built for the "AI Innovation Hackathon" theme
*"Build an AI Co-Pilot for Industry"* — solving the **HR Resume Screening** challenge.

## What This Project Does

Upload or paste a Job Description and multiple candidate resumes (PDF). The AI Co-Pilot will:

1. Parse resumes (PDF; DOCX/TXT supported by the parser module).
2. Extract skills, experience, education, certifications, projects, and embedding-based similarity.
3. Predict a **job-fit probability** using a custom **PyTorch ANN** (the primary predictive engine).
4. Explain every prediction using **SHAP** (Explainable AI) + a confidence score.
5. Use an **LLM (OpenRouter/OpenAI-compatible API)** only for summarization, natural-language explanation, and report writing.
6. Let an HR reviewer **Approve / Reject / Hold** each AI recommendation (Human-in-the-Loop).
7. Generate a **downloadable PDF report** with an executive summary.

## Data Modalities Used (Handbook Requirement: 3+)

| # | Modality | How it's used |
|---|----------|----------------|
| 1 | PDF documents | Resume and JD files parsed with `pdfplumber` |
| 2 | Text (NLP) | Skill extraction, experience/education parsing, Sentence Transformers similarity |
| 3 | Tabular data | 11 engineered numeric features fed into the ANN model |

## Project Structure

```
HR-Resume-Screening/
├── app.py                     # Main Streamlit app (entry point)
├── config.py                  # Central configuration & constants
├── requirements.txt
├── .env.example
├── .env                       # Paste your API key here (optional for LLM features)
├── data/
│   ├── sample_job_description.txt
│   ├── sample_resumes/        # Sample resumes for quick testing
│   └── training_data.csv      # Auto-generated synthetic training data
├── models/
│   ├── ann_model.py           # PyTorch ANN architecture
│   ├── train_model.py         # Training script (auto-runs if no saved model found)
│   └── saved_model/           # Saved model weights + scaler (auto-created)
├── src/
│   ├── resume_parser.py       # PDF/DOCX/TXT parsing
│   ├── text_processor.py      # NLP: skills, experience, education, similarity
│   ├── feature_engineering.py # Tabular feature vector builder
│   ├── prediction_service.py  # Loads/auto-trains ANN, exposes predict()
│   ├── explainability.py      # SHAP explanations + confidence score
│   ├── llm_service.py         # LLM calls (summary/explanation/report)
│   ├── database.py            # SQLite storage for candidates + HITL decisions
│   ├── report_generator.py    # PDF report generation (reportlab)
│   └── utils.py               # Helper functions
├── database/
│   └── hr_copilot.db          # Auto-created SQLite database
└── outputs/
    └── reports/               # Optional local copies of generated reports
```

## Setup Instructions

### 1. Create a virtual environment (Python 3.11 recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key (optional)

Copy `.env.example` to `.env` and paste your API key:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

> The app will still run without a key — the LLM-based text sections will show a
> placeholder message instead of AI-generated text. The ANN prediction and SHAP
> explainability work fully offline without any API key.

### 4. Run the app

```bash
streamlit run app.py
```

The very first time you run the app, it will automatically generate a synthetic
training dataset and train the ANN model (takes a few seconds). This is expected —
you do not need to run anything manually.

### 5. (Optional) Re-train the model manually

```bash
python models/train_model.py
```

## Quick Test

1. Go to **"1. Upload & Screen"**.
2. Paste the contents of `data/sample_job_description.txt` into the Job Description box.
3. Upload PDF resumes, or convert the sample `.txt` files in `data/sample_resumes/` to PDF for upload.
4. Click **Run AI Screening**.
5. Review the **ranked results table** on the same page.
6. Go to **"2. Human-in-the-Loop Review"** to see AI recommendations, SHAP explanations,
   and Approve/Reject/Hold decisions.
7. Go to **"4. Generate Report"** to build and download the PDF report.

## ANN Feature Inputs

The ANN uses 11 engineered numeric features:

- Skill match percentage
- Experience (normalized)
- Education score
- Resume-JD text similarity
- Resume length
- Certifications score
- Projects score
- Resume quality
- Matched skills count
- Missing skills count
- Resume embedding strength

## Notes on the ANN Model

Since no labeled "hired vs not-hired" dataset is publicly required by the handbook, the
ANN is trained on a **rule-based synthetic dataset** (`models/train_model.py`) that mimics
realistic scoring patterns with added noise. This keeps the deep learning model as the genuine
predictive engine (not the LLM), exactly as required by the hackathon rules.

## Evaluation Rubric Mapping

| Rubric Item | Where it's addressed |
|---|---|
| Multimodal Integration | PDF + Text NLP + Tabular features |
| Deep Learning Implementation | `models/ann_model.py`, `models/train_model.py` |
| Human-in-the-Loop | "2. Human-in-the-Loop Review" page (Approve/Reject/Hold) |
| Explainable AI | `src/explainability.py` (SHAP + confidence) |
| System Architecture | Modular `src/` + `models/` structure |
| User Experience | Streamlit multi-page app with ranked results |
| Presentation | PDF report generation |
