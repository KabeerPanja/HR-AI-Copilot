import uuid
import re
from datetime import datetime


def generate_id(prefix: str = "CAND") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,+#-]", " ", text)
    return text.strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)
