# Goal-Based Agent: Job Application Assistant
import os
import re
from typing import Dict, Optional

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="🎯 Job Application Assistant",
    page_icon="🧠",
    layout="centered",
)

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

REQUIRED_FIELDS = ("name", "email", "skills")


# ----------------------------
# Session state
# ----------------------------
def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "application_info" not in st.session_state:
        st.session_state.application_info = {
            "name": None,
            "email": None,
            "skills": None,
        }
    if "goal_complete" not in st.session_state:
        st.session_state.goal_complete = False
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""


init_state()


# ----------------------------
# Helpers
# ----------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def missing_fields(info: Dict[str, Optional[str]]) -> list[str]:
    return [key for key in REQUIRED_FIELDS if not info.get(key)]


def merge_info(new_info: Dict[str, Optional[str]]) -> None:
    """Update only the fields that we successfully extracted."""
    for key, value in new_info.items():
        if value and not st.session_state.application_info.get(key):
            st.session_state.application_info[key] = value


def extract_application_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract name, email, and skills from plain text.
    Works for chat messages and simple resume text.
    """
    info = {"name": None, "email": None, "skills": None}
    if not text:
        return info

    # Name patterns
    name_patterns = [
        r"(?:my name is|i am|i'm|i am called|name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"(?:name[:\-]\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"(?:full name[:\-]\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info["name"] = normalize_whitespace(match.group(1)).title()
            break

    # Email
    email_match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    if email_match:
        info["email"] = email_match.group(0)

    # Skills from chat text
    skills_match = re.search(
        r"(?:skills are|i know|i can use|skills|technical skills)[:\-]?\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if skills_match:
        skills = skills_match.group(1).strip()
        skills = skills.split("\n")[0]
        info["skills"] = normalize_whitespace(skills).rstrip(".")

    return info


def extract_text_from_pdf(uploaded_file) -> str:
    """Read all text from uploaded PDF."""
    file_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    finally:
        doc.close()


def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def extract_info_from_resume_text(text: str) -> Dict[str, Optional[str]]:
    """
    Slightly smarter resume parsing.
    First tries generic extraction, then adds simple resume-specific heuristics.
    """
    info = extract_application_info(text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # If name is missing, sometimes the first line of a resume is the name
    if not info["name"] and lines:
        first_line = lines[0]
        if 2 <= len(first_line.split()) <= 4 and re.match(r"^[A-Za-z][A-Za-z\s.'-]+$", first_line):
            info["name"] = normalize_whitespace(first_line).title()

    # Resume skills section
    skills_section = re.search(
        r"(?:skills|technical skills)[:\-\s]*(.*?)(?:\n\s*(?:projects|experience|education|certifications|summary|objective)\b|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if skills_section and not info["skills"]:
        skills = skills_section.group(1)
        skills = skills.replace("\u2022", ", ").replace("•", ", ").replace("\n", ", ")
        skills = re.sub(r"[-]+", ", ", skills)
        skills = normalize_whitespace(skills).strip(", ")
        if skills:
            info["skills"] = skills

    return info


def build_summary(info: Dict[str, Optional[str]]) -> str:
    return (
        f"✅ Name: {info['name']}\n"
        f"📧 Email: {info['email']}\n"
        f"🛠️ Skills: {info['skills']}\n"
    )


def get_llm():
    """Gemini chat model. Uses GOOGLE_API_KEY from .env or environment."""
    try:
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0,
        )
    except Exception:
        return None


def generate_ai_reply(user_text: str) -> str:
    info = st.session_state.application_info
    missing = missing_fields(info)

    if not missing:
        return (
            f"✅ Great! I have your name, email, and skills. "
            f"Your application info is complete."
        )

    llm = get_llm()
    fallback = f"⏳ I still need: {', '.join(missing)}."

    if llm is None:
        return fallback

    system_msg = SystemMessage(
        content=(
            "You are a concise job application assistant. "
            "Your only goal is to collect the user's name, email, and skills. "
            "Ask only for the missing fields. Keep the reply short and friendly."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"User message: {user_text}\n"
            f"Known info: {info}\n"
            f"Missing fields: {missing}\n\n"
            "Write the next assistant message."
        )
    )

    try:
        response = llm.invoke([system_msg, human_msg])
        return (response.content or fallback).strip()
    except Exception:
        return fallback


def rerun_app() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ----------------------------
# UI
# ----------------------------
st.title("🧠 Goal-Based Agent \n Job Application Assistant")
st.markdown(
    "Tell me your **name**, **email**, and **skills**. "
    "You can also upload a resume in PDF or TXT."
    "\n The app will extract info from your resume and help you complete your application."
)

if not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.warning("GOOGLE_API_KEY not found. The app will still run, but Gemini replies may fall back.")

# Sidebar
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])

if resume is not None:
    try:
        if resume.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(resume)
        else:
            text = extract_text_from_txt(resume)

        st.session_state.resume_text = text
        resume_info = extract_info_from_resume_text(text)
        merge_info(resume_info)

        st.sidebar.success("Resume processed successfully.")
        st.sidebar.subheader("Extracted Info")
        st.sidebar.write(resume_info)

    except Exception as e:
        st.sidebar.error(f"Could not read resume: {e}")

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.application_info = {"name": None, "email": None, "skills": None}
    st.session_state.goal_complete = False
    st.session_state.summary = ""
    st.session_state.resume_text = ""
    rerun_app()

# Current status
with st.sidebar:
    st.subheader("Current Application Info")
    st.json(st.session_state.application_info)

    current_missing = missing_fields(st.session_state.application_info)
    if current_missing:
        st.warning(f"Still need: {', '.join(current_missing)}")
    else:
        st.success("All required info collected")

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your name, email, or skills...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Extract from user text and merge into state
    extracted = extract_application_info(user_input)
    merge_info(extracted)

    # Generate assistant reply
    bot_reply = generate_ai_reply(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    # Check goal completion
    if not missing_fields(st.session_state.application_info):
        st.session_state.goal_complete = True
        st.session_state.summary = build_summary(st.session_state.application_info)

# Final completion UI
if st.session_state.goal_complete:
    st.success("🎉 All information collected! You're ready to apply.")
    st.text_area("Application Summary", st.session_state.summary, height=120)

    st.download_button(
        label="📥 Download Application Summary",
        data=st.session_state.summary,
        file_name="application_summary.txt",
        mime="text/plain",
    )
