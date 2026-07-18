# 🧠 Job Application Assistant
A simple **Goal-Based AI Agent** built with **Streamlit**, **LangChain**, and **Google Gemini** that helps collect job application details such as **name, email, and skills**, and can also extract information from an uploaded **resume (PDF/TXT)**.

---

## ✨ Features
- Collects **name**, **email**, and **skills** through chat
- Supports **resume upload** (`.pdf` and `.txt`)
- Extracts basic details from the resume
- Shows current application info in the sidebar
- Generates a **summary** of collected details
- Lets the user **download** the summary as a text file
- Includes a **Reset Chat** option

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit**
- **LangChain**
- **Google Gemini**
- **python-dotenv**
- **PyMuPDF (fitz)** for PDF text extraction

---

## 📁 Project Structure
```text
Job_Application_Assistant/
├── App.py
├── .env
├── requirements.txt
└── README.md
