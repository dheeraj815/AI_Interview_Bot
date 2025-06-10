# 🤖 AI Interview Practice Bot

## 📌 Description
This project is an **AI-powered Interview Practice Bot** that simulates real HR-style questions and evaluates your answers using **semantic similarity** techniques. It helps users prepare for common interview questions with instant feedback and scoring.

## 🎯 Features
- Asks real interview questions (e.g., "Tell me about yourself", "Why should we hire you?")
- User types in answers directly in the UI
- Compares your answer with an ideal response using **NLP-based semantic similarity**
- Displays a **percentage score** and AI feedback

## 🛠️ Tech Stack
- 🐍 Python
- Streamlit (for frontend)
- Sentence Transformers (Hugging Face)
- scikit-learn
- NLP (semantic similarity)

## 🚀 How to Run

1. **Install Dependencies**
```bash
pip install -r requirements.txt
streamlit run interview_bot.py
