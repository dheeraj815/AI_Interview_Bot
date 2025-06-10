import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample Interview Q&A
questions = {
    "Tell me about yourself.": "I am a computer science student with a strong passion for AI and hands-on experience in building real-world projects.",
    "What are your strengths?": "I am a quick learner, a team player, and I enjoy solving complex problems using technology.",
    "Why should we hire you?": "I bring a combination of technical skills, enthusiasm, and a desire to grow that aligns with your company's goals.",
    "Where do you see yourself in 5 years?": "In a leadership position, contributing to cutting-edge projects and mentoring others.",
    "Describe a challenging situation and how you handled it.": "In my final year project, I faced data inconsistency issues. I solved it by implementing robust preprocessing and version control."
}

# Streamlit app config
st.set_page_config(page_title="AI Interview Bot", page_icon="🧠")
st.title("🤖 AI Interview Practice Bot")
st.markdown(
    "Type your answer below to get instant AI feedback based on semantic similarity with an ideal answer.")

# Loop through each question
for question, ideal_answer in questions.items():
    st.subheader(f"🗣️ {question}")
    user_input = st.text_area("Your Answer", key=question)

    if st.button(f"Evaluate", key="eval_" + question):
        if not user_input.strip():
            st.warning("❗ Please enter your answer first.")
        else:
            # Encode both answers
            emb_user = model.encode(user_input, convert_to_tensor=True)
            emb_ideal = model.encode(ideal_answer, convert_to_tensor=True)

            # Calculate similarity
            similarity = util.pytorch_cos_sim(emb_user, emb_ideal).item()
            score = round(similarity * 100, 2)

            # Display score
            st.markdown(f"**🧠 Similarity Score: {score}%**")

            # Feedback
            if score > 80:
                st.success(
                    "✅ Excellent! Your answer closely matches an ideal response.")
            elif score > 60:
                st.info("🟡 Good, but could be more aligned.")
            else:
                st.error(
                    "❌ Needs improvement. Try making it more relevant and structured.")
