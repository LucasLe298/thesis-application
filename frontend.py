import streamlit as st
import requests
from datetime import datetime 

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f7f9fb !important;
    color: #1c1c1e;
    font-size: 55px !important;
    padding: 0 !important;
    margin: 0 !important;
}
h1 {
    font-size: 100px !important;
    font-weight: 900 !important;
    color: #1c1c1e !important;
    margin: 0 !important;
    padding: 0 !important;
}
input, textarea {
    font-size: 80px !important;
}
button[kind="primary"] {
    font-size: 45px !important;
    padding: 10px 25px !important;
}
.chat-container {
    max-width: 1200px;
    margin:0;
    
}
.bubble-user {
    background: #0d7ad4;
    padding: 20px 26px;
    border-radius: 28px 10px 28px 28px;
    align-self: flex-end;
    max-width: 100%;
    margin: 24px 0 8px auto;
    font-size: 80px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
}
.bubble-bot {
    background: #ffffff;
    padding: 20px 26px;
    border-radius: 10px 28px 28px 28px;
    max-width: 100%;
    margin: 8px 0 24px 0;
    font-size: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
}
.emotion-tag {
    display: inline-block;
    background: #6c63ff;
    color: white;
    font-size: 28px;
    padding: 10px 18px;
    margin: 6px 6px 0 0;
    border-radius: 24px;
}
</style>
""", unsafe_allow_html=True)

# ===== INIT SESSION =====
if "chat" not in st.session_state:
    st.session_state.chat = []

# ===== TITLE =====
st.markdown("Emotion Recognition From Text")


# ===== INPUT FORM =====
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("User:", placeholder="Enter text here...", label_visibility="collapsed")
    submitted = st.form_submit_button("Gửi")

    if submitted and user_input.strip():
        try:
            res = requests.post(
                "http://localhost:8000/predict",
                json={"text": user_input, "model": "roberta"}
            )
            if res.status_code == 200:
                result = res.json()
                sorted_emo = sorted(result["all_probabilities"], key=lambda x: x["probability"], reverse=True)
                st.session_state.chat.append({
                    "text": user_input,
                    "predicted": sorted_emo
                })
            else:
                st.error("❌ Lỗi từ server.")
        except Exception as e:
            st.error("⚠️ Không kết nối được tới FastAPI.")

# ===== CHAT OUTPUT =====
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for c in st.session_state.chat:
    st.markdown(f'<div class="bubble-user">{c["text"]}</div>', unsafe_allow_html=True)
    bot_tags = ""
    for e in c["predicted"][:3]:  # Top 3 emotions
        prob = int(e["probability"] * 100)
        emoji = {
            "joy": "😂", "sadness": "😢", "anger": "😡",
            "disgust": "🤢", "fear": "😱", "surprise": "😮", "neutral": "😐"
        }.get(e["emotion"], "")
        bot_tags += f'<span class="emotion-tag">{emoji} {e["emotion"].capitalize()} ({prob}%)</span>'
    st.markdown(f'<div class="bubble-bot">{bot_tags}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
