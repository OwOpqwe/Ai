
import streamlit as st
import requests

API_URL = "https://charlies-ai.onrender.com/ask"

st.set_page_config(
    page_title="Personal AI Engine",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #10131b;
    color: #ffffff;
}
header {visibility: hidden;}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label {
    color: #ffffff;
}

[data-testid="stChatMessage"] {
    background: #1b2532;
    border: 1px solid #303b4b;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border-radius: 18px;
}

.status {
    color: #55d69e;
    font-size: 13px;
    font-weight: bold;
}
.subtitle {
    color: #9ba9bb;
}
.footer {
    text-align: center;
    color: #788699;
    font-size: 12px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🤖 Personal AI Engine")
st.markdown(
    '<p class="subtitle">Your personal neural network assistant.</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="status">● AI ENGINE ONLINE</p>',
    unsafe_allow_html=True
)

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.markdown("### How can I help you?")
    st.caption("Ask questions about the fictional personal profile.")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
question = st.chat_input("Message your AI...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60
                )

                response.raise_for_status()
                answer = response.json()["answer"]

            except requests.exceptions.RequestException:
                answer = "I couldn't connect to my AI engine. Please try again."

            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

st.markdown(
    '<div class="footer">Powered by a custom-trained PyTorch neural network</div>',
    unsafe_allow_html=True
)
