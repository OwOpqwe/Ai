
import streamlit as st
import requests

st.set_page_config(
    page_title="My Personal AI",
    page_icon="🤖"
)

st.title("🤖 My Personal AI Assistant")

st.write("Ask me questions about my fictional personal profile!")

question = st.text_input("Enter your question:")

if st.button("Ask AI"):
    if question.strip():
        try:
            response = requests.post(
                "https://charlies-ai.onrender.com/ask",
                json={"question": question},
                timeout=10
            )

            response.raise_for_status()

            answer = response.json()["answer"]

            st.success(answer)

        except requests.RequestException as error:
            st.error(f"Could not connect to the AI API: {error}")
    else:
        st.warning("Please enter a question.")
