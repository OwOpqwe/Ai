import streamlit as st
import requests

st.set_page_config(
    page_title="Alex AI",
    page_icon="🤖"
)

st.title("🤖 Alex AI")

st.write(
    "Welcome! Ask my personally trained AI "
    "a question about Alex."
)

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask AI"):

    if question.strip():

        try:

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question},
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            st.success(result["answer"])

            st.caption(
                "Predicted category: "
                + result["predicted_intent"]
            )

            st.caption(
                "Confidence: "
                + str(result["confidence"])
            )

        except requests.RequestException as error:

            st.error(
                "Could not connect to AI API: "
                + str(error)
            )

    else:

        st.warning("Please enter a question.")
