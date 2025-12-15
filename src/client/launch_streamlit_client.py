import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/movie-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agents designed to answer questions about movies.
        """
    )

    st.header("Example Questions")
    st.markdown("- What is the most recent movie?")
    st.markdown("- What is the average rating value for the movie Sounder?")
    st.markdown(
        """- What is the overall sentiment analysis regarding the acting of Robert de Niro?
        Keep it as one word""")
    st.markdown("- Which movies are not reviewed by Robert Brooks?")
    st.markdown("- What are the co-actors of Robert de Niro, grouped by movie?")



st.title("Movie System Chatbot")
st.info(
    """Ask me questions about movies, reviews, actors, directors or genres"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            This usually means the chatbot failed at generating a query to
            answer your question. Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
