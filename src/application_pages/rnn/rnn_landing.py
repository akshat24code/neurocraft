import streamlit as st

from src.application_pages.rnn.next_word import next_word_page
from src.application_pages.rnn.rnn_sentiment import rnn_sentiment_page


def rnn_application_page():
    st.title("RNN Applications")
    st.info(
        "Choose an RNN application to explore. Each tool demonstrates a different "
        "real-world use case of vanilla recurrent neural networks."
    )

    selected_app = st.radio(
        "Select RNN Application",
        ("Sentiment Analyzer", "Next Word Predictor"),
        horizontal=True,
        key="rnn_app_selector",
    )

    st.divider()

    if selected_app == "Sentiment Analyzer":
        rnn_sentiment_page("RNN")
    else:
        next_word_page("RNN")


def lstm_application_page():
    st.title("LSTM Applications")
    st.info(
        "Choose an LSTM application to explore. Each tool demonstrates a different "
        "real-world use case of long short-term memory models."
    )

    selected_app = st.radio(
        "Select LSTM Application",
        ("Sentiment Analyzer", "Next Word Predictor"),
        horizontal=True,
        key="lstm_app_selector",
    )

    st.divider()

    if selected_app == "Sentiment Analyzer":
        rnn_sentiment_page("LSTM")
    else:
        next_word_page("LSTM")
