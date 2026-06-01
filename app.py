import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0f1117;
}

h1,h2,h3,h4,h5,h6,p,label {
    color: white !important;
}

.stTextArea textarea {
    background-color: #1e1e1e !important;
    color: white !important;
}

.stButton > button {
    width: 100%;
    background-color: #00d4ff;
    color: black;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODELS
# ------------------------------

@st.cache_resource
def load_models():

    rnn = tf.keras.models.load_model("rnn_model.keras")
    lstm = tf.keras.models.load_model("lstm_model.keras")
    gru = tf.keras.models.load_model("gru_model.keras")

    return rnn, lstm, gru

rnn_model, lstm_model, gru_model = load_models()

# ------------------------------
# IMDB WORD INDEX
# ------------------------------

word_index = imdb.get_word_index()

max_length = 200

# ------------------------------
# TEXT TO SEQUENCE
# ------------------------------

def text_to_sequence(text):

    words = text.lower().split()

    sequence = []

    for word in words:
        sequence.append(
            word_index.get(word, 2)
        )

    return sequence

# ------------------------------
# PREDICTION
# ------------------------------

def predict_sentiment(model, text):

    seq = text_to_sequence(text)

    padded = pad_sequences(
        [seq],
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    score = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if score >= 0.5
        else "Negative"
    )

    return sentiment, float(score)

# ------------------------------
# HEADER
# ------------------------------

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

# ------------------------------
# SIDEBAR
# ------------------------------

selected_model = st.sidebar.radio(
    "Choose Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# ------------------------------
# INPUT
# ------------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=180
)

# ------------------------------
# PREDICT
# ------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        if selected_model == "SimpleRNN":
            sentiment, score = predict_sentiment(
                rnn_model,
                review
            )

        elif selected_model == "LSTM":
            sentiment, score = predict_sentiment(
                lstm_model,
                review
            )

        else:
            sentiment, score = predict_sentiment(
                gru_model,
                review
            )

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.write(
            f"Confidence: {score*100:.2f}%"
        )

        st.progress(float(score))

        positive = score * 100
        negative = (1-score) * 100

        chart_df = pd.DataFrame(
            {
                "Probability":[
                    positive,
                    negative
                ]
            },
            index=[
                "Positive",
                "Negative"
            ]
        )

        st.bar_chart(chart_df)

        st.subheader("Compare All Models")

        rnn_sent, rnn_score = predict_sentiment(
            rnn_model,
            review
        )

        lstm_sent, lstm_score = predict_sentiment(
            lstm_model,
            review
        )

        gru_sent, gru_score = predict_sentiment(
            gru_model,
            review
        )

        comparison = pd.DataFrame({

            "Model":[
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],

            "Sentiment":[
                rnn_sent,
                lstm_sent,
                gru_sent
            ],

            "Confidence":[
                round(rnn_score*100,2),
                round(lstm_score*100,2),
                round(gru_score*100,2)
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True
        )