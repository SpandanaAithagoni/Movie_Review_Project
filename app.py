import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp{
    background-color:#0f1117;
}
h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}
.stTextArea textarea{
    background-color:#1e1e1e !important;
    color:white !important;
}
.stButton>button{
    width:100%;
    background-color:#00d4ff;
    color:black;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}

    try:
        models["SimpleRNN"] = tf.keras.models.load_model("rnn_model.keras")
    except:
        models["SimpleRNN"] = None

    try:
        models["LSTM"] = tf.keras.models.load_model("lstm_model.keras")
    except:
        models["LSTM"] = None

    try:
        models["GRU"] = tf.keras.models.load_model("gru_model.keras")
    except:
        models["GRU"] = None

    return models

models = load_models()

word_index = imdb.get_word_index()
max_length = 200

def text_to_sequence(text):
    words = text.lower().split()
    return [word_index.get(word, 2) for word in words]

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

st.title("🎬 Movie Review Sentiment Analysis")
st.subheader("Deep Learning Based Sentiment Classification")

st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

st.sidebar.subheader("Model Status")

for model_name, model_obj in models.items():
    if model_obj is not None:
        st.sidebar.success(f"{model_name} Loaded")
    else:
        st.sidebar.error(f"{model_name} Missing")

review = st.text_area(
    "Enter Movie Review",
    height=180
)

examples = [
    "This movie was amazing and I loved every minute of it.",
    "The film was boring and a complete waste of time.",
    "Excellent acting and fantastic storyline.",
    "Terrible screenplay and poor direction."
]

st.subheader("Sample Reviews")

for example in examples:
    if st.button(example):
        review = example

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        model = models[selected_model]

        if model is None:
            st.error(f"{selected_model} model file not found.")
        else:

            sentiment, score = predict_sentiment(
                model,
                review
            )

            st.success(
                f"Predicted Sentiment: {sentiment}"
            )

            st.metric(
                "Confidence",
                f"{score*100:.2f}%"
            )

            st.progress(float(score))

            positive = score * 100
            negative = (1 - score) * 100

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

            st.subheader("Prediction Probability")

            st.bar_chart(chart_df)

            st.subheader("All Model Comparison")

            comparison = []

            for model_name, model_obj in models.items():

                if model_obj is not None:

                    pred_sentiment, pred_score = predict_sentiment(
                        model_obj,
                        review
                    )

                    comparison.append(
                        {
                            "Model": model_name,
                            "Sentiment": pred_sentiment,
                            "Confidence": round(pred_score * 100, 2)
                        }
                    )

            st.dataframe(
                pd.DataFrame(comparison),
                use_container_width=True
            )
