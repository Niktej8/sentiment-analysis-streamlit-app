import streamlit as st
from transformers import pipeline

# Load the model once
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment = load_model()

# App UI
st.title("🧠 Sentiment Analysis App")
st.markdown("""
Welcome to the **Sentiment Analysis App**!  
Type any sentence below to instantly find out whether it carries a **Positive** or **Negative** emotion.  
This app uses the fine-tuned **DistilBERT** model — `distilbert-base-uncased-finetuned-sst-2-english` — for accurate sentiment predictions.
""")

text = st.text_area("✍️ Enter your text here:")

if st.button("Analyze Sentiment"):
    if text.strip():
        result = sentiment(text)[0]
        label = result['label']
        score = result['score']
        # Color-coded output
        if label == "POSITIVE":
            st.markdown(
                f"<h3 style='color:green;'>😊 Sentiment: {label}</h3>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='color:red;'>😞 Sentiment: {label}</h3>", 
                unsafe_allow_html=True
            )

        st.write(f"**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter some text before analyzing.")
