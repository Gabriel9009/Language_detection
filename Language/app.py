import streamlit as st
import joblib
import re
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "language_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Language Detector",
    page_icon="🌍",
    layout="centered"
)


# Language labels
# ---------------------------------------------------
language_names = {
    "en": "🇬🇧 English",
    "es": "🇪🇸 Spanish",
    "sw": "🇰🇪 Swahili",
    "zh": "🇨🇳 Mandarin"
}


# Clean text function
# ---------------------------------------------------
def clean_text(text):
    text = text.strip()

    if not any('\u4e00' <= c <= '\u9fff' for c in text):
        text = text.lower()

    text = re.sub(r'\s+', ' ', text)

    return text


# Predict function
# ---------------------------------------------------
def predict_language(text):

    cleaned = clean_text(text)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    probabilities = model.predict_proba(vectorized)[0]

    confidence_scores = {
        language_names[label]: round(prob * 100, 2)
        for label, prob in zip(model.classes_, probabilities)
    }

    return prediction, confidence_scores


# UI
# ---------------------------------------------------
st.title("🌍 Language Detection App")
st.write("Detect whether text is English, Spanish, Swahili, or Mandarin.")


user_input = st.text_area(
    "Enter some text",
    height=150,
    placeholder="Type something here..."
)


if st.button("Detect Language"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        prediction, scores = predict_language(user_input)

        st.success(f"Predicted Language: {language_names[prediction]}")

        st.subheader("Confidence Scores")

        for lang, score in scores.items():
            st.progress(score / 100)
            st.write(f"{lang}: {score}%")
