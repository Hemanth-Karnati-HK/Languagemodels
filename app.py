# Import necessary libraries
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Define function to load models
@st.cache(allow_output_mutation=True)
def load_models():
    classification_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)

    summarization_model_name = 't5-base'
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

    return classification_model, classification_tokenizer, summarization_model, summarization_tokenizer

classification_model, classification_tokenizer, summarization_model, summarization_tokenizer = load_models()

# Title of the app
st.title('Text Classification and Summarization with Hugging Face')

# Take user input
text = st.text_area("Enter text:", "")
submit_button = st.button("Analyze Text")

# Predict function for sentiment analysis
def predict_sentiment(text):
    inputs = classification_tokenizer(text, return_tensors="pt")
    outputs = classification_model(**inputs)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    return probs.detach().numpy()

# Predict function for text summarization
def summarize_text(text):
    inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = summarization_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '')

if submit_button:
    if text:
        with st.spinner("Analyzing..."):
            # Sentiment analysis
            probs = predict_sentiment(text)
            st.markdown(f"**Positive sentiment:** `{probs[0][1]:.2f}`")
            st.markdown(f"**Negative sentiment:** `{probs[0][0]:.2f}`")

            # Text summarization
            summary = summarize_text(text)
            st.markdown(f"**Summary:** `{summary}`")
    else:
        st.warning("Please enter text to analyze.")
