# Import necessary libraries
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Define function to load models
@st.cache_data(allow_output_mutation=True)
def load_models():
    classification_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name, model_max_length=512)

    summarization_model_name = 't5-base'
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name, model_max_length=512)

    return classification_model, classification_tokenizer, summarization_model, summarization_tokenizer

classification_model, classification_tokenizer, summarization_model, summarization_tokenizer = load_models()

# Title of the app
st.title('Text Classification and Summarization with Hugging Face')

# Take user input
text = st.text_area("Enter text:", "")
submit_button = st.button("Analyze Text")

# Predict function for sentiment analysis
def predict_sentiment(text):
    tokenized_text = classification_tokenizer.tokenize(text)
    results = []
    
    # Break text into chunks of max_model_length tokens
    for i in range(0, len(tokenized_text), classification_tokenizer.model_max_length):
        chunk = tokenized_text[i:i+classification_tokenizer.model_max_length]
        chunk = classification_tokenizer.convert_tokens_to_string(chunk)

        inputs = classification_tokenizer(chunk, return_tensors="pt", truncation=True, padding='max_length')
        outputs = classification_model(**inputs)
        probs = torch.nn.functional.softmax(outputs[0], dim=-1)
        results.append(probs.detach().numpy())
    return results

# Predict function for text summarization
def summarize_text(text):
    tokenized_text = summarization_tokenizer.tokenize(text)
    summaries = []
    
    # Break text into chunks of max_model_length tokens
    for i in range(0, len(tokenized_text), summarization_tokenizer.model_max_length):
        chunk = tokenized_text[i:i+summarization_tokenizer.model_max_length]
        chunk = summarization_tokenizer.convert_tokens_to_string(chunk)
        
        inputs = summarization_tokenizer.encode("summarize: " + chunk, return_tensors="pt", truncation=True, padding='max_length')
        outputs = summarization_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarization_tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '')
        summaries.append(summary)
    return summaries

if submit_button:
    if text:
        with st.spinner("Analyzing..."):
            # Sentiment analysis
            results = predict_sentiment(text)
            for i, probs in enumerate(results):
                st.markdown(f"**Result {i+1}:**")
                st.markdown(f"**Positive sentiment:** `{probs[0][1]:.2f}`")
                st.markdown(f"**Negative sentiment:** `{probs[0][0]:.2f}`")

            # Text summarization
            summaries = summarize_text(text)
            for i, summary in enumerate(summaries):
                st.markdown(f"**Summary {i+1}:** `{summary}`")
    else:
        st.warning("Please enter text to analyze.")
