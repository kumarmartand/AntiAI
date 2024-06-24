import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string

# Loading GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def plot_top_repeated_words(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]
    
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)

    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title="Top 10 Most Repeated Words")
    st.plotly_chart(fig, use_container_width=True)

st.title("Anti-AI: AI Plagiarism Tool")

text_area = st.text_area("Enter your text")

if text_area:
    if st.button("Analyze"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Your Text")
            st.success(text_area)
        
        with col2:
            st.info("Calculated Score")
            perplexity = calculate_perplexity(text_area)
            burstiness_score = calculate_burstiness(text_area)

            st.success(f"Perplexity Score: {perplexity}")
            st.success(f"Burstiness Score: {burstiness_score}")

            if perplexity > 30000 and burstiness_score < 0.2:
                st.error("Text Analysis Result: AI Generated Content")
            else:
                st.success("Text Analysis Result: Human Generated Content")

            st.warning("DISCLAIMER: This tool only provides assistance for checking the plagiarism of any content provided. It is not fully accurate for any further decision-making.")

        with col3:
            st.info("Basic Information")
            plot_top_repeated_words(text_area)
