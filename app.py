# %%
import requests
from bs4 import BeautifulSoup
# Text scraper with basic cleaning to extract article paragraphs and remove common junk content like ads, footers, and author bios.
def get_article_text(url):
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    article = soup.find("article")

    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    cleaned_paragraphs = []

    junk_phrases = [
        "subscribe",
        "newsletter",
        "sign up",
        "donate",
        "support our",
        "comment below",
        "we would love your thoughts",
        "follow us",
        "award-winning reporting",
        "nonprofit",
        "media company",
        "click here",
        "read more",
        "share this story",
        "about the author"
    ]

    for p in paragraphs:

        text = p.get_text().strip()

        # Skip short paragraphs
        if len(text) < 40:
            continue

        # Remove footer / ads / author bios
        if any(phrase in text.lower() for phrase in junk_phrases):
            continue

        cleaned_paragraphs.append(text)

    return cleaned_paragraphs

# %%
# Use a zero-shot classification model for better context understanding and more accurate sentiment detection, especially for negative content.
from transformers import pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli")

def analyze_sentiment_ai(paragraphs):
    """
    Run context-aware sentiment classification on each paragraph.
    Uses zero-shot classification for more accurate negative detection.
    """
    results = []
    max_paragraphs = min(len(paragraphs), 25)

    candidate_labels = ["positive", "neutral", "negative"]

    for paragraph in paragraphs[:max_paragraphs]:
        if len(paragraph.strip()) < 40:
            continue

        res = classifier(paragraph, candidate_labels=candidate_labels)
        
        top_idx = res["scores"].index(max(res["scores"]))
        label = res["labels"][top_idx]
        score = res["scores"][top_idx]

        results.append({
            "text": paragraph,
            "label": label,
            "score": score
        })

    return results

# %%
# Aggregate paragraph-level sentiment into an overall article sentiment score with weighted scoring to highlight critical or negative content.
def aggregate_sentiment(results):
    score = 0
    for r in results:
        if r['label'] == 'positive':
            score += r['score'] * 0.5
        elif r['label'] == 'negative':
            score -= r['score'] * 1.5  
    if score > 1:
        return "positive"
    elif score < -0.5:
        return "negative"
    else:
        return "neutral"

def compute_aggregate_score(results):
    score = 0
    for r in results:
        if r['label'] == 'POSITIVE':
            score += r['score'] * 0.5
        else:
            score -= r['score']
    return score

# %%
# Adjust sentiment classification using contextual keywords associated with criticism, risk, or controversial topics.
def adjust_for_keywords(text, sentiment, aggregate_score):
    negative_keywords = [
        "genocidal", "war crimes", "violence", "threat",
        "controversy", "surveillance", "risk", "privacy", "backlash"
    ]
    hits = sum(word.lower() in text.lower() for word in negative_keywords)

    if hits >= 2:
        return "negative"
    return sentiment

# %%
# Classify the article theme based on domain-specific keyword matching to identify if it's about government/defense, privacy, AI/technology, or finance.
def classify_theme(text):
    themes = {
        "Government/Defense": ["military", "government", "defense", "contract"],
        "Privacy": ["privacy", "surveillance", "data concerns"],
        "AI/Technology": ["AI", "artificial intelligence", "software", "platform"],
        "Finance": ["revenue", "earnings", "stock", "profit"]
    }

    text_lower = text.lower()
    scores = {}

    for theme, keywords in themes.items():
        scores[theme] = sum(text_lower.count(keyword) for keyword in keywords)

    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Other"


# %%
# Main function to analyze an article URL and return sentiment, theme, and detailed paragraph-level results.
def analyze_article(url):
    paragraphs = get_article_text(url)

    ai_results = analyze_sentiment_ai(paragraphs)

    aggregate_score = sum(
        r['score'] if r['label'] == 'positive' else -r['score']
        for r in ai_results
    )

    sentiment = aggregate_sentiment(ai_results)
    sentiment = adjust_for_keywords(" ".join(paragraphs), sentiment, aggregate_score)

    theme = classify_theme(" ".join(paragraphs))

    return {
        "sentiment": sentiment,
        "details": ai_results,
        "theme": theme,
        "aggregate_score": aggregate_score
    }

# %%
# Streamlit app to input article URL, display sentiment analysis results, and show paragraph-level details for transparency and debugging.
import streamlit as st

st.title("Palantir Sentiment Monitoring Tool")

url = st.text_input("Enter the URL of the article:")

if st.button("Analyze"):
    if url:
        with st.spinner("Analyzing..."):
            result = analyze_article(url)

        st.subheader("Analysis")
        st.write(f"Sentiment: {result['sentiment']}")
        st.write(f"Theme: {result['theme']}")
        
        st.subheader("Paragraph-level results:")
        for i, r in enumerate(result["details"]):
            st.markdown(f"**Paragraph {i+1}: {r['label']} ({r['score']:.2f})**")
            st.write(r["text"])
            st.markdown("---")
        
        
        aggregate_score = compute_aggregate_score(result["details"])
        st.write(f"Aggregate score: {aggregate_score:.2f}")
        
    else:
        st.warning("Please enter a URL.")

# %%
st.markdown("---")

st.subheader("About This Tool")

st.write("""
This application analyzes the sentiment of news articles related to Palantir by extracting article text 
from a user-provided URL and applying a transformer-based natural language processing model.

The workflow includes:

• Extracting article paragraphs using BeautifulSoup  
• Running **paragraph-level sentiment classification** using a **zero-shot HuggingFace model (`facebook/bart-large-mnli`)** for improved context understanding  
• Aggregating paragraph sentiment into an overall article sentiment score with **weighted scoring** to highlight critical or negative content  
• Adjusting sentiment classification using contextual keywords associated with criticism, risk, or controversial topics  
• Classifying the article theme based on domain-specific keyword matching  

This tool is designed to help monitor how Palantir is discussed across media sources and identify whether 
coverage trends are positive, neutral, or negative in tone, with greater accuracy for long-form, formal news content.
""")

st.caption("Built with Python, Streamlit, HuggingFace Transformers, and BeautifulSoup.")


