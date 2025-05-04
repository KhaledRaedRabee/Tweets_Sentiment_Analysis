import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- App Config ---
st.set_page_config(page_title="Tweet Sentiment Analyzer",
                   page_icon="🐦",
                   layout="centered",
                   initial_sidebar_state="auto")

# --- Custom CSS for better look ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        color: white;
        background-color: #1DA1F2;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
pipeline_LR = joblib.load('C:/Users/NTC/Desktop/TweetsSentiment/logistic_regression_pipeline.pkl')

# --- App Title ---
st.title("🐦 Tweet Sentiment Analyzer")
st.subheader("Is your tweet Positive or Negative?")

st.write("Enter a tweet below and let our sentiment analyzer predict its mood!")

# --- Input Tweet ---
tweet = st.text_area("✍️ Type your tweet here:", placeholder="I love how beautiful the weather is today!", height=150)

# --- Predict Button ---
if st.button("🔍 Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        user_tweet = pd.Series([tweet])
        prediction = pipeline_LR.predict(user_tweet)[0]

        if prediction == 1:
            st.success("✅ **Predicted Sentiment: Positive** 😊")
        else:
            st.error("❌ **Predicted Sentiment: Negative** 😞")

# --- Optional WordCloud Section ---
st.markdown("---")
st.subheader("📊 Frequent Words in Tweets (Training Data)")

# Optional — load your training dataset to make a wordcloud
try:
    df = pd.read_csv('C:/Users/NTC/Desktop/TweetsSentiment/tweet_data.csv')  # Replace with your dataset file
    all_words = ' '.join(df['text'].dropna().astype(str).tolist())

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='Blues', collocations=False).generate(all_words)

    plt.figure(figsize=(10, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
except Exception as e:
    st.info("ℹ️ Wordcloud preview not available (training dataset missing).")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Twitter Sentiment Classifier Demo")