# ============================================================
# 💬 WhatsApp Chat Analyzer (NLP Project)
# ============================================================
# Features:
# 1. Basic chat statistics (messages, words, dates)
# 2. Sentiment analysis & emoji usage
# 3. Talk balance / dominance index
# 4. Response energy (timing analysis)
# 5. Style profiling (sentence, vocab, punctuation)
# 6. Politeness & directness analysis
# 7. Keyword extraction (TF-IDF based)
# 8. Summary insights
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from datetime import datetime
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer  # <-- NEW for keyword extraction
nltk.download('punkt')

# ============================================================
# 🧩 Helper Functions
# ============================================================

# --- Parse WhatsApp exported .txt file ---
def parse_chat(file_content):
    """
    Parses WhatsApp exported chat text (.txt) into a structured DataFrame.
    Handles both 12-hour (AM/PM) and 24-hour time formats.
    Example:
    '12/10/2024, 10:45 pm - Name: message'
    """
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[apAP][mM])?)\s-\s([^:]+):\s(.*)$'
    data = []

    for line in file_content.splitlines():
        match = re.match(pattern, line)
        if match:
            date_str, time_str, sender, message = match.groups()
            dt = None

            # Try multiple datetime formats
            for fmt in ("%d/%m/%Y %I:%M %p", "%d/%m/%y %I:%M %p",
                        "%m/%d/%Y %I:%M %p", "%m/%d/%y %I:%M %p",
                        "%d/%m/%Y %H:%M", "%d/%m/%y %H:%M"):
                try:
                    dt = datetime.strptime(f"{date_str} {time_str.strip()}", fmt)
                    break
                except:
                    continue

            if dt:
                data.append([dt, sender.strip(), message.strip()])

    df = pd.DataFrame(data, columns=['datetime', 'sender', 'message'])

    # Remove system messages and media placeholders
    df = df[~df['message'].str.contains("end-to-end encrypted", case=False, na=False)]
    df = df[~df['message'].str.startswith('<Media omitted>')]

    return df.reset_index(drop=True)


# --- Clean and normalize text ---
def clean_message(msg):
    msg = re.sub(r"http\S+", "", msg)  # remove links
    msg = re.sub(r"[^A-Za-z\s]", "", msg)  # keep only text
    msg = msg.lower()
    return msg


# --- Sentiment calculation using TextBlob ---
def get_sentiment(text):
    if not text.strip():
        return 0
    return TextBlob(text).sentiment.polarity


# --- Extract emojis from a message ---
def extract_emojis(s):
    return [c for c in s if c in emoji.EMOJI_DATA]


# ============================================================
# 🚀 Streamlit UI Layout
# ============================================================

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("💬 WhatsApp Chat Analyzer (NLP Project)")
st.markdown("Upload your exported **1-on-1 WhatsApp chat (.txt)** file to begin analysis.")

uploaded_file = st.file_uploader("📁 Upload Chat File", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    df = parse_chat(content)

    if df.empty:
        st.error("❌ Couldn't parse the chat file. Please upload a valid WhatsApp chat export.")
    else:
        st.success("✅ Chat successfully parsed!")

        # Preprocess and enrich data
        df['clean_text'] = df['message'].apply(clean_message)
        df['sentiment_score'] = df['clean_text'].apply(get_sentiment)
        df['emoji_list'] = df['message'].apply(extract_emojis)

        senders = df['sender'].unique()
        if len(senders) < 2:
            st.warning("This analysis works best for 1-on-1 chats.")
        user1, user2 = senders[0], senders[1] if len(senders) > 1 else ("User", "Other")

        # ============================================================
        # 📊 BASIC CHAT STATS
        # ============================================================
        st.header("📊 Basic Chat Overview")
        total_msgs = len(df)
        total_words = df['clean_text'].apply(lambda x: len(x.split())).sum()
        start, end = df['datetime'].min(), df['datetime'].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", total_msgs)
        col2.metric("Total Words", total_words)
        col3.metric("Chat Duration", f"{(end-start).days} days")

        # ============================================================
        # 😊 SENTIMENT & EMOJI ANALYSIS
        # ============================================================
        st.header("😊 Sentiment & Emoji Insights")
        sentiment_avg = df.groupby('sender')['sentiment_score'].mean()
        emoji_count = df['emoji_list'].explode().value_counts().head(10)

        # --- Sentiment Chart ---
        fig, ax = plt.subplots()
        ax.bar(sentiment_avg.index, sentiment_avg.values, color=["#81C784", "#64B5F6"])
        ax.set_title("Average Sentiment per User", fontsize=12)
        ax.set_xlabel("Participants")
        ax.set_ylabel("Average Sentiment")
        ax.set_xticklabels(sentiment_avg.index, rotation=0)
        st.pyplot(fig)

        # --- Emoji Chart ---
        fig, ax = plt.subplots()
        ax.bar(emoji_count.index, emoji_count.values, color="#FFD54F")
        ax.set_title("Top Emojis Used", fontsize=12)
        ax.set_xlabel("Emoji")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(emoji_count.index, rotation=0, fontsize=14)
        st.pyplot(fig)

        # ============================================================
        # 🗣 TALK BALANCE / DOMINANCE
        # ============================================================
        st.header("🗣 Talk Balance & Dominance Index")
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        word_totals = df.groupby('sender')['word_count'].sum()
        dominance = word_totals / word_totals.sum()

        fig, ax = plt.subplots()
        ax.bar(word_totals.index, word_totals.values, color=["#90CAF9", "#F48FB1"])
        ax.set_title("Word Share per Participant", fontsize=12)
        ax.set_xlabel("Participants")
        ax.set_ylabel("Total Words")
        ax.set_xticklabels(word_totals.index, rotation=0)
        st.pyplot(fig)

        st.info(f"🗣 {dominance.idxmax()} leads the conversation with {dominance.max()*100:.1f}% of total words.")

        # ============================================================
        # ⚡ RESPONSE ENERGY / TIMING
        # ============================================================
        st.header("⚡ Response Energy & Activity Patterns")
        df['response_gap_seconds'] = df['datetime'].diff().dt.total_seconds()
        df['previous_sender'] = df['sender'].shift(1)
        response_times = df[df['sender'] != df['previous_sender']].groupby('sender')['response_gap_seconds'].mean()

        fig, ax = plt.subplots()
        ax.bar(response_times.index, response_times.values, color=["#FF8A65", "#4DB6AC"])
        ax.set_title("Average Response Time (in seconds)", fontsize=12)
        ax.set_xlabel("Participants")
        ax.set_ylabel("Average Response Time (s)")
        ax.set_xticklabels(response_times.index, rotation=0)
        st.pyplot(fig)

        # ============================================================
        # ✍️ STYLE PROFILING
        # ============================================================
        st.header("✍️ Style Profiling (Linguistic Style)")
        def style_metrics(messages):
            text = ' '.join(messages)
            words = re.findall(r'\w+', text.lower())
            sentences = re.split(r'[.!?]', text)
            return {
                "Average Sentence Length": len(words)/max(1,len(sentences)),
                "Lexical Diversity": len(set(words))/max(1,len(words)),
                "Question Usage (%)": text.count('?')/max(1,len(messages))*100,
                "Exclamation Usage (%)": text.count('!')/max(1,len(messages))*100,
                "Pronoun Usage (%)": sum(w in ["i","you","we","he","she","they"] for w in words)/max(1,len(words))*100
            }

        style_stats = df.groupby('sender')['message'].apply(style_metrics)
        st.dataframe(pd.DataFrame(style_stats.tolist(), index=style_stats.index))

        # ============================================================
        # 🙇 POLITENESS & DIRECTNESS
        # ============================================================
        st.header("🙇 Politeness & Directness Analysis")
        polite_words = ["please", "thank", "sorry", "could", "would", "may", "might", "appreciate"]
        hedges = ["maybe", "perhaps", "sort of", "kind of"]

        def politeness_score(messages):
            text = ' '.join(messages).lower()
            return {
                "Politeness Indicators": sum(word in text for word in polite_words),
                "Hedging Indicators": sum(word in text for word in hedges),
                "Direct Expressions": text.count('!')
            }

        polite_stats = df.groupby('sender')['message'].apply(politeness_score)
        st.dataframe(pd.DataFrame(polite_stats.tolist(), index=polite_stats.index))
        st.caption("Higher politeness → more courteous tone; higher directness → more assertive tone.")

        # ============================================================
        # 🔑 KEYWORD EXTRACTION (TF-IDF)
        # ============================================================
        st.header("🔑 Keyword Extraction (TF-IDF Based)")
        st.markdown("Shows top keywords used by each participant, reflecting their main conversation themes.")

        # Prepare TF-IDF per participant
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        sender_keywords = {}

        for sender, messages in df.groupby('sender')['clean_text']:
            texts = messages.tolist()
            if len(texts) < 3:
                continue
            X = tfidf.fit_transform(texts)
            mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
            terms = tfidf.get_feature_names_out()
            top_indices = mean_tfidf.argsort()[::-1][:10]
            sender_keywords[sender] = [terms[i] for i in top_indices]

        keyword_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sender_keywords.items()]))
        st.dataframe(keyword_df)
        st.caption("Top 10 keywords per participant based on TF-IDF importance.")

        # ============================================================
        # 📋 SUMMARY INSIGHTS
        # ============================================================
        st.header("📋 Summary Insights")
        summary_text = f"""
        - **{dominance.idxmax()}** leads the conversation with **{dominance.max()*100:.1f}%** of the total words.  
        - **{response_times.idxmin()}** replies faster on average (**{response_times.min():.1f} sec**).  
        - **{style_stats.index[0][0]}** tends to use longer sentences, while **{style_stats.index[1][0]}** has slightly higher lexical diversity.  
        """
        st.markdown(summary_text)

        st.success("✅ Analysis complete! Scroll above for detailed insights.")

else:
    st.info("Please upload your WhatsApp chat text file to start analysis.")
