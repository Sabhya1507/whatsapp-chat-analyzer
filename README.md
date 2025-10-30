# whatsapp-chat-analyzer
A Streamlit-based NLP app that analyzes exported one-on-one WhatsApp chats. It provides insights on sentiment, emoji usage, talk balance, response timing, linguistic style, and politeness using text preprocessing, tokenization, and sentiment analysis techniques.


# 💬 WhatsApp Chat Analyzer (NLP Project)

An interactive **Streamlit NLP app** that analyzes exported 1-on-1 WhatsApp chats.  
It provides insights about chat dynamics, sentiment, emoji use, politeness, linguistic style, and more — all in one clean dashboard.

---

## 🚀 Features

Parse WhatsApp exported `.txt` chats  
Sentiment analysis (TextBlob)  
Emoji and word usage frequency  
Talk balance / dominance index  
Response energy (average reply time)  
Style profiling (sentence length, lexical diversity, pronoun use)  
Politeness & directness metrics  
Interactive charts via Plotly  

---

## 🧩 How It Works

1. Export a **1-on-1 chat** from WhatsApp (without media).
   - In WhatsApp: `⋮` → *More* → *Export Chat* → *Without Media*
2. Upload the `.txt` file into the app.
3. View insights and charts generated from NLP analysis.

---

## 🛠️ Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/sabhya1507/whatsapp-chat-analyzer.git
cd whatsapp-chat-analyzer
pip install -r requirements.txt

```

## How to run:

Either save your own exported chat you wish to analyze to a local folder of your liking
or copy the sample_chat.txt file into a txt file of your own and save.

```bash
streamlit run app.py
```

When prompted, upload the text file on the browser window.

## Example Outputs:

You’ll get insights like:

Average sentiment per participant
Most used emojis
Talk balance and dominance ratio
Average response time
Linguistic style and politeness measures
