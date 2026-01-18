import streamlit as st
import pandas as pd
import re
import base64
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Breaking Bad Chat Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. WALLPAPER SETUP ---
def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Dark overlay - Increased opacity for better contrast */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; 
                left: 0;
                width: 100%; 
                height: 100%;
                background: rgba(0, 0, 0, 0.6); 
                z-index: -1;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

# Try to load local wallpaper
set_background('my_wallpaper.jpg')

# --- 2. CSS STYLING (High Contrast Update) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto+Mono:wght@400;700&display=swap');
    
    /* Global Text */
    .stApp { color: #ffffff; }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title Animation */
    @keyframes neonGlow {
        0%, 100% { text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41; }
        50% { text-shadow: 0 0 20px #00ff41, 0 0 30px #00ff41, 0 0 40px #00ff41; }
    }
    
    /* Hero Section - High Opacity */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        background: rgba(0, 0, 0, 0.85);
        border-radius: 20px;
        margin-bottom: 40px;
        border: 1px solid rgba(0,255,65,0.4);
        box-shadow: 0 0 20px rgba(0,0,0,0.8);
    }
    .main-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 72px;
        color: #00ff41;
        letter-spacing: 8px;
        margin: 0;
        animation: neonGlow 2s ease-in-out infinite;
    }
    .element-symbol {
        display: inline-block;
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
        color: #000;
        padding: 8px 16px;
        margin: 0 4px;
        border: 2px solid #00ff41;
        box-shadow: 0 0 20px rgba(0,255,65,0.5);
        font-size: 56px;
        border-radius: 4px;
    }
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        font-size: 18px;
        color: #00ff41;
        letter-spacing: 3px;
        margin-top: 10px;
    }
    
    /* Data Tables - High Opacity Background */
    .stDataFrame {
        border: 1px solid #00ff41 !important;
        border-radius: 10px;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(10, 10, 10, 0.95);
        padding: 10px;
        border-radius: 10px;
    }

    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
        color: #000;
        border: none;
        padding: 10px 25px;
        font-family: 'Bebas Neue', cursive;
        font-size: 20px;
        letter-spacing: 1px;
        border-radius: 5px;
        text-transform: uppercase;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        box-shadow: 0 0 15px #00ff41;
        transform: scale(1.02);
    }
    
    /* Headers */
    .section-header {
        font-family: 'Bebas Neue', cursive;
        font-size: 32px;
        color: #00ff41;
        letter-spacing: 2px;
        margin-top: 30px;
        margin-bottom: 15px;
        padding: 10px;
        background: rgba(0, 0, 0, 0.85);
        border-radius: 8px;
        border-left: 5px solid #00ff41;
    }
    
    /* Stats Boxes */
    .stat-box {
        background: rgba(10, 10, 10, 0.9);
        border: 1px solid #00ff41;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .stat-val { font-family: 'Bebas Neue'; font-size: 42px; color: #fff; text-shadow: 2px 2px 4px #000; }
    .stat-lbl { font-family: 'Roboto Mono'; font-size: 14px; color: #00ff41; font-weight: bold; }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.85);
        padding: 20px;
        border-radius: 10px;
        border: 1px dashed #00ff41;
    }

    /* FIXED: Success/Error Message Boxes Visibility */
    div[data-testid="stNotification"], div[data-testid="stAlert"] {
        background-color: rgba(10, 10, 10, 0.95) !important;
        border: 1px solid #00ff41 !important;
        color: #ffffff !important;
        font-family: 'Roboto Mono', monospace;
    }
    /* The specific success icon color */
    div[data-testid="stNotification"] svg {
        fill: #00ff41 !important;
    }

</style>
""", unsafe_allow_html=True)


# --- 3. UTILS & PARSING ---
def preprocess(text):
    """Preprocess text for ML model"""
    import re
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
    except:
        return text.lower() if isinstance(text, str) else ""
    
    if not isinstance(text, str): return ""
    # Remove punctuations and numbers
    punctuations_removed_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(punctuations_removed_text.lower())
    filter_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemma = WordNetLemmatizer()
    l_tokens = [lemma.lemmatize(token) for token in filter_tokens]
    return ' '.join(l_tokens)

def parse_whatsapp_txt(content):
    """Robust WhatsApp Parser"""
    lines = content.splitlines()
    
    # Regex 1: iOS [12/12/2020, 10:10:10 PM] Sender: Msg
    ios_pattern = re.compile(r'^[\u2000-\u206F]*\[\d{1,2}/\d{1,2}/\d{2,4},.*?:.*?\]')
    ios_msg_pattern = re.compile(r'^[\u2000-\u206F]*\[.*?\] (.*?): (.*)$')
    
    # Regex 2: Android 12/12/2020, 10:10 PM - Sender: Msg
    android_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? -')
    android_msg_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? - (.*?): (.*)$')

    junk_filters = [
        "Messages and calls are end-to-end encrypted", "created group", "added", "left", 
        "changed the subject", "security code changed", "image omitted", "sticker omitted", 
        "video omitted", "GIF omitted", "This message was deleted", "document omitted"
    ]

    data, current_sender, current_buffer = [], None, []
    
    def clean(t): return re.sub(r'[\u2000-\u206F]', '', t).replace('~', '').strip()

    for line in lines:
        line = line.strip()
        if not line: continue

        is_ios = ios_pattern.match(line)
        is_android = android_pattern.match(line)

        if is_ios or is_android:
            if current_sender and current_buffer:
                data.append([current_sender, " ".join(current_buffer)])
            
            current_sender, current_buffer = None, []
            
            match = ios_msg_pattern.match(line) if is_ios else android_msg_pattern.match(line)
            
            if match:
                sender, message = map(clean, match.groups())
                if any(j in message for j in junk_filters): continue
                if any(j in sender.lower() for j in ["created group", "added", "left"]): continue
                
                current_sender = sender
                current_buffer.append(message)
        else:
            if current_sender: current_buffer.append(clean(line))
            
    if current_sender and current_buffer:
        data.append([current_sender, " ".join(current_buffer)])
        
    return pd.DataFrame(data, columns=["sender", "message"])


def load_models():
    try:
        svm_clf = joblib.load('svm_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return svm_clf, tfidf_vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `train_model.py` first.")
        return None, None

# --- 4. CORE LOGIC: ML + SMART LOOP BACK ---
def analyze_chat_data(df, svm_clf, tfidf_vectorizer):
    # 1. Filter Data & Prep
    df['clean_msg'] = df['message'].apply(preprocess)
    df = df[df['clean_msg'].str.len() > 0]
    
    # 2. Get Statistics (Include everyone with >0 messages)
    stats = df.groupby('sender').agg({
        'message': ['count', lambda x: np.mean([len(str(m).split()) for m in x])]
    }).reset_index()
    stats.columns = ['sender', 'msg_count', 'avg_words']
    stats = stats[stats['msg_count'] > 0] 

    if stats.empty: return None

    # 3. Calculate All Probabilities First
    senders = stats['sender'].tolist()
    # List of all (Sender, Character, Score)
    all_potential_matches = []

    for sender in senders:
        user_msgs = df[df['sender'] == sender]['clean_msg']
        full_text = " ".join(user_msgs)
        if not full_text.strip(): vector = tfidf_vectorizer.transform([" "]) 
        else: vector = tfidf_vectorizer.transform([full_text])
            
        raw_scores = svm_clf.decision_function(vector)[0]
        # Softmax
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / exp_scores.sum()
        
        # Add every character's score for this person to the pool
        for idx, char_name in enumerate(svm_clf.classes_):
            all_potential_matches.append({
                'sender': sender,
                'char': char_name,
                'score': probs[idx]
            })

    # 4. Balanced Draft Pick Algorithm
    # Sort ALL matches by confidence (highest first)
    all_potential_matches.sort(key=lambda x: x['score'], reverse=True)
    
    assignments = {} # {sender: (char, score)}
    char_usage = {c: 0 for c in svm_clf.classes_} # Track how many times each char is used
    
    # We loop through "Usage Tiers". 
    # Tier 1: Assign everyone until all chars used once.
    # Tier 2: Allow chars to be used a 2nd time.
    # Tier 3: ...
    
    unassigned_senders = set(senders)
    current_max_usage = 1 
    
    while unassigned_senders:
        progress_made = False
        
        # Try to assign based on current match list
        for match in all_potential_matches:
            s = match['sender']
            c = match['char']
            
            # If sender needs assignment AND character is below current usage limit
            if s in unassigned_senders and char_usage[c] < current_max_usage:
                assignments[s] = (c, match['score'])
                unassigned_senders.remove(s)
                char_usage[c] += 1
                progress_made = True
        
        # If we went through the whole list and people are still waiting,
        # it means all characters are full for this Tier. Increase limit.
        if not progress_made and unassigned_senders:
            current_max_usage += 1
            
            # Safety break: If usage goes absurdly high (e.g. 1000), just force assign logic
            if current_max_usage > 1000:
                for s in list(unassigned_senders):
                    # Just give them their absolute top pick regardless of usage
                    top_pick = [m for m in all_potential_matches if m['sender'] == s][0]
                    assignments[s] = (top_pick['char'], top_pick['score'])
                    unassigned_senders.remove(s)
                break

    # 5. Format Final Data
    final_rows = []
    for person, (char, conf) in assignments.items():
        p_stat = stats[stats['sender'] == person].iloc[0]
        final_rows.append({
            'Participant': person,
            'Breaking Bad Character': char,
            'Confidence': f"{conf:.1%}",
            'Message Count': int(p_stat['msg_count']),
            'Avg Words/Msg': f"{p_stat['avg_words']:.1f}"
        })
    
    return pd.DataFrame(final_rows)


# --- 5. MAIN APP UI ---
def main():
    # Hero Title
    st.markdown("""
    <div class="hero-container">
        <h1 class="main-title">
            <span class="element-symbol">Br</span>eaking 
            <span class="element-symbol">Ba</span>d
        </h1>
        <div class="subtitle">CHAT ANALYZER</div>
    </div>
    """, unsafe_allow_html=True)

    svm_clf, tfidf_vectorizer = load_models()
    if not svm_clf: st.stop()

    # Upload Section
    st.markdown('<div class="section-header">1. Upload Chat</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        txt_file = st.file_uploader("Upload WhatsApp .txt", type=['txt'])
    with col2:
        csv_file = st.file_uploader("Upload CSV (sender, message)", type=['csv'])

    df = None
    if csv_file:
        df = pd.read_csv(csv_file)
    elif txt_file:
        content = txt_file.getvalue().decode('utf-8')
        df = parse_whatsapp_txt(content)

    # Analysis Section
    if df is not None and not df.empty:
        # Custom Success Message with High Contrast
        st.markdown(f"""
        <div style="background:rgba(10,10,10,0.9); border:1px solid #00ff41; padding:10px; border-radius:5px; margin-bottom:10px;">
            <span style="color:#00ff41; font-weight:bold;">✅ SUCCESS:</span> 
            Loaded {len(df)} messages from {df['sender'].nunique()} participants.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("COOK (RUN ANALYSIS)"):
            with st.spinner("⚗️ Distilling personality traits..."):
                results_df = analyze_chat_data(df, svm_clf, tfidf_vectorizer)
            
            if results_df is not None:
                # Display Stats
                st.markdown('<div class="section-header">2. Statistics</div>', unsafe_allow_html=True)
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(df)}</div><div class="stat-lbl">TOTAL MESSAGES</div></div>', unsafe_allow_html=True)
                with sc2:
                    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(results_df)}</div><div class="stat-lbl">PARTICIPANTS</div></div>', unsafe_allow_html=True)
                with sc3:
                    st.markdown(f'<div class="stat-box"><div class="stat-val">{results_df["Message Count"].mean():.0f}</div><div class="stat-lbl">AVG MSG COUNT</div></div>', unsafe_allow_html=True)

                # Display Table
                st.markdown('<div class="section-header">3. Lab Results</div>', unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Download CSV
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ DOWNLOAD ANALYSIS CSV",
                    csv_data,
                    "breaking_bad_analysis.csv",
                    "text/csv"
                )
            else:
                st.warning("No data found to analyze.")

if __name__ == "__main__":
    main()