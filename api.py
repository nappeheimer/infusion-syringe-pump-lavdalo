from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
import re
import io
from utils import preprocess # Make sure utils.py is in the same folder

app = FastAPI()

# 1. ALLOW REACT TO TALK TO PYTHON (CORS)
# This is crucial. Without this, your browser will block the connection.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this to your React URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD MODELS
try:
    svm_clf = joblib.load('svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Error: Models not found. Run train_model.py first.")

# 3. HELPER FUNCTIONS (Copied from your Streamlit app)
def clean_whatsapp_content(file_content):
    lines = file_content.splitlines()
    ios_date_pattern = re.compile(r'^[\u2000-\u206F]*\[\d{1,2}/\d{1,2}/\d{2,4},.*?:.*?\]')
    ios_msg_pattern = re.compile(r'^[\u2000-\u206F]*\[.*?\] (.*?): (.*)$')
    android_date_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? -')
    android_msg_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? - (.*?): (.*)$')

    junk_filters = ["Messages and calls are end-to-end encrypted", "created group", "added", "left", "image omitted", "sticker omitted", "video omitted"]

    data, current_sender, current_buffer = [], None, []
    def clean(t): return re.sub(r'[\u2000-\u206F]', '', t).replace('~', '').strip()

    for line in lines:
        line = line.strip()
        if not line: continue
        is_ios = ios_date_pattern.match(line)
        is_android = android_date_pattern.match(line)

        if is_ios or is_android:
            if current_sender and current_buffer:
                data.append([current_sender, " ".join(current_buffer)])
            current_sender, current_buffer = None, []
            match = ios_msg_pattern.match(line) if is_ios else android_msg_pattern.match(line)
            if match:
                sender, message = map(clean, match.groups())
                if any(j in message for j in junk_filters): continue
                if any(j in sender for j in ["created group", "added", "left"]): continue
                current_sender = sender
                current_buffer.append(message)
        else:
            if current_sender: current_buffer.append(clean(line))
            
    if current_sender and current_buffer:
        data.append([current_sender, " ".join(current_buffer)])
        
    return pd.DataFrame(data, columns=["sender", "message"])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 4. THE API ENDPOINT
@app.post("/analyze")
async def analyze_chat(file: UploadFile = File(...)):
    # Read file
    content = await file.read()
    string_data = content.decode("utf-8")
    
    # Clean Data
    dk = clean_whatsapp_content(string_data)
    
    if dk.empty:
        raise HTTPException(status_code=400, detail="No valid messages found")

    # Filter & Preprocess
    dk = dk.sort_values(by=['sender'])
    dk = dk[dk['message'].str.len() >= 1]
    value_counts = dk['message'].value_counts()
    dk = dk[~dk['message'].isin(value_counts[value_counts < 1].index)]
    
    if len(dk) == 0:
        raise HTTPException(status_code=400, detail="Not enough data after filtering")

    dk['message'] = dk['message'].apply(preprocess)

    # Predict
    X_to_predict = dk['message'].values.astype('U')
    tfidf_test_vectors = tfidf_vectorizer.transform(X_to_predict)
    raw_scores = svm_clf.decision_function(tfidf_test_vectors)
    
    # Process Scores
    score_df = pd.DataFrame(raw_scores, columns=svm_clf.classes_)
    score_df['sender'] = dk['sender'].values
    avg_scores = score_df.groupby('sender').mean()
    prob_scores = avg_scores.apply(softmax, axis=1)

    # Draft Pick Algorithm
    results = []
    friends_queue = list(prob_scores.index)
    assignments = {}

    while friends_queue:
        taken_characters = set()
        potential_matches = []
        for friend in friends_queue:
            friend_probs = prob_scores.loc[friend]
            sorted_matches = friend_probs.sort_values(ascending=False)
            for char_name, prob in sorted_matches.items():
                potential_matches.append({'friend': friend, 'character': char_name, 'score': prob})
        
        potential_matches.sort(key=lambda x: x['score'], reverse=True)
        round_assignments = []
        
        for match in potential_matches:
            friend, char = match['friend'], match['character']
            if friend in assignments or char in taken_characters: continue
            assignments[friend] = char
            taken_characters.add(char)
            round_assignments.append(friend)
            
        friends_queue = [f for f in friends_queue if f not in assignments]
        if not round_assignments and friends_queue:
            friend_to_force = friends_queue[0]
            best_char = prob_scores.loc[friend_to_force].idxmax()
            assignments[friend_to_force] = best_char
            friends_queue.pop(0)

    # Format for Frontend
    final_response = []
    for friend, character in assignments.items():
        confidence = prob_scores.loc[friend, character]
        final_response.append({
            "participant": friend,
            "character": character,
            "confidence": round(confidence * 100, 1) # Returns number like 85.5
        })
        
    return {"results": final_response}
