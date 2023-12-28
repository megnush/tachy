import tkinter as tk
from tkinter import scrolledtext, simpledialog
import sqlite3
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

# Set the NLTK data path to the existing data directory
nltk.data.path.append("C:/Users/rohitkumar/AppData/Roaming/nltk_data")


# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize NLTK components
nltk.download('punkt')

# Database setup
db_name = "ai_chatbot.db"

def create_database():
    conn = sqlite3.connect(db_name)
    conn.execute('CREATE TABLE IF NOT EXISTS chat_history (question TEXT PRIMARY KEY, answer TEXT)')
    conn.commit()
    conn.close()

create_database()

def add_or_update_response(question, answer):
    try:
        processed_question = simple_text_processing(question)
        processed_answer = simple_text_processing(answer)  # Optionally preprocess the answer too

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO chat_history (question, answer) VALUES (?, ?)", (processed_question, processed_answer))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")


def get_response(question):
    try:
        processed_question = simple_text_processing(question)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT answer FROM chat_history WHERE question=?", (processed_question,))
        answer = cursor.fetchone()
        conn.close()
        return answer[0] if answer else None
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None

def simple_text_processing(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords, and lemmatize the words
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

    # Join processed tokens back into a string
    return ' '.join(processed_tokens)

# Define keyword-based responses
keyword_responses = {
    # Your predefined keyword responses
    # ...
}

# ML Model Setup
def train_model():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_history")
    data = cursor.fetchall()
    conn.close()

    if data:
        global vectorizer, model
        questions, responses = zip(*[(simple_text_processing(q), r) for q, r in data])

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(questions)

        # Handling imbalanced data
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X, responses)

        model = RandomForestClassifier()

        # Cross-validation to assess model performance
        scores = cross_val_score(model, X_res, y_res, cv=5)
        print("Model accuracy (cross-validation):", np.mean(scores))

        model.fit(X_res, y_res)
        print("Model trained on", len(X_res), "resampled data points")

train_model()

def get_ml_response(question):
    question_vector = vectorizer.transform([question])
    predicted = model.predict(question_vector)[0]
    predicted_proba = model.predict_proba(question_vector)[0]
    max_proba = max(predicted_proba)
    confidence_threshold = 0.5  # Adjust the threshold based on testing
    if max_proba >= confidence_threshold:
        return predicted
    else:
        return "I'm not sure about that. Could you clarify or ask something else?"

# GUI and Chatbot Logic
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("500x400")

chat_history = scrolledtext.ScrolledText(root, state='disabled', height=20, width=60, wrap=tk.WORD)
chat_history.pack(padx=10, pady=10)

entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=(0, 10))
entry.bind("<Return>", lambda event: send_message())

def send_message():
    user_input = entry.get().strip()
    entry.delete(0, tk.END)

    if user_input:
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"You: {user_input}\n")

        processed_input = simple_text_processing(user_input)
        response = get_ml_response(processed_input)

        if not response:
            for keyword, reply in keyword_responses.items():
                if keyword in processed_input:
                    response = reply
                    break

        if response:
            chat_history.insert(tk.END, f"AI: {response}\n")
        else:
            user_response = simpledialog.askstring("Teach Me", "I don't know that. What should I say?")
            if user_response:
                add_or_update_response(processed_input, user_response)
                train_model()  # Retrain the model with new data
                chat_history.insert(tk.END, f"AI: Thanks, I've learned something new.\n")

        chat_history.yview(tk.END)
        chat_history.configure(state='disabled')

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()
