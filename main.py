import nltk
import json
import os
import tkinter as tk
from tkinter import messagebox
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt')
nltk.download('stopwords')

# Vietnamese stopwords
vietnamese_stopwords = set([
    'và', 'là', 'của', 'các', 'có', 'để', 'theo', 'đã', 'một', 'không', 'tôi', 'bạn', 'này', 'cái', 'lúc', 'nào', 'như', 'với', 'cho', 'hơn', 'sẽ', 'được', 'cùng', 'nhiều', 'từ', 'khi', 'mới', 'vì', 'lại', 'nên', 'đang', 'chưa', 'vì', 'mà', 'được', 'bị', 'sau', 'tất', 'những', 'đã', 'lên'
])

def preprocess_text(text):
    # Handle exact phrase searches by checking for quotes around multi-word phrases
    phrases = re.findall(r'"(.*?)"', text)
    for phrase in phrases:
        # Preprocess the phrase (ignore stopwords and stemming for phrases)
        phrase_tokens = word_tokenize(phrase.lower())
        phrase = ' '.join([word for word in phrase_tokens if word not in vietnamese_stopwords and word.isalpha()])
        text = text.replace(f'"{phrase}"', phrase)  # Replace the phrase in the text for tokenization
    
    # Proceed with normal text preprocessing (tokenization, stopwords removal, stemming)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in vietnamese_stopwords and word.isalpha()]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)


def load_data(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                data.append(json.load(file))
    return data


def create_index(data):
    documents = [doc['content'] for doc in data]
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Save index data to a JSON file
    index_data = {
        'vocabulary': vectorizer.get_feature_names_out().tolist(),
        'tfidf_matrix': tfidf_matrix.toarray().tolist(),
    }
    with open('index.json', 'w', encoding='utf-8') as json_file:
        json.dump(index_data, json_file, ensure_ascii=False, indent=4)
    
    return vectorizer, tfidf_matrix, data


def search(query, vectorizer, tfidf_matrix, data):
    phrases = re.findall(r'"(.*?)"', query)
    
    if phrases:
        # Exact phrase search: Match the phrases exactly
        for phrase in phrases:
            query = query.replace(f'"{phrase}"', f'({phrase})')  # Treat as exact match
    else:
        # Standard query processing
        query = preprocess_text(query)

    # Boolean search support (AND, OR, NOT operations)
    query = query.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')  # Normalize the operators
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_k = 10
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        result = data[idx]
        score = cosine_similarities[idx]
        results.append({
            'title': result['title'],
            'author': result['author'],
            'date': result['date'],
            'score': score,
            'content': result['content']
        })
    
    return results


def display_results(results):
    result_window = tk.Toplevel(root)
    result_window.title("Search Results")
    
    if not results:
        messagebox.showinfo("No Results", "No documents found for your query.")
        return
    
    canvas = tk.Canvas(result_window)
    scrollbar = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
    canvas.config(yscrollcommand=scrollbar.set)
    

    results_frame = tk.Frame(canvas)
    for idx, result in enumerate(results):
        result_frame = tk.Frame(results_frame)
        result_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(result_frame, text=f"Title: {result['title']}", font=('Arial', 12, 'bold')).pack(anchor='w')
        tk.Label(result_frame, text=f"Author: {result['author']}", font=('Arial', 10)).pack(anchor='w')
        tk.Label(result_frame, text=f"Date: {result['date']}", font=('Arial', 10)).pack(anchor='w')
        tk.Label(result_frame, text=f"Relevance Score: {result['score']:.4f}", font=('Arial', 10)).pack(anchor='w')
        tk.Label(result_frame, text=f"Content: {result['content'][:150]}...", font=('Arial', 10, 'italic')).pack(anchor='w')
        tk.Button(result_frame, text="Read More", command=lambda idx=idx: show_full_content(results[idx])).pack(side='right', padx=5, pady=5)


    canvas.create_window((0, 0), window=results_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Update the scroll region
    results_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


def show_full_content(result):
    content_window = tk.Toplevel(root)
    content_window.title(result['title'])
    
    tk.Label(content_window, text=f"Title: {result['title']}", font=('Arial', 14, 'bold')).pack(pady=10)
    tk.Label(content_window, text=f"Author: {result['author']}", font=('Arial', 12)).pack(pady=5)
    tk.Label(content_window, text=f"Date: {result['date']}", font=('Arial', 12)).pack(pady=5)
    tk.Label(content_window, text=f"Content: {result['content']}", font=('Arial', 10)).pack(padx=10, pady=10)

def on_search():
    query = search_entry.get()
    if query:
        results = search(query, vectorizer, tfidf_matrix, data)
        display_results(results)

root = tk.Tk()
root.title("Text Search Engine")

directory_path = 'data'
data = load_data(directory_path)
vectorizer, tfidf_matrix, data = create_index(data)


search_frame = tk.Frame(root)
search_frame.pack(pady=20)

search_entry = tk.Entry(search_frame, font=('Arial', 14), width=40)
search_entry.pack(side='left', padx=10)

search_button = tk.Button(search_frame, text="Search", command=on_search, font=('Arial', 14))
search_button.pack(side='left')


root.mainloop()
