import os
import json
import nltk
import re
import tkinter as tk
from tkinter import messagebox, filedialog
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Sumy imports ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Download NLTK dependencies (you can comment them out if already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# If you have Vietnamese content, define your Vietnamese stopwords here
vietnamese_stopwords = {
    'và', 'là', 'của', 'các', 'có', 'để', 'theo', 'đã', 'một',
    'không', 'tôi', 'bạn', 'này', 'cái', 'lúc', 'nào', 'như', 'với',
    'cho', 'hơn', 'sẽ', 'được', 'cùng', 'nhiều', 'từ', 'khi', 'mới',
    'vì', 'lại', 'nên', 'đang', 'chưa', 'mà', 'bị', 'sau', 'tất',
    'những', 'lên'
}

# Language settings for Sumy
LANGUAGE = "english"   # or "vietnamese" if you have a custom approach
SENTENCES_COUNT = 3    # number of sentences for single-post summary
MULTI_SENTENCES_COUNT = 5  # number of sentences for multi-doc summary

def preprocess_text(text):
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
    return vectorizer, tfidf_matrix, data

def search_query(query, vectorizer, tfidf_matrix, data):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k = 10
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        result = data[idx]
        score = cosine_similarities[idx]
        results.append({
            'title': result.get('title', 'No Title'),
            'author': result.get('author', 'No Author'),
            'date': result.get('date', 'No Date'),
            'score': score,
            'content': result.get('content', '')
        })
    return results

def calculate_summation(results):
    total_word_count = sum(len(result['content'].split()) for result in results)
    total_posts = len(results)
    return total_word_count, total_posts

# --- Single-document Summarization with Sumy ---
def summarize_single_post(text, language=LANGUAGE, sentence_count=SENTENCES_COUNT):
    # Create a plaintext parser for the text
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    # Use LSA Summarizer (you could also use LuhnSummarizer, LexRankSummarizer, etc.)
    summarizer = LsaSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)

    summary_sentences = summarizer(parser.document, sentence_count)
    summary_text = "\n".join(str(sentence) for sentence in summary_sentences)
    if not summary_text.strip():
        summary_text = "Summary could not be generated (text too short or repetitive)."
    return summary_text

# --- Multi-document Summarization with Sumy ---
def multi_document_summarize(results, language=LANGUAGE, sentence_count=MULTI_SENTENCES_COUNT):
    merged_text = "\n".join([r['content'] for r in results])
    parser = PlaintextParser.from_string(merged_text, Tokenizer(language))
    summarizer = LsaSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)

    summary_sentences = summarizer(parser.document, sentence_count)
    summary_text = "\n".join(str(sentence) for sentence in summary_sentences)
    if not summary_text.strip():
        summary_text = "Summary could not be generated (text too short or repetitive)."
    return summary_text

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
        result_frame = tk.Frame(results_frame, bd=1, relief='solid', padx=5, pady=5)
        result_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(result_frame, text=f"Title: {result['title']}", font=('Arial', 12, 'bold')).pack(anchor='w')
        tk.Label(result_frame, text=f"Author: {result['author']}", font=('Arial', 10)).pack(anchor='w')
        tk.Label(result_frame, text=f"Date: {result['date']}", font=('Arial', 10)).pack(anchor='w')
        tk.Label(result_frame, text=f"Relevance Score: {result['score']:.4f}", font=('Arial', 10)).pack(anchor='w')
        
        # Show partial content
        preview_text = (result['content'][:150] + '...') if len(result['content']) > 150 else result['content']
        tk.Label(result_frame, text=f"Content: {preview_text}", font=('Arial', 10, 'italic')).pack(anchor='w')
        
        # Summarize button for each item
        def summarize_post(res=result):
            single_summary = summarize_single_post(res['content'])
            tk.messagebox.showinfo("Single Post Summary", single_summary)
        
        tk.Button(result_frame, text="Summarize this Post",
                  command=summarize_post, 
                  font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)

    # Summation feature for all 10 results
    summation_label_frame = tk.Frame(results_frame)
    summation_label_frame.pack(fill='x', padx=10, pady=5)
    total_words, total_posts = calculate_summation(results)
    summation_label = tk.Label(summation_label_frame, 
                               text=f"Total Word Count: {total_words} | Total Posts: {total_posts}", 
                               font=('Arial', 12, 'bold'))
    summation_label.pack(anchor='w', pady=5)
    
    # Multi-document Summation button for top results
    def on_multi_doc_summarize():
        summary_text = multi_document_summarize(results)
        summary_window = tk.Toplevel(result_window)
        summary_window.title("Combined Summary")
        tk.Label(summary_window, text="Multi-document Summary", font=('Arial', 12, 'bold')).pack(pady=5)
        text_box = tk.Text(summary_window, wrap='word', width=80, height=20)
        text_box.pack(padx=10, pady=5)
        text_box.insert('1.0', summary_text)
    
    tk.Button(summation_label_frame, text="Summarize All (Top 10)",
              command=on_multi_doc_summarize, 
              font=('Arial', 10, 'bold')).pack(anchor='w')

    canvas.create_window((0, 0), window=results_frame, anchor="nw")
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    results_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def on_summation():
    results = search_query(search_entry.get(), vectorizer, tfidf_matrix, data)
    if results:
        total_words, total_posts = calculate_summation(results)
        messagebox.showinfo("Summation Results",
                            f"Total Word Count (Top 10): {total_words}\nTotal Posts (Top 10): {total_posts}")

def on_search():
    query = search_entry.get()
    if query:
        results = search_query(query, vectorizer, tfidf_matrix, data)
        display_results(results)

def on_load_data():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        global data, vectorizer, tfidf_matrix
        data = load_data(folder_selected)
        vectorizer, tfidf_matrix, data = create_index(data)
        messagebox.showinfo("Data Loaded", f"Successfully loaded data from {folder_selected}.")

# --- MAIN GUI SETUP ---
root = tk.Tk()
root.title("Text Search & Multi-document Summation (Sumy)")

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Load Data Folder", command=on_load_data)
menu_bar.add_cascade(label="File", menu=file_menu)

default_directory_path = 'data'
data = load_data(default_directory_path)
vectorizer, tfidf_matrix, data = create_index(data)

search_frame = tk.Frame(root)
search_frame.pack(pady=20)
search_entry = tk.Entry(search_frame, font=('Arial', 14), width=40)
search_entry.pack(side='left', padx=10)

search_button = tk.Button(search_frame, text="Search", command=on_search, font=('Arial', 14))
search_button.pack(side='left')

summation_button = tk.Button(search_frame, text="Summation", command=on_summation, font=('Arial', 14))
summation_button.pack(side='left')

root.mainloop()
