import os
import json
import nltk
import streamlit as st
from datetime import date, datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

nltk.download('punkt')
nltk.download('stopwords')

vietnamese_stopwords = {
    'và','là','của','các','có','để','theo','đã','một','không','tôi','bạn','này','cái','lúc',
    'nào','như','với','cho','hơn','sẽ','được','cùng','nhiều','từ','khi','mới','vì','lại','nên',
    'đang','chưa','mà','bị','sau','tất','những','lên'
}

LANGUAGE = "english"

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in vietnamese_stopwords]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def load_data(directory_path):
    data_list = []
    for file in os.listdir(directory_path):
        if file.endswith('.json'):
            with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                data_list.append(json.load(f))
    return data_list

def create_index(dataset):
    docs = [d['content'] for d in dataset]
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix, dataset

def convert_date_str_to_obj(d_str):
    try:
        return datetime.strptime(d_str, "%Y-%m-%d").date()
    except:
        return None

def filter_by_date(results, from_date, to_date):
    if not from_date and not to_date:
        return results
    filtered = []
    for item in results:
        raw_date = item.get('date', '')
        obj_date = convert_date_str_to_obj(raw_date)
        if obj_date:
            if from_date and to_date:
                if from_date <= obj_date <= to_date:
                    filtered.append(item)
            elif from_date and not to_date:
                if obj_date >= from_date:
                    filtered.append(item)
            elif to_date and not from_date:
                if obj_date <= to_date:
                    filtered.append(item)
        else:
            filtered.append(item)
    return filtered

def search_query(query, vectorizer, tfidf_matrix, dataset, from_d, to_d):
    q = preprocess_text(query)
    q_vec = vectorizer.transform([q])
    cos_sim = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = cos_sim.argsort()[-10:][::-1]
    output = []
    for i in top_idx:
        doc = dataset[i]
        output.append({
            'title': doc.get('title', 'No Title'),
            'author': doc.get('author', 'No Author'),
            'date': doc.get('date', 'No Date'),
            'score': cos_sim[i],
            'content': doc.get('content', '')
        })
    filtered = filter_by_date(output, from_d, to_d)
    return filtered

def summarize_text(text, sentence_count, lang=LANGUAGE):
    parser = PlaintextParser.from_string(text, Tokenizer(lang))
    summarizer = LsaSummarizer(Stemmer(lang))
    summarizer.stop_words = get_stop_words(lang)
    lines = summarizer(parser.document, sentence_count)
    s = "\n".join(str(x) for x in lines)
    if not s.strip():
        s = "Summary could not be generated."
    return s

def multi_document_summarize(items, sentence_count, lang=LANGUAGE):
    merged = "\n".join([i['content'] for i in items])
    parser = PlaintextParser.from_string(merged, Tokenizer(lang))
    summarizer = LsaSummarizer(Stemmer(lang))
    summarizer.stop_words = get_stop_words(lang)
    lines = summarizer(parser.document, sentence_count)
    s = "\n".join(str(x) for x in lines)
    if not s.strip():
        s = "Summary could not be generated."
    return s

def calculate_summation(results):
    total_word_count = sum(len(r['content'].split()) for r in results)
    total_posts = len(results)
    return total_word_count, total_posts

def get_summary_length(choice, slider_value):
    if choice == "Short":
        return 3
    elif choice == "Medium":
        return 5
    elif choice == "Long":
        return 10
    elif choice == "Custom":
        return slider_value
    return 3

def main():
    st.set_page_config(page_title="IR Search & Summarization", layout="wide")
    st.title("IR Search & Summarization")

    default_path = 'data'
    data = load_data(default_path)
    vec, matrix, data = create_index(data)

    st.write("## Search")
    query = st.text_input("Enter your search query")
    cols = st.columns(2)
    from_d = cols[0].date_input("Date From", value=None)
    to_d = cols[1].date_input("Date To", value=None)

    st.write("## Summary Configuration")
    length_choice = st.selectbox("Select Summary Length", ["Short", "Medium", "Long", "Custom"])
    slider_value = 3
    if length_choice == "Custom":
        slider_value = st.slider("Number of sentences", 1, 20, 5)

    col_search, col_summation = st.columns(2)
    if col_search.button("Search"):
        results = search_query(query, vec, matrix, data, from_d, to_d)
        if not results:
            st.warning("No documents found")
        else:
            st.subheader("Search Results")
            for idx, r in enumerate(results):
                with st.expander(f"Title: {r['title']} | Score: {r['score']:.4f}", expanded=False):
                    st.write(f"Author: {r['author']}")
                    st.write(f"Date: {r['date']}")
                    snippet = r['content'][:150] + '...' if len(r['content']) > 150 else r['content']
                    st.write(snippet)
                    if st.button(f"Read more {idx}"):
                        short_read = r['content'][:300] + '...' if len(r['content']) > 300 else r['content']
                        st.info(short_read)
                    if st.button(f"Summarize {idx}"):
                        cnt = get_summary_length(length_choice, slider_value)
                        summary = summarize_text(r['content'], cnt)
                        st.info(summary)
                        st.write(f"Word count: {len(summary.split())}")

    if col_summation.button("Summation"):
        results = search_query(query, vec, matrix, data, from_d, to_d)
        if results:
            total_words, total_posts = calculate_summation(results)
            st.info(f"Total Word Count (Top 10): {total_words} | Total Posts (Top 10): {total_posts}")
            cnt = get_summary_length(length_choice, slider_value)
            full_summary = multi_document_summarize(results, cnt)
            st.markdown(f"**Combined Summary:**\n{full_summary}")
            st.write(f"Word count: {len(full_summary.split())}")
        else:
            st.warning("No documents found for summation")

if __name__ == '__main__':
    main()
