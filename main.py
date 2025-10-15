import os
import glob
import pickle
import argparse
from tqdm import tqdm
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from textblob import TextBlob
import logging
from pdfminer.high_level import extract_text
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# ==========================================================
# 🧩 Ensure NLTK data (fixes lookup errors automatically)
# ==========================================================
def ensure_nltk_data():
    """Ensure that all required NLTK data packages are downloaded."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_data()

PERCENTAGE_THRESHOLD = 0.1
TOP_DOCUMENTS = 5


# ==========================================================
# 📘 PDF Processor
# ==========================================================
class PDFProcessor:
    """Handles PDF extraction and text preprocessing."""

    @staticmethod
    def extract_text_by_page(file_path):
        """Extracts text content page by page from a PDF file."""
        texts = []
        try:
            full_text = extract_text(file_path).replace("\n", " ")
            texts = [text for text in full_text.split('\f')]
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {e}")
        return texts

    @staticmethod
    def preprocess(text):
        """Preprocesses a given text."""
        tokens = [token.lower() for token in word_tokenize(text) if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(token, PDFProcessor._get_wordnet_pos(pos_tag))
            for token, pos_tag in pos_tags
        ]
        return ' '.join(tokens)

    @staticmethod
    def _get_wordnet_pos(tag):
        """Maps POS tag to first character used by WordNetLemmatizer."""
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)


# ==========================================================
# 🧠 Doc2Vec Processor
# ==========================================================
class Doc2VecProcessor:
    """Handles Doc2Vec related functionalities."""

    @staticmethod
    def train_doc2vec_model(docs, vector_size=150, window=5, min_count=2, workers=4, epochs=50):
        """Train a Doc2Vec model with the provided documents."""
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
        model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)
        return model

    @staticmethod
    def infer_vector(model, doc):
        """Infer vector for a document using a trained Doc2Vec model."""
        return model.infer_vector(word_tokenize(doc.lower()))


# ==========================================================
# 🏗️ Index Builder
# ==========================================================
class IndexBuilder:
    """Handles index building operations."""

    def __init__(self, mode="tfidf"):
        self.mode = mode

    def _process_file(self, file_path):
        """Process a single file by extracting and preprocessing its text page by page."""
        pages = PDFProcessor.extract_text_by_page(file_path)
        processed_data = []
        for page_idx, page_text in enumerate(pages):
            processed_page = PDFProcessor.preprocess(page_text)
            sentiment = TextBlob(page_text).sentiment.polarity
            processed_data.append({'text': processed_page, 'sentiment': sentiment})
        return processed_data

    def build(self, directory_path, batch_size=1000):
        """Builds the index from a given directory of PDF files page by page."""
        file_paths = glob.glob(os.path.join(directory_path, '*.pdf'))
        processed_pages = []
        document_pages = []

        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i: i + batch_size]
            with Pool(cpu_count()) as pool:
                processed_pages_data = list(tqdm(pool.imap(self._process_file, batch_paths), total=len(batch_paths)))
            for file_idx, data in enumerate(processed_pages_data):
                for page_idx, processed_page_data in enumerate(data):
                    processed_pages.append(processed_page_data['text'])
                    document_pages.append((batch_paths[file_idx], page_idx, processed_page_data['sentiment']))

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_pages)

        data = {'vectorizer': vectorizer, 'document_pages': document_pages}
        if self.mode == "lsi":
            lsi_model = TruncatedSVD(n_components=50)
            data['lsi_matrix'] = lsi_model.fit_transform(tfidf_matrix)
            data['lsi_model'] = lsi_model
        elif self.mode == "doc2vec":
            d2v_processor = Doc2VecProcessor()
            self.d2v_model = d2v_processor.train_doc2vec_model(processed_pages)
            data['d2v_model'] = self.d2v_model
            data['document_vectors'] = [self.d2v_model.dv[i] for i in range(len(processed_pages))]
        else:
            data['tfidf_matrix'] = tfidf_matrix

        return data


# ==========================================================
# 🔍 Search Engine
# ==========================================================
class SearchEngine:
    """Handles search functionalities."""

    def __init__(self, index_data, mode):
        self.mode = mode
        self.data = index_data
        if self.mode == 'doc2vec':
            self.d2v_model = index_data['d2v_model']

    def query(self, text, top_k=10):
        """Queries the search engine and retrieves relevant pages."""
        preprocessed_query = PDFProcessor.preprocess(text)

        if self.mode == "lsi":
            query_vector = self.data['vectorizer'].transform([preprocessed_query])
            lsi_query_vector = self.data['lsi_model'].transform(query_vector)
            similarities = cosine_similarity(self.data['lsi_matrix'], lsi_query_vector).flatten()
        elif self.mode == "doc2vec":
            query_vector = Doc2VecProcessor.infer_vector(self.d2v_model, preprocessed_query)
            scores = cosine_similarity([query_vector], self.data['document_vectors'])
            similarities = scores[0]
        else:
            query_vector = self.data['vectorizer'].transform([preprocessed_query])
            similarities = cosine_similarity(self.data['tfidf_matrix'], query_vector).flatten()

        top_indices = similarities.argsort()[:-top_k - 1:-1]
        scores = similarities[top_indices]

        paths = [
            (self.data['document_pages'][index][0],
             self.data['document_pages'][index][1],
             self.data['document_pages'][index][2])
            for index in top_indices
        ]

        # --- aggregate doc-level scores ---
        total_pages_per_doc = {}
        for doc, page, sentiment in self.data['document_pages']:
            total_pages_per_doc[doc] = total_pages_per_doc.get(doc, 0) + 1

        doc_similarity_aggregate = {}
        for index in similarities.argsort()[:-int(PERCENTAGE_THRESHOLD * len(similarities)) - 1:-1]:
            doc_path, page, sentiment = self.data['document_pages'][index]
            doc_similarity_aggregate[doc_path] = doc_similarity_aggregate.get(doc_path, {'score': 0, 'sentiment_sum': 0})
            doc_similarity_aggregate[doc_path]['score'] += similarities[index]
            doc_similarity_aggregate[doc_path]['sentiment_sum'] += sentiment

        normalized_similarity = {}
        for doc, agg in doc_similarity_aggregate.items():
            normalized_similarity[doc] = {
                'score': agg['score'] / total_pages_per_doc[doc],
                'average_sentiment': agg['sentiment_sum'] / total_pages_per_doc[doc]
            }

        sorted_docs = sorted(normalized_similarity.items(), key=lambda kv: kv[1]['score'], reverse=True)
        return paths, scores, sorted_docs[:TOP_DOCUMENTS]


# ==========================================================
# 💾 Utility Functions
# ==========================================================
def save_index(index_file, data):
    with open(index_file, 'wb') as f:
        pickle.dump(data, f)

def load_index(index_file):
    with open(index_file, 'rb') as f:
        return pickle.load(f)

def get_multiline_input(prompt, end_keyword="END"):
    print(prompt, f"(Type '{end_keyword}' on a new line to finish)")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == end_keyword:
                break
            lines.append(line)
        except EOFError:
            break
    return '\n'.join(lines)


# ==========================================================
# 🚀 Main Function
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Build an index and search PDFs.")
    parser.add_argument('--index', type=str, default='index_data.pkl', help='Path to the index file.')
    parser.add_argument('--docs', type=str, default='docs', help='Path to the directory containing PDF documents.')
    parser.add_argument('--update-index', action='store_true', help='Update the index if it already exists.')
    parser.add_argument('--mode', type=str, choices=['tfidf', 'lsi', 'doc2vec'], default='tfidf',
                        help='The indexing and search mode.')
    args = parser.parse_args()

    # Load or build index
    if os.path.exists(args.index) and not args.update_index:
        print("Loading existing index...")
        index_data = load_index(args.index)
    else:
        print("Building new index...")
        indexer = IndexBuilder(args.mode)
        index_data = indexer.build(args.docs)
        save_index(args.index, index_data)
        print(f"Index saved to {args.index}")

    search_engine = SearchEngine(index_data, args.mode)

    while True:
        query = get_multiline_input("Enter your search query")

        if not query.strip():
            print("Empty query, please try again or type 'exit' to stop.")
            continue
        if query.strip().lower() == 'exit':
            break

        paths, scores, sorted_docs = search_engine.query(query)

        # ✅ Display results
        if not paths:
            print("No results found.\n")
        else:
            print("\nTop Pages:")
            for rank, ((doc_path, page_idx, sentiment), score) in enumerate(zip(paths, scores), start=1):
                print(f"{rank}. {os.path.basename(doc_path)} - Page {page_idx + 1} | Score: {float(score):.4f} | Sentiment: {sentiment:+.3f}")

            print("\nTop Documents:")
            for rank, (doc_path, agg) in enumerate(sorted_docs, start=1):
                print(f"{rank}. {os.path.basename(doc_path)} | Normalized Score: {agg['score']:.4f} | Avg Sentiment: {agg['average_sentiment']:+.3f}")
            print()


if __name__ == "__main__":
    main()
