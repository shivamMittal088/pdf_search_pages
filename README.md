# PDF Search Engine
This is a Python-based PDF search engine that allows you to search for relevant documents based on a query. The search engine utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to calculate the relevance scores between the query and the documents in a given directory.

## Prerequisites
Make sure you have the following dependencies installed:

- Python 3.x
- PyPDF2
- NLTK (Natural Language Toolkit)
- scikit-learn (sklearn)
- tqdm

You can install the required packages by running the following command:
```commandline
pip install -r requirements.txt
```

## Usage
### 1. Data Preparation
Place your PDF documents in a directory of your choice (e.g., "docs").
The search engine will extract text from the PDFs for indexing and searching.
### 2. Indexing
To build the TF-IDF index for the PDF documents, run the following command:
```commandline
python main.py --docs <path_to_docs_directory> --index <index_filename.pkl> --update-index
```
- `<path_to_docs_directory>`: Path to the directory containing the PDF documents.
- `<index_filename.pkl>`: Name of the index file to be created or updated.
- `--update-index` (optional): Use this flag to update the index if it already exists.

The indexing process may take some time, depending on the number and size of the PDF documents. The progress will be displayed with a progress bar.
### 3. Searching
Once the index is built, you can perform searches using the following command:
```commandline
python pdf_search.py --index <index_filename.pkl>
```

- `<index_filename.pkl>`: Path to the index file created during the indexing step.
Enter your query when prompted. The search engine will display the top 5 most relevant documents based on your query.

To exit the search engine, simply enter 'q' when prompted for a query.

## Additional Notes
- The search engine preprocesses the documents by tokenizing the text, removing punctuation, converting text to lowercase, removing stopwords, and performing stemming using the Porter stemming algorithm.
- The search engine uses cosine similarity to calculate the relevance scores between the query and the documents. The documents with the highest relevance scores are considered the most relevant.
- The search engine saves the index data (TF-IDF matrix, vectorizer, document paths, and titles) as a pickle file for faster loading in subsequent runs. The index file can be updated by using the `--update-index` flag during the indexing step.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.










conda activate pdfsearch312

python main.py --mode doc2vec --update-index --docs docs --index index_data_doc2vec.pkl
