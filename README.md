# 📚 PDF Semantic Similarity Project

This project was developed in Python to calculate **semantic similarity scores** between academic PDF documents.

---

## 🎯 Project Purpose

The primary purpose of this project is to calculate content-based **semantic similarity** scores between academic PDF articles written on different topics and identify documents with similar content.

It provides a foundational infrastructure that can be used in areas such as:

* Document clustering and classification
* Semantic search engine development
* Academic content analysis
* Similar document detection
* Information retrieval systems

---

## 🌟 Features

* ✅ Text extraction from PDF documents
* 🧠 Text vectorization using a BERT-based language model
* 📊 Content similarity calculation using cosine similarity
* 🗃️ Storage of results in an SQLite database
* 📂 Organized data management through a PDF silo structure
* 🖥️ Simple command-line usage

---

## 🛠️ Technologies Used

* 🐍 Python 3
* 🤗 `all-MiniLM-L6-v2` language model from Hugging Face
* 🧰 Python libraries:

  * `sentence-transformers`
  * `PyMuPDF` (`fitz`)
  * `sqlite3`

---

## 📁 Project Structure

```text
PDF-similarity/
├── main.py
│   └── Creates the database, adds PDF files to the database,
│       and starts the similarity calculation process
│
├── PDF_Similarity.py
│   └── Processes PDF contents and compares extracted texts
│
├── Similarity_scores.py
│   └── Calculates similarity scores and stores them in the database
│
├── PDF/
│   └── Directory containing the PDF documents
│
└── pdf_silo.db
    └── SQLite database containing PDF information and similarity results
```

---

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/betulkizilkaya/PDF-similarity.git
cd PDF-similarity
```

### 2. Add PDF Documents

Place the PDF documents you want to compare inside the `PDF/` directory.

### 3. Install the Required Libraries

```bash
pip install sentence-transformers pymupdf
```

### 4. Run the Project

```bash
python main.py
```

---

## 🧠 Output

* Similarity scores between PDF documents are displayed in the terminal.
* All calculated results are stored in the `similarity_scores` table inside the `pdf_silo.db` SQLite database.

The similarity scores are calculated using cosine similarity between the semantic vector representations of the extracted PDF texts.

---

## 🔍 How It Works

1. PDF documents are read from the `PDF/` directory.
2. Text content is extracted from each PDF using PyMuPDF.
3. Extracted texts are converted into semantic vector representations using the `all-MiniLM-L6-v2` Sentence Transformer model.
4. Cosine similarity is calculated between document vectors.
5. The calculated similarity scores are displayed in the terminal.
6. The results are saved to the SQLite database.

---

## 📌 Example Workflow

```text
PDF Documents
      ↓
Text Extraction
      ↓
Sentence Embeddings
      ↓
Cosine Similarity
      ↓
Similarity Scores
      ↓
SQLite Database
```

---

## 📄 License

This project is licensed under the MIT License.

© 2025 [Betül Kızılkaya](https://github.com/betulkizilkaya)

For more information, see the [LICENSE](LICENSE) file.
