# Semantic Plagiarism Detection Engine

A command-line plagiarism detection system developed for the **Data Mining** course at **Amirkabir University of Technology (Tehran Polytechnic)**.

The project implements and compares two widely used approaches for document similarity detection:

- **Shingling + MinHash + Locality Sensitive Hashing (LSH)**
- **TF-IDF Weighted SimHash**

The goal is to efficiently identify duplicate, near-duplicate, and plagiarized documents while comparing the trade-offs between retrieval speed and detection quality.

---

## Demo


https://github.com/user-attachments/assets/a498577f-be47-47c3-a5a6-1834cc8c9328




---

## Features

- Text preprocessing pipeline
  - Text normalization
  - Tokenization
  - Stopword removal
  - Stemming
  - Word shingling
- Exact Jaccard similarity computation
- MinHash signature generation
- Locality Sensitive Hashing (LSH) for candidate retrieval
- TF-IDF weighted SimHash implementation
- Compare two documents directly
- Retrieve similar documents from a corpus
- Evaluate retrieval performance on the PAN-PC-11 dataset
- Modular project structure for easy extension

---

## Project Structure

```text
semantic-plagiarism-engine/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── project_spec.tex
│   └── project_spec.pdf
├── data/
│   ├── sample_corpus/
│   ├── raw/
│   └── processed/
├── src/
│   └── plagiarism_engine/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── PipelineCli.py
│       ├── pipeline.py
│       ├── minhash.py
│       ├── lsh.py
│       ├── simhash.py
│       ├── dataset.py
│       ├── retrieval.py
│       ├── evaluation.py
│       └── cli.py
├── outputs/
└── tests/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/maryam2006sadeghi/semantic-plagiarism-engine.git

cd semantic-plagiarism-engine
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate it.

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

The project uses standard Python libraries together with:

- nltk
- numpy
- pandas
- pytest

The first execution automatically downloads the required NLTK resources.

---

## Command Line Interface

### Compare Two Documents

```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/source/doc_01.txt \
    --file-b data/sample_corpus/suspicious/doc_02.txt \
    --output outputs/two_file_compare.json
```

The generated JSON file contains:

- Exact Jaccard similarity
- Estimated MinHash similarity
- SimHash similarity

---

### Search for Similar Documents

```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --shingle-size 3 \
    --threshold 0.25 \
    --output outputs/candidates.csv
```

The dataset directory should contain two folders:

```text
sample_corpus/
├── source/
└── suspicious/
```

The output is a CSV file containing candidate document pairs and their similarity scores.

---

### Evaluate Retrieval Performance

```bash
python -m plagiarism_engine.cli evaluate \
    --data data/raw/pan-plagiarism-corpus-2011 \
    --candidates outputs/candidates.csv \
    --top-k 10 \
    --output outputs/metrics
```

Evaluation metrics include:

- Precision
- Recall
- F1-score
- MAP
- MRR
- Precision@k
- Recall@k
- Hit@k

---

## Interactive Mode

Running the CLI without arguments starts an interactive menu.

```bash
python -m plagiarism_engine.cli
```

The menu allows you to choose between:

1. Compare two documents
2. Search a document collection
3. Evaluate retrieval performance
4. Exit

---

## Experimental Results

Experiments were conducted using the **PAN-PC-11** plagiarism corpus.

| Metric | MinHash + LSH | SimHash |
|---------|--------------:|---------:|
| Precision | 0.4835 | 0.0099 |
| Recall | 0.0075 | 0.0099 |
| F1 Score | 0.0147 | 0.0099 |
| MAP | 0.0118 | 0.0099 |
| MRR | 0.0200 | 0.0099 |
| Precision@10 | 0.0199 | 0.0099 |
| Recall@10 | 0.0118 | 0.0099 |
| Hit@10 | 0.0200 | 0.0099 |

### Discussion

The MinHash + LSH pipeline achieved substantially higher precision by filtering unlikely document pairs before performing similarity comparisons. This significantly reduced the number of false positives.

Both approaches achieved relatively low recall on PAN-PC-11. This is expected because word-based shingles are sensitive to paraphrasing and lexical variation. Performance may improve by reducing the similarity threshold, using smaller shingle sizes, or incorporating semantic document representations.

---

## Documentation

The project report contains:

- Algorithm descriptions
- Implementation details
- Parameter selection
- Experimental evaluation
- Error analysis

Files:

```text
docs/project_spec.pdf
docs/project_spec.tex
```

---

## Running Tests

```bash
pytest tests/
```


---

## Dataset

The experiments use the **PAN-PC-11 Plagiarism Corpus**.

Due to its size, the dataset is **not included** in this repository. Download it separately and place it under:

```text
data/raw/
```

A small sample dataset is included for testing and demonstration purposes.

---

## Notes

- The core implementations of MinHash, LSH, and SimHash are written from scratch.
- The project is designed as a command-line application; no graphical interface is included.
- Large datasets and generated outputs are excluded from version control using `.gitignore`.

---

## License

This repository is intended for educational purposes as part of the Data Mining course at Amirkabir University of Technology.
