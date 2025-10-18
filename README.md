# CAFA6 : Protein Function Prediction

CAFA6 -- Baseline pipeline (updated for Kaggle CAFA-6 submission format: Id, GO_term, score)

This baseline now produces submission files matching the official CAFA-6 requirements:
Each line = one (protein, GO_term) pair with an associated probability.

Output CSV format:

```csv
Id,GO_term,score
P9WHI7,GO:0009274,0.931
P9WHI7,GO:0071944,0.540
P04637,GO:0043565,0.640
...
```

Includes:
* Data parsing (FASTA/CSV)
* Feature extraction (AA composition, k-mer TF-IDF)
* Embedding extraction (ESM/ProtBert via HuggingFace)
* Baseline Logistic Regression model
* Top-K flat submission output

### 1. feature extraction from GO terms and FASTA sequences

```bash
uv run python3 main.py --mode featurize --train ./data/train/train_terms.tsv --seqs ./data/train/train_sequences.fasta --out ./data/train/train_for_pipeline.csv
```

### 2. Model training based on embeddings of step 1

TBU

### 3. Inference based on model of step 2

TBU

