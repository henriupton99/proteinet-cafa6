# CAFA6 : Protein Function Prediction

Baseline pipeline (updated for Kaggle CAFA-6 submission format: Id, GO_term, score)

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

## Step 1 : Features embeddings generation

```bash
uv run python main.py 
--mode featurize
--train_terms path/to/train_terms.csv
--fasta path/to/train_sequences.fasta
--out_prefix /path/to/data/test_run
--sample 1000
--top-go 100
```

# Step 2 : Model training

```bash
uv run python main.py 
--mode train
--out_prefix path/to/data/test_run
--model path/to/model.joblib
```

## Step 3 : Inference and submission

```bash
uv run python3 main.py 
--mode predict 
--model path/to/model.joblib 
--vectorizer path/to/vectorizer.joblib 
--fasta path/to/testsuperset.fasta 
--out_prefix path/to/predictions/sample_submission
--top-k 20
```

