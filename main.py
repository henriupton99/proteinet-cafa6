import argparse
import pandas as pd
from scipy import sparse
from joblib import dump, load
from sklearn.preprocessing import MultiLabelBinarizer
from utils import read_fasta, build_feature_matrix, KmerTfidf, make_submission_dataframe
from models import BaselineModel

def prepare_train_csv(train_terms_path, fasta_path, out_csv, top_k_go=None, sample_size=None):
    terms = pd.read_csv(train_terms_path, sep='\t')
    if top_k_go is not None:
        top_terms = terms['term'].value_counts().nlargest(top_k_go).index
        terms = terms[terms['term'].isin(top_terms)]

    go_grouped = (
        terms.groupby('EntryID')['term']
        .apply(lambda x: ' '.join(sorted(x)))
        .reset_index()
        .rename(columns={'EntryID': 'Id', 'term': 'GO_terms'})
    )

    seq_dict = read_fasta(fasta_path)
    seq_df = pd.DataFrame(list(seq_dict.items()), columns=['Id', 'sequence'])
    train_df = seq_df.merge(go_grouped, on='Id', how='inner')

    if sample_size is not None and len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"⚠️ Using a random sample of {sample_size} sequences for debug mode")

    train_df.to_csv(out_csv, index=False)
    print(f"✅ Saved merged train CSV: {out_csv}")
    return train_df

def featurize_mode(train_terms, fasta_path, out_prefix, top_k_go=None, sample_size=None):
    print("🔧 Building training feature matrix...")
    train_df = prepare_train_csv(train_terms, fasta_path, out_prefix + '_train.csv',
                                 top_k_go=top_k_go, sample_size=sample_size)
    seqs = train_df['sequence'].tolist()
    
    km = KmerTfidf(k=3, max_features=2000)
    km.fit(seqs)
    dump(km, out_prefix + '_vectorizer.joblib')
    print(f"💾 Saved Kmer vectorizer: {out_prefix}_vectorizer.joblib")

    X = build_feature_matrix(seqs, use_kmer=True, pretrained_vec=km)
    sparse.save_npz(out_prefix + '_features.npz', X)
    print(f"💾 Saved feature matrix: {out_prefix}_features.npz")
    return train_df, X

def train_mode(train_csv, feature_file, model_path, vectorizer_file):
    print("🧠 Training lightweight BaselineModel...")
    df = pd.read_csv(train_csv)
    X = sparse.load_npz(feature_file)
    y_list = [s.split() for s in df['GO_terms']]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    model = BaselineModel()
    model.fit(X, Y)
    
    dump(model, model_path)
    dump(mlb, model_path + '.mlb.joblib')
    print(f"✅ Model saved: {model_path}")
    print(f"✅ MultiLabelBinarizer saved: {model_path}.mlb.joblib")

def predict_mode(model_path, vectorizer_path, test_fasta, out_csv, top_k=20):
    print("🔮 Generating predictions...")
    model = load(model_path)
    mlb = load(model_path + '.mlb.joblib')
    km = load(vectorizer_path)

    fasta = read_fasta(test_fasta)
    ids, seqs = list(fasta.keys()), list(fasta.values())

    X = build_feature_matrix(seqs, use_kmer=True, pretrained_vec=km)
    scores = model.predict_proba(X)

    df_sub = make_submission_dataframe(ids, mlb.classes_.tolist(), scores, top_k=args.top_k, go_desc=None)
    df_sub.to_csv(out_csv, sep='\t', index=False)
    print(f"✅ Saved Kaggle-ready TSV submission ({top_k} GO terms/protein): {out_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['featurize', 'train', 'predict'], required=True)
    p.add_argument('--train_terms', help='Path to train_terms.csv')
    p.add_argument('--fasta', help='Path to train_sequences.fasta or test_sequences.fasta')
    p.add_argument('--out_prefix')
    p.add_argument('--model')
    p.add_argument('--vectorizer')
    p.add_argument('--top-go', type=int, default=100)
    p.add_argument('--sample', type=int, default=1000)
    p.add_argument('--top-k', type=int, default=20)
    args = p.parse_args()

    if args.mode == 'featurize':
        featurize_mode(args.train_terms, args.fasta, args.out_prefix,
                       top_k_go=args.top_go, sample_size=args.sample)

    elif args.mode == 'train':
        train_mode(args.out_prefix + '_train.csv',
                   args.out_prefix + '_features.npz',
                   args.model,
                   args.out_prefix + '_vectorizer.joblib')

    elif args.mode == 'predict':
        predict_mode(args.model, args.vectorizer, args.fasta, args.out_prefix + "_submission.tsv", top_k=args.top_k)
