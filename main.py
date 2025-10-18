import argparse
import pandas as pd
import numpy as np
from joblib import load, dump
from scipy import sparse

from utils import read_fasta, build_feature_matrix, make_submission_dataframe, esm_batch_embed_hf
from models import BaselineModel, MultiLabelBinarizer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['inspect','featurize','embed','train','predict'], required=True)
    p.add_argument('--train')
    p.add_argument('--test')
    p.add_argument('--seqs')
    p.add_argument('--features')
    p.add_argument('--model')
    p.add_argument('--out', default='submission.csv')
    p.add_argument('--top-k', type=int, default=100, help='number of top GO terms per protein to include in submission')
    args = p.parse_args()
    if args.mode == 'inspect':
        if args.train:
            df = pd.read_csv(args.train)
            print("Train head:\n", df.head())
            print("Number of sequences:", len(df))
        if args.test:
            df = pd.read_csv(args.test)
            print("Test head:\n", df.head())
            
    if args.mode == 'featurize':
        if args.train and args.seqs:
            # load GO terms CSV
            terms = pd.read_csv(args.train, sep='\t') # train_terms.csv
            go_grouped = terms.groupby('EntryID')['term'].apply(lambda x: ' '.join(sorted(x))).reset_index()
            go_grouped.rename(columns={'EntryID':'Id', 'term':'GO_terms'}, inplace=True)

            # Load FASTA train sequences
            seq_dict = read_fasta(args.seqs)
            seq_df = pd.DataFrame(list(seq_dict.items()), columns=['Id','sequence'])

            # merge GO terms infos with fasta sequences to get in one csv file :
            # - Id , sequence, GO terms ' ' separated
            train_df = seq_df.merge(go_grouped, on='Id')
            train_csv = args.out or 'train_for_pipeline.csv'
            train_df.to_csv(train_csv, index=False)
            print('Saved CSV for training:', train_csv)

            # build feature matrix
            seqs = train_df['sequence'].tolist()
            X = build_feature_matrix(seqs, use_kmer=True)
            feat_file = train_csv.replace('.csv', '_features.npz')
            sparse.save_npz(feat_file, X)
            print('Saved training features:', feat_file)
            
    if args.mode == 'embed':
        if args.seqs:
            fasta = read_fasta(args.seqs)
            ids, seqs = list(fasta.keys()), list(fasta.values())
            emb = esm_batch_embed_hf(seqs)
            np.save(args.out or 'embeddings.npy', emb)
            print("Saved embeddings:", args.out or 'embeddings.npy')
    if args.mode == 'train':
        if args.train:
            df = pd.read_csv(args.train)
            seqs = df['sequence'].tolist()
            X = build_feature_matrix(seqs, use_kmer=True)
            y_list = [s.split() for s in df['GO_terms']]
            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform(y_list)
            model = BaselineModel()
            model.fit(X, Y)
            dump(model, args.model)
            dump(mlb, args.model + '.mlb.joblib')
            print("Saved trained model:", args.model)
    if args.mode == 'predict':
        model = load(args.model)
        mlb = load(args.model + '.mlb.joblib')
        if args.test.endswith('.fasta'):
            fasta = read_fasta(args.test)
            ids, seqs = list(fasta.keys()), list(fasta.values())
        else:
            df = pd.read_csv(args.test)
            ids, seqs = df['Id'].tolist(), df['sequence'].tolist()
        X = build_feature_matrix(seqs, use_kmer=True)
        scores = model.predict_proba(X)
        df_sub = make_submission_dataframe(ids, mlb.classes_.tolist(), scores, top_k=args.top_k)
        df_sub.to_csv(args.out, index=False)
        print(f'Saved Kaggle-ready submission (flat format with top {args.top_k}):', args.out)
