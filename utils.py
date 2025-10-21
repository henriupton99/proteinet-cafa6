from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
    
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def read_fasta(fasta_path):
    seqs = {}
    cur_id = None
    cur_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id and cur_seq:
                    seqs[cur_id] = ''.join(cur_seq)
                header = line[1:].strip()
                
                if '|' in header:
                    parts = header.split('|')
                    if len(parts) > 1:
                        cur_id = parts[1]
                    else:
                        cur_id = parts[0]
                else:
                    cur_id = header.split()[0] 
                cur_seq = []
            else:
                cur_seq.append(line)
                
    if cur_id and cur_seq:
        seqs[cur_id] = ''.join(cur_seq)

    return seqs

def aa_composition(seq: str) -> np.ndarray:
    seq = seq.upper()
    vec = np.zeros(len(AMINO_ACIDS))
    L = len(seq)
    for i, aa in enumerate(AMINO_ACIDS):
        vec[i] = seq.count(aa) / L if L > 0 else 0
    return vec

def sequence_length_feature(seq: str) -> np.ndarray:
    return np.array([len(seq)])

def kmerize(seq: str, k: int) -> List[str]:
    seq = seq.upper()
    return [seq[i:i+k] for i in range(len(seq)-k+1)] if len(seq) >= k else []
  
class KmerTfidf:
    def __init__(self, k=3, max_features=20000):
        self.k = k
        self.vec = TfidfVectorizer(analyzer=self.kmer_analyzer, max_features=max_features)

    def kmer_analyzer(self, seq):
        seq = seq.upper()
        if len(seq) < self.k:
            return []
        return [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]

    def fit(self, seqs: List[str]):
        self.vec.fit(seqs)
        return self

    def transform(self, seqs: List[str]):
        return self.vec.transform(seqs)

def esm_batch_embed_hf(seqs: List[str], model_name='facebook/esm2_t6_8M_UR50D', batch_size=32, device=None) -> np.ndarray:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        HAS_TF = True
    except Exception:
        HAS_TF = False
    if not HAS_TF:
        raise ImportError('transformers and torch are required for embedding extraction.')
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            toks = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            hid = out.last_hidden_state
            mask = toks.get('attention_mask', None)
            if mask is None:
                seq_emb = hid.mean(dim=1)
            else:
                seq_emb = (hid * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            all_emb.append(seq_emb.cpu().numpy())
    return np.vstack(all_emb)


def make_submission_dataframe(ids: List[str], terms: List[str], scores: np.ndarray, top_k: int=100) -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(ids):
        top_idx = np.argsort(scores[i])[::-1][:top_k]
        for j in top_idx:
            rows.append({'Id': pid, 'GO_term': terms[j], 'score': float(scores[i, j])})
    return pd.DataFrame(rows, columns=['Id','GO_term','score'])

def build_feature_matrix(
    seqs: List[str],
    use_kmer=True,
    k=3,
    max_kmer_features=20000,
    pretrained_vec=None
    ):
    
    comp = np.vstack([sequence_length_feature(s) for s in seqs])
    aa = np.vstack([aa_composition(s) for s in seqs])
    X_base = np.hstack([comp, aa])
    
    if use_kmer:
        if pretrained_vec is not None:
            Xk = pretrained_vec.transform(seqs)
        else:
            km = KmerTfidf(k=k, max_features=max_kmer_features)
            km.fit(seqs)
            Xk = km.transform(seqs)

        X = hstack([csr_matrix(X_base), Xk])
    else:
        X = csr_matrix(X_base)

    return X
