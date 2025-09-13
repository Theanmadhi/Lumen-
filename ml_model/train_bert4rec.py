# ml_model/train_bert4rec.py
import os
import ast
import json
import argparse
from typing import List, Dict, Tuple
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Data utilities
# ---------------------------
def safe_parse_prev(s):
    # Handle missing values
    if s is None:
        return []
    if isinstance(s, float) and pd.isna(s):
        return []
    # If already a list, just return it
    if isinstance(s, (list, tuple)):
        return list(s)
    # If it's a string (e.g., JSON), try parsing
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return []
    return []


def load_json(path: str) -> pd.DataFrame:
    """
    Loads either JSON array or JSONL (one JSON object per line) into a DataFrame.
    Ensures 'previous_products' is always a list.
    """
    try:
        # Try JSON lines first
        df = pd.read_json(path, lines=True)
    except ValueError:
        # Fallback: normal JSON (list or dict)
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # dict of users -> list of products
            rows = []
            for user, items in data.items():
                for entry in items:
                    rows.append({"user_id": user, **entry})
            df = pd.DataFrame(rows)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    
    # Ensure previous_products column exists
    if 'previous_products' not in df.columns:
        df['previous_products'] = [[] for _ in range(len(df))]
    else:
        df['previous_products'] = df['previous_products'].apply(safe_parse_prev)

    return df

def build_item_mapping(dfs: List[pd.DataFrame]) -> Tuple[Dict[int,int], Dict[int,int]]:
    # collect unique product ids from previous_products and current_product_id
    items = set()
    for df in dfs:
        for seq in df['previous_products']:
            items.update([int(x) for x in seq if x is not None])
        # include current_product_id if present and >0
        if 'current_product_id' in df.columns:
            items.update([int(x) for x in df['current_product_id'].dropna().unique() if x != 0])
    items = sorted(items)
    # mapping: PAD=0, items->1..N, MASK = N+1
    item2idx = {item: idx+1 for idx, item in enumerate(items)}
    idx2item = {v:k for k,v in item2idx.items()}
    return item2idx, idx2item

# ---------------------------
# Dataset with masking (BERT4Rec style)
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, item2idx: Dict[int,int], max_len=50, mask_prob=0.15):
        self.rows = df.to_dict('records')
        self.item2idx = item2idx
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_items = len(item2idx)
        self.MASK_ID = self.num_items + 1  # special mask index
        self.PAD_ID = 0

    def encode_seq(self, seq: List[int]) -> List[int]:
        # map product ids to indices, drop unknowns
        ids = [self.item2idx.get(int(x), None) for x in seq if x is not None]
        ids = [i for i in ids if i is not None]
        # truncate/pad (right padding)
        if len(ids) > self.max_len:
            ids = ids[-self.max_len:]
        pad_len = self.max_len - len(ids)
        return ids + [self.PAD_ID] * pad_len

    def mask_sequence(self, seq_ids: List[int]) -> Tuple[List[int], List[int]]:
        # seq_ids length == max_len
        labels = [-100] * self.max_len  # -100 to be ignored by CE
        input_ids = seq_ids.copy()
        nonpad_positions = [i for i, v in enumerate(seq_ids) if v != self.PAD_ID]
        # sample positions to mask (roughly mask_prob of non-pad tokens)
        n_to_mask = max(1, int(round(len(nonpad_positions) * self.mask_prob))) if nonpad_positions else 0
        mask_positions = random.sample(nonpad_positions, n_to_mask) if n_to_mask > 0 else []
        for pos in mask_positions:
            labels[pos] = seq_ids[pos]  # original id
            input_ids[pos] = self.MASK_ID  # replace with MASK token index
        return input_ids, labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        seq = r.get('previous_products', []) or []
        encoded = self.encode_seq(seq)
        input_ids, labels = self.mask_sequence(encoded)
        attention_mask = [1 if x != self.PAD_ID else 0 for x in encoded]
        sample = {
            'input_ids': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(attention_mask),
            'current_product_id': r.get('current_product_id', 0)
        }
        return sample

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    current_products = [b['current_product_id'] for b in batch]
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'current_products': current_products}

# ---------------------------
# BERT4Rec model (Transformer encoder)
# ---------------------------
class BERT4Rec(nn.Module):
    def __init__(self, num_items:int, embed_dim=128, max_len=50, n_layers=2, n_heads=4, ffn_dim=256, dropout=0.1):
        """
        num_items: number of unique items (actual items). Model reserves:
          PAD=0, items->1..num_items, MASK=num_items+1
        """
        super().__init__()
        self.num_items = num_items
        self.vocab_size = num_items + 1   # 0..num_items -> outputs (0 is PAD)
        self.emb_size = embed_dim
        # embedding table needs to include the MASK token index (num_items+1)
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, L)
        B, L = input_ids.size()
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.dropout(x).transpose(0,1)  # (L, B, E) for transformer
        src_key_padding_mask = (input_ids == 0)  # True where PAD -> ignores those positions
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (L, B, E)
        out = out.transpose(0,1)  # (B, L, E)
        out = self.layernorm(out)
        # project to item logits using tied weights (item_embedding weights for indices 0..num_items)
        emb_weight = self.item_embedding.weight[:self.vocab_size]  # (vocab_size, E)
        logits = torch.matmul(out, emb_weight.t())  # (B, L, vocab_size)
        return logits

# ---------------------------
# Training & evaluation
# ---------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="train"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)  # -100 where ignored
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)  # (B, L, V)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B*L, V), labels.view(B*L), ignore_index=-100)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_next_item(model, df_test: pd.DataFrame, item2idx: Dict[int,int], idx2item: Dict[int,int],
                   device, max_len=50, k_list=(5,10,20)):
    """
    For evaluation we use current_product_id as 'ground truth' next item.
    Only evaluate rows where current_product_id>0 and the item is in mapping.
    """
    model.eval()
    M = len(df_test)
    hits = {k: 0 for k in k_list}
    mrr_sum = {k: 0.0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}
    total = 0
    with torch.no_grad():
        for _, row in df_test.iterrows():
            cur = int(row.get('current_product_id', 0) or 0)
            if cur == 0 or cur not in item2idx:
                continue  # skip if no ground truth
            seq = safe_parse_prev(row.get('previous_products', []))
            seq_idx = [item2idx.get(int(x)) for x in seq if int(x) in item2idx]
            # truncate to max_len
            if len(seq_idx) > max_len:
                seq_idx = seq_idx[-max_len:]
            pad_len = max_len - len(seq_idx)
            input_ids = seq_idx + [0] * pad_len
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # (1, L)
            attention_mask = (input_ids != 0).long().to(device)
            logits = model(input_ids, attention_mask)  # (1, L, V)
            # use the last non-pad position for prediction; if no non-pad (cold user) take position 0
            lengths = int((attention_mask==1).sum().item())
            last_pos = max(0, lengths - 1)
            scores = logits[0, last_pos]  # (V,)
            # topk
            topk = torch.topk(scores, max(k_list)).indices.cpu().numpy().tolist()
            total += 1
            for k in k_list:
                topk_k = topk[:k]
                gt = item2idx[cur]  # index in vocab
                if gt in topk_k:
                    hits[k] += 1
                    # MRR contribution:
                    rank = topk_k.index(gt) + 1
                    mrr_sum[k] += 1.0 / rank
                    # DCG contribution:
                    ndcg_sum[k] += 1.0 / np.log2(rank + 1)
    # compute metrics
    metrics = {}
    for k in k_list:
        if total == 0:
            metrics[f'HR@{k}'] = 0.0
            metrics[f'MRR@{k}'] = 0.0
            metrics[f'NDCG@{k}'] = 0.0
        else:
            metrics[f'HR@{k}'] = hits[k] / total
            metrics[f'MRR@{k}'] = mrr_sum[k] / total
            metrics[f'NDCG@{k}'] = ndcg_sum[k] / total
    return metrics, total

# ---------------------------
# Main
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df = load_json(args.train)
    test_df = load_json(args.test)

    # mapping
    item2idx, idx2item = build_item_mapping([train_df, test_df])
    print("Unique items:", len(item2idx))

    # datasets
    train_ds = SeqDataset(train_df, item2idx, max_len=args.max_len, mask_prob=args.mask_prob)
    test_ds = SeqDataset(test_df, item2idx, max_len=args.max_len, mask_prob=args.mask_prob)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # init model
    model = BERT4Rec(num_items=len(item2idx), embed_dim=args.embed_dim, max_len=args.max_len,
                     n_layers=args.n_layers, n_heads=args.n_heads, ffn_dim=args.ffn_dim, dropout=args.dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_hr = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {loss:.4f}")
        metrics, total = eval_next_item(model, test_df, item2idx, idx2item, device, max_len=args.max_len, k_list=(5,10,20))
        print(f"Evaluated on {total} users with current_product_id>0")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}")
        # save best
        if metrics['HR@10'] > best_hr:
            best_hr = metrics['HR@10']
            out = {'model_state': model.state_dict(), 'item2idx': item2idx, 'idx2item': idx2item, 'args': vars(args)}
            torch.save(out, args.save_path)
            print("Saved best model to", args.save_path)

    print("Training done. Best HR@10:", best_hr)

    # example inference helper
    print("\nExample: get top-10 for first user in test set (if available):")
    if len(test_df) > 0:
        example_seq = safe_parse_prev(test_df.iloc[0].get('previous_products', []))
        recs = recommend_for_sequence(model, example_seq, item2idx, idx2item, device, max_len=args.max_len, top_k=10)
        print("Recommendations (item_id list):", recs)

# ---------------------------
# Inference helper
# ---------------------------
def recommend_for_sequence(model, seq: List[int], item2idx:Dict[int,int], idx2item:Dict[int,int], device=None, max_len=50, top_k=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    seq_idx = [item2idx.get(int(x)) for x in seq if int(x) in item2idx]
    if len(seq_idx) > max_len:
        seq_idx = seq_idx[-max_len:]
    pad_len = max_len - len(seq_idx)
    input_ids = seq_idx + [0]*pad_len
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    attention_mask = (input_ids != 0).long().to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # (1, L, V)
        lengths = int((attention_mask==1).sum().item())
        last_pos = max(0, lengths - 1)
        scores = logits[0, last_pos].cpu().numpy()
    topk_idx = list(np.argsort(-scores)[:top_k])
    # map back to original item ids (remember our indices are 1..N mapping)
    inv_idx = {v:k for k,v in item2idx.items()}
    recs = [inv_idx.get(idx, None) for idx in topk_idx if idx in inv_idx]
    return recs

# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/train_data/train_subscriptions.json')
    parser.add_argument('--test', type=str, default='data/test_data/test_subscriptions.json')
    parser.add_argument('--save_path', type=str, default='bert4rec_best.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
