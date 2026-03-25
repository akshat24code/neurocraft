import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "src" / "assets"


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled, _ = torch.max(out, dim=1)
        out = self.dropout(pooled)
        return self.fc(out).squeeze(1)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


def pad_sequence(tokens, max_len):
    tokens = tokens[:max_len]
    return tokens + [0] * (max_len - len(tokens))


def build_dataloader(csv_path, text_col, label_col, batch_size):
    with open(ASSETS_DIR / "word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(ASSETS_DIR / "config.pkl", "rb") as f:
        config = pickle.load(f)

    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    texts = df[text_col].map(clean_text).tolist()
    labels = df[label_col].astype(float).tolist()

    sequences = []
    for text in texts:
        token_ids = [word2idx.get(token, 1) for token in text.split()]
        sequences.append(pad_sequence(token_ids, config["max_len"]))

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), config["vocab_size"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataloader, vocab_size = build_dataloader(args.dataset, args.text_col, args.label_col, args.batch_size)

    model = LSTMClassifier(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment model checkpoint for NeuroCraft.")
    parser.add_argument("--dataset", required=True, help="CSV path with text and binary label columns.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Binary label column name.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=str(ASSETS_DIR / "lstm_model.pth"))
    train(parser.parse_args())
