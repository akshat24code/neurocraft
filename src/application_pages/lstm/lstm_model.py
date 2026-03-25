import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = ROOT / "src" / "assets"
DEFAULT_OUTPUT = ASSETS_DIR / "lstm_model.pth"


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze(1)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


def pad_sequence(tokens: list[int], max_len: int) -> list[int]:
    tokens = tokens[:max_len]
    return tokens + [0] * (max_len - len(tokens))


def load_assets() -> tuple[dict, dict]:
    with open(ASSETS_DIR / "word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(ASSETS_DIR / "config.pkl", "rb") as f:
        config = pickle.load(f)
    return word2idx, config


def build_dataloader(csv_path: str, text_col: str, label_col: str, batch_size: int):
    word2idx, config = load_assets()

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


def train_model(dataset: str, text_col: str, label_col: str, epochs: int, batch_size: int, lr: float, output: str, cpu: bool):
    dataloader, vocab_size = build_dataloader(dataset, text_col, label_col, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")

    model = LSTMModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment checkpoint for NeuroCraft.")
    parser.add_argument("--dataset", required=True, help="CSV path with text and binary label columns.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Binary label column name.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    train_model(
        dataset=args.dataset,
        text_col=args.text_col,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output=args.output,
        cpu=args.cpu,
    )
