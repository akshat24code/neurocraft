import argparse
import pickle
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "assets" / "rnn" / "next_word"
SEQUENCE_LENGTH = 5


class LSTMNextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def build_vocab(tokens, min_freq):
    counts = Counter(tokens)
    vocab = ["<UNK>"] + [token for token, freq in counts.items() if freq >= min_freq]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def load_corpus(path):
    text = Path(path).read_text(encoding="utf-8")
    return text.lower().split()


def build_dataset(tokens, word_to_idx):
    sequences = []
    targets = []
    for idx in range(SEQUENCE_LENGTH, len(tokens)):
        context = tokens[idx - SEQUENCE_LENGTH:idx]
        target = tokens[idx]
        sequences.append([word_to_idx.get(token, 0) for token in context])
        targets.append(word_to_idx.get(target, 0))

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(x, y)


def train(args):
    tokens = load_corpus(args.corpus)
    word_to_idx, idx_to_word = build_vocab(tokens, args.min_freq)
    dataset = build_dataset(tokens, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = LSTMNextWordModel(len(word_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

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

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with open(ASSET_DIR / "vocab.pkl", "wb") as f:
        pickle.dump((word_to_idx, idx_to_word), f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM next-word checkpoint for NeuroCraft.")
    parser.add_argument("--corpus", required=True, help="Path to a plain text corpus file.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=str(ASSET_DIR / "lstm_wikitext2.pth"))
    train(parser.parse_args())
