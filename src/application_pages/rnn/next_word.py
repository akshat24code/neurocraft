import pickle
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parents[2]
ASSET_DIR = BASE_DIR / "assets" / "rnn" / "next_word"

DEFAULT_EMBED_SIZE = 64
DEFAULT_HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 5


with open(ASSET_DIR / "vocab.pkl", "rb") as f:
    word_to_idx, idx_to_word = pickle.load(f)

vocab_size = len(word_to_idx)

MODEL_CONFIGS = {
    "RNN": {
        "label": "Vanilla RNN",
        "checkpoint": ASSET_DIR / "rnn_wikitext2.pth",
    },
    "LSTM": {
        "label": "LSTM",
        "checkpoint": ASSET_DIR / "lstm_wikitext2.pth",
    },
}


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def build_model(model_type):
    if model_type == "LSTM":
        return LSTMModel(vocab_size, DEFAULT_EMBED_SIZE, DEFAULT_HIDDEN_SIZE)
    return RNNModel(vocab_size, DEFAULT_EMBED_SIZE, DEFAULT_HIDDEN_SIZE)


@st.cache_resource
def load_model(model_type):
    checkpoint_path = MODEL_CONFIGS[model_type]["checkpoint"]
    if not checkpoint_path.exists():
        return None

    model = build_model(model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_next(text_input, model_type):
    model = load_model(model_type)
    if model is None:
        return None

    words = text_input.lower().split()[-SEQUENCE_LENGTH:]
    while len(words) < SEQUENCE_LENGTH:
        words.insert(0, "<UNK>")

    seq = [word_to_idx.get(w, 0) for w in words]
    seq = torch.tensor(seq).unsqueeze(0)

    with torch.no_grad():
        output = model(seq)
        predicted_idx = torch.argmax(output, dim=1).item()

    return idx_to_word.get(predicted_idx, "<UNK>")


def next_word_page(model_type="RNN"):
    model_label = MODEL_CONFIGS[model_type]["label"]

    st.title(f"{model_label} Next Word Predictor")
    st.info(f"This {model_type} model is trained on WikiText-2.")
    st.markdown(f"Type a sentence and let the {model_label} model predict the next word.")

    checkpoint_path = MODEL_CONFIGS[model_type]["checkpoint"]
    if checkpoint_path.exists():
        st.caption(f"Loaded checkpoint: `{checkpoint_path.name}`")
    else:
        st.warning(
            f"No pretrained {model_type} checkpoint was found at "
            f"`{checkpoint_path.name}`. Add that file to enable inference."
        )

    user_input = st.text_input("Enter your text:", key=f"{model_type.lower()}_next_word_input")

    if st.button("Predict Next Word", key=f"{model_type.lower()}_next_word_button"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            return

        prediction = predict_next(user_input, model_type)
        if prediction is None:
            st.error(f"{model_type} inference is unavailable until its checkpoint is added.")
        else:
            st.success(f"Next word: **{prediction}**")


if __name__ == "__main__":
    st.set_page_config(page_title="Sequence Text Predictor", layout="centered")
    next_word_page()
