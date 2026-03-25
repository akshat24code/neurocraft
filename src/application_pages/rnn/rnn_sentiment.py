# ==============================
# 1. IMPORTS
# ==============================
import io
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import requests
import speech_recognition as sr
import streamlit as st
import torch
import torch.nn as nn
from dotenv import load_dotenv

# ==============================
# 2. ENV & PATHS
# ==============================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR.parents[1] / "assets"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# ==============================
# 3. LOAD ASSETS
# ==============================
@st.cache_resource
def load_assets():
    with open(ASSETS_DIR / "word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open(ASSETS_DIR / "config.pkl", "rb") as f:
        config = pickle.load(f)
    return word2idx, config


word2idx, config = load_assets()
vocab_size = config["vocab_size"]
max_len = config["max_len"]

MODEL_CONFIGS = {
    "RNN": {
        "label": "Vanilla RNN",
        "checkpoint": ASSETS_DIR / "rnn_model.pth",
    },
    "LSTM": {
        "label": "LSTM",
        "checkpoint": ASSETS_DIR / "lstm_model.pth",
    },
}


# ==============================
# 4. MODEL
# ==============================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
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
        out, _ = self.rnn(emb)
        pooled, _ = torch.max(out, dim=1)
        out = self.dropout(pooled)
        return self.fc(out).squeeze(1)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3, pad_idx=0):
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


@st.cache_resource
def load_model(model_type):
    checkpoint_path = MODEL_CONFIGS[model_type]["checkpoint"]
    if not checkpoint_path.exists():
        return None

    model_cls = LSTMModel if model_type == "LSTM" else RNNModel
    model = model_cls(vocab_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# ==============================
# 5. HELPERS
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


def pad_sequence_manual(seq, max_length):
    seq = seq[:max_length]
    return seq + [0] * (max_length - len(seq))


def predict_probability(text, model_type):
    model = load_model(model_type)
    if model is None:
        return None

    cleaned = clean_text(text)
    tokens = [word2idx.get(w, 1) for w in cleaned.split()]
    padded = pad_sequence_manual(tokens, max_len)
    tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logit = model(tensor).item()
        return torch.sigmoid(torch.tensor(logit)).item()


def predict_sentiment(text, model_type):
    output = predict_probability(text, model_type)
    if output is None:
        return None, None

    if output >= 0.5:
        return "Positive", round(output * 100, 1)
    return "Negative", round((1 - output) * 100, 1)


def predict_csv_sentiment(df, text_column, model_type):
    rows = []
    for value in df[text_column].fillna(""):
        text = str(value)
        sentiment, confidence = predict_sentiment(text, model_type)
        rows.append({
            "predicted_sentiment": sentiment,
            "confidence_percent": confidence,
            "cleaned_text": clean_text(text),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def get_word_contributions(text, model_type):
    words = text.split()
    if not words:
        return []

    base_prob = predict_probability(text, model_type)
    if base_prob is None:
        return []

    contributions = []
    for i, word in enumerate(words):
        without_word = " ".join(words[:i] + words[i + 1:])
        prob_without = predict_probability(without_word, model_type) if without_word.strip() else 0.5
        contributions.append((word, base_prob - prob_without))
    return contributions


def get_word_color(contribution):
    mag = min(abs(contribution) / 0.25, 1.0)
    if contribution >= 0.02:
        intensity = int(180 * mag)
        return f"rgb(0, {100 + intensity}, 60)"
    if contribution <= -0.02:
        intensity = int(180 * mag)
        return f"rgb({150 + intensity // 2}, 40, 40)"
    return "rgb(150, 140, 100)"


def word_sentiment_html(text, model_type):
    items = get_word_contributions(text, model_type)
    spans = []
    for word, contribution in items:
        color = get_word_color(contribution)
        if contribution >= 0.02:
            label = "Positive contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        elif contribution <= -0.02:
            label = "Negative contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        else:
            label = "Neutral contribution"
            pct = round(min(abs(contribution) * 100, 100), 1)
        spans.append(
            f"<span title='{label}: {pct}%' style='"
            f"color:{color};"
            f"font-weight:600;"
            f"padding:1px 2px;"
            f"border-radius:3px;"
            f"cursor:default;"
            f"'>{word}</span>"
        )
    return "<p style='line-height:2; font-size:1rem;'>" + " ".join(spans) + "</p>"


def speech_to_text_from_audio(audio_bytes: bytes):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data), None
    except sr.UnknownValueError:
        return "", "Could not understand speech. Please speak more clearly."
    except Exception as e:
        return "", f"Speech transcription failed: {e}"


def improve_text_llm(text, api_key):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{
            "role": "user",
            "content": (
                "Rewrite the following text to sound polished, professional, and clear. "
                "Keep the original meaning. Return only the improved text, no preamble.\n\n"
                f"{text}"
            ),
        }],
        "max_tokens": 512,
        "temperature": 0.6,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        return f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Request failed: {e}"


# ==============================
# 6. UI
# ==============================
def rnn_sentiment_page(model_type="RNN"):
    model_label = MODEL_CONFIGS[model_type]["label"]

    st.title(f"{model_label} Sentiment Analyzer")

    checkpoint_path = MODEL_CONFIGS[model_type]["checkpoint"]
    has_checkpoint = checkpoint_path.exists()
    if has_checkpoint:
        st.caption(f"Loaded checkpoint: `{checkpoint_path.name}`")
    else:
        st.warning(
            f"No pretrained {model_type} checkpoint was found at "
            f"`{checkpoint_path.name}`. Add that file to enable inference."
        )

    st.info(
        f"This {model_type} model is trained on the **IMDB movie review dataset** "
        "(50,000 reviews, binary sentiment). Results are most reliable for "
        "English movie/product review-style text.",
    )

    st.divider()
    st.subheader("Single Text Input")

    state_prefix = model_type.lower()
    text = st.text_area(
        "Enter your review or opinion:",
        value=st.session_state.get(f"{state_prefix}_spoken_text", ""),
        height=160,
        placeholder="Type something or use the microphone...",
        key=f"{state_prefix}_sentiment_text",
    )

    st.caption("Speak input (browser microphone)")
    audio_clip = st.audio_input("Record your voice", label_visibility="collapsed")

    if audio_clip is not None:
        audio_bytes = audio_clip.getvalue()
        clip_size = len(audio_bytes)
        previous_size = st.session_state.get(f"{state_prefix}_last_audio_clip_size")
        if clip_size > 0 and clip_size != previous_size:
            with st.spinner("Transcribing your speech..."):
                spoken_text, speech_error = speech_to_text_from_audio(audio_bytes)
            st.session_state[f"{state_prefix}_last_audio_clip_size"] = clip_size
            if speech_error:
                st.warning(speech_error)
            elif spoken_text.strip():
                st.session_state[f"{state_prefix}_spoken_text"] = spoken_text
                st.rerun()

    col_analyze, col_speak = st.columns([3, 1])
    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, key=f"{state_prefix}_analyze")
    with col_speak:
        st.button("Speak", use_container_width=True, disabled=True, help="Use the voice recorder above.", key=f"{state_prefix}_speak")

    st.divider()

    ready_key = f"{state_prefix}_analysis_ready"
    analyzed_key = f"{state_prefix}_analyzed_text"
    sentiment_key = f"{state_prefix}_sentiment"
    confidence_key = f"{state_prefix}_confidence"

    if ready_key not in st.session_state:
        st.session_state[ready_key] = False
    if analyzed_key not in st.session_state:
        st.session_state[analyzed_key] = ""

    if text != st.session_state.get(analyzed_key, ""):
        st.session_state[ready_key] = False

    if analyze_clicked:
        if not has_checkpoint:
            st.warning(f"Add `{checkpoint_path.name}` first to run {model_type} predictions.")
        elif not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Running inference..."):
                sentiment, confidence = predict_sentiment(text, model_type)

            if sentiment is None:
                st.session_state[ready_key] = False
                st.error(f"{model_type} inference is unavailable until its checkpoint is added.")
                return

            st.session_state[ready_key] = True
            st.session_state[analyzed_key] = text
            st.session_state[sentiment_key] = sentiment
            st.session_state[confidence_key] = confidence

    if st.session_state.get(ready_key, False):
        st.subheader("Sentiment Analysis")
        sentiment = st.session_state.get(sentiment_key)
        confidence = st.session_state.get(confidence_key)

        st.caption(f"Prediction model: `{model_type}`")

        if sentiment == "Positive":
            st.success(f"**{sentiment}** - {confidence}% confidence")
        else:
            st.error(f"**{sentiment}** - {confidence}% confidence")

        st.progress(confidence / 100)

        with st.expander("Word-level sentiment breakdown - click to expand"):
            st.caption(
                "Each word is colored by its contribution to the full sentence prediction: "
                "**green** = positive leaning, **red** = negative leaning, "
                "**tan** = neutral. Hover over a word to see its exact score."
            )
            st.markdown(
                word_sentiment_html(st.session_state.get(analyzed_key, ""), model_type),
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Batch Predict From CSV")
    st.caption("Upload a CSV, choose the text column, and run sentiment predictions for every row.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key=f"{state_prefix}_csv_upload",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(10), use_container_width=True)

            text_column = st.selectbox(
                "Select text column",
                options=batch_df.columns.tolist(),
                key=f"{state_prefix}_csv_text_col",
            )

            if st.button("Predict CSV", type="primary", key=f"{state_prefix}_predict_csv"):
                if not has_checkpoint:
                    st.warning(f"Add `{checkpoint_path.name}` first to run {model_type} predictions.")
                else:
                    with st.spinner(f"Running {model_type} predictions on CSV..."):
                        result_df = predict_csv_sentiment(batch_df, text_column, model_type)

                    st.success(f"Predictions complete for {len(result_df)} rows.")
                    st.dataframe(result_df.head(20), use_container_width=True)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{state_prefix}_sentiment_predictions.csv",
                        mime="text/csv",
                        key=f"{state_prefix}_download_csv",
                    )
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")

    st.divider()
    st.subheader("Refine with LLM")

    if NVIDIA_API_KEY:
        st.success("NVIDIA API key loaded from `.env`")
        active_key = NVIDIA_API_KEY
    else:
        active_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Paste your NVIDIA API key here. You can also set NVIDIA_API_KEY in a .env file.",
            key=f"{state_prefix}_nvidia_key",
        )
        if active_key:
            st.success("API key entered")
        else:
            st.error("No API key - enter one above to enable this feature")

    if st.button("Improve Text", key=f"{state_prefix}_improve"):
        if not text.strip():
            st.warning("Enter some text above first.")
        elif not active_key:
            st.warning("Provide an API key to use this feature.")
        else:
            with st.spinner("Calling LLM..."):
                improved = improve_text_llm(text, active_key)
            st.markdown("**Refined text:**")
            st.write(improved)
