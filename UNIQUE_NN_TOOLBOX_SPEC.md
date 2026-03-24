# NeuroCraft Lab - Unique Toolbox Specification

## Vision
NeuroCraft Lab is a single, interactive Streamlit toolbox designed to help learners and builders understand neural networks through guided concepts, real applications, and AI-assisted experimentation.

This document is a unique implementation blueprint (not a cloned readme) for building and maintaining the full product in one place.

## Core Product Pillars
- Interactive learning UI with clear section-based navigation.
- Learner-first neural network walkthrough modules.
- RNN mini-applications with practical text tasks.
- Computer vision lab with modular OpenCV detectors.
- AI Playground for dataset profiling and script generation.
- Built-in IRIS dataset for immediate experimentation.

## Functional Requirements

### 1) Streamlit Learning Interface
- Single entry app with stable sidebar navigation.
- Clear separation of Home, Learner, Applications, Docs, AI Playground.
- Session-state based route handling to avoid page conflicts.

### 2) Learner Modules
- Perceptron:
  - Logic gate learning (AND/OR/NAND/XOR exploration).
  - Optional custom CSV upload flow.
- Forward Propagation:
  - Stepwise neuron activation walkthrough.
  - Visualized weighted sum + activation transitions.
- Backward Propagation:
  - Error flow and gradient update explanation.
  - Parameter update simulation controls.
- MLP:
  - Binary and multiclass training path.
  - CSV or IRIS driven feature/target selection.

### 3) RNN Applications
- RNN landing screen with mode selector.
- Next Word Predictor:
  - Model trained on WikiText-2.
  - Top-k next token suggestion workflow.
- Sentiment Analyzer:
  - IMDB-trained sequence model.
  - Text input with confidence-based label output.

### 4) OpenCV Applications
- OpenCV landing with detection mode selector.
- Detection modes:
  - Face Detection
  - Eye + Smile Detection
  - Stop Sign Detection
  - Real-Time Face Count
- Input modes:
  - Webcam live mode
  - Video upload mode
  - Image upload mode
- Environment-aware runtime:
  - Local: cv2 camera path
  - Cloud: WebRTC fallback

### 5) AI Playground
- CSV uploader with schema/quality profiling.
- Auto-infer problem type (classification/regression suggestions).
- Generate training script from selected target + model profile.
- Optional in-app execution and metric summary view.

### 6) Built-In Data
- Include `IRIS.csv` in `/data` as an always-available default dataset.

## Target Architecture

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── IRIS.csv
└── src/
    ├── __init__.py
    ├── ai_playground_pages/
    │   └── ask_ai.py
    ├── application_pages/
    │   ├── open_cv/
    │   │   ├── open_cv_landing.py
    │   │   ├── open_cv_shared.py
    │   │   ├── open_cv_core.py
    │   │   ├── open_cv_webcam.py
    │   │   ├── open_cv_video.py
    │   │   ├── open_cv_image.py
    │   │   ├── open_cv_detection.py
    │   │   ├── cascades/
    │   │   │   └── haarcascade_*.xml
    │   │   └── sample/
    │   └── rnn/
    │       ├── rnn_landing.py
    │       ├── next_word.py
    │       └── rnn_sentiment.py
    ├── learner_pages/
    │   ├── perceptron_ui.py
    │   ├── forward_propagation.py
    │   ├── backward_propagation.py
    │   └── mlp.py
    └── assets/
        ├── image/
        │   └── nn_image.jpg
        ├── documents/
        │   ├── perceptron.py
        │   ├── forward_propagation.py
        │   ├── back_propagation.py
        │   └── mnp.py
        ├── rnn/
        │   ├── next_word/
        │   │   ├── vocab.pkl
        │   │   └── rnn_wikitext2.pth
        │   └── sentiment/
        │       ├── rnn_model.pth
        │       ├── word2idx.pkl
        │       └── config.pkl
        └── open_cv/
```

## Design Principles
- Keep detection logic pure and UI-independent where possible.
- Keep every major feature behind its own landing/controller page.
- Prefer explicit path resolution using `Path(__file__).resolve()`.
- Fail gracefully with user-friendly warnings for missing assets.
- Maintain a "teach-first" UX: explanation + controls + results.

## Suggested Navigation Labels
- Home
- AI Playground
- Learner
- Applications
- Documentation

## Quality Checklist
- App starts cleanly with `streamlit run app.py`.
- Each sidebar branch resolves exactly one active page.
- OpenCV runs in both local webcam and cloud-safe modes.
- RNN pages handle missing model files with clear messages.
- AI Playground validates CSV shape and target availability.
- IRIS fallback option always available in learner flows.

## Future Extensions
- Add CNN learner path with image classification.
- Add LSTM explainer mode to RNN section.
- Add confusion matrix + ROC diagnostics globally.
- Add model export/import in AI Playground.
- Add multilingual UI hints for beginner users.
