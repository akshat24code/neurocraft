import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import os
import traceback
import io
import sys
from contextlib import redirect_stdout
from dotenv import load_dotenv

load_dotenv()

# ==============================
# CONFIG
# ==============================
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_ID = "meta/llama-3.1-70b-instruct"
MAX_FILE_SIZE_MB = 50
MAX_CHART_ROWS = 100


# ==============================
# LLM CALL
# ==============================
def call_nvidia_llm(prompt, retries=2, max_tokens=1000):
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not set in environment."

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                import time
                time.sleep(2 ** attempt)
            else:
                return f"API Error {response.status_code}: {response.text}"
        except requests.exceptions.Timeout:
            if attempt == retries:
                return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"

    return "Error: Max retries exceeded."


# ==============================
# CACHE KEY
# ==============================
def get_df_hash(df):
    content = df.head(10).to_string() + str(df.shape)
    return hashlib.md5(content.encode()).hexdigest()


# ==============================
# EDA
# ==============================
@st.cache_data(show_spinner=False)
def analyze_data(df_hash_key, df):
    summary = {}
    summary["shape"] = df.shape
    summary["dtypes"] = df.dtypes.astype(str)
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    summary["duplicates"] = int(df.duplicated().sum())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    summary["numerical"] = num_cols
    summary["categorical"] = cat_cols

    if num_cols:
        desc = df[num_cols].describe().T
        desc["skewness"] = df[num_cols].skew()
        desc["outlier_flag"] = (
            (df[num_cols] < (desc["mean"] - 3 * desc["std"])) |
            (df[num_cols] > (desc["mean"] + 3 * desc["std"]))
        ).sum()
        summary["num_stats"] = desc

    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique": df[col].nunique(),
            "top_value": df[col].mode()[0] if not df[col].mode().empty else "N/A",
            "top_freq": int(df[col].value_counts().iloc[0]) if df[col].value_counts().shape[0] > 0 else 0
        }
    summary["cat_stats"] = cat_stats

    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top_corr = (
            upper.stack()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_corr.columns = ["Feature A", "Feature B", "Correlation"]
        summary["top_correlations"] = top_corr

    return summary


# ==============================
# COLUMN CHART
# ==============================
@st.cache_data(show_spinner=False)
def get_column_chart_data(df_hash_key, col_name, col_dtype, df):
    col_data = df[col_name].dropna()
    if pd.api.types.is_numeric_dtype(col_data):
        vc = col_data.value_counts().sort_index()
        if vc.shape[0] > MAX_CHART_ROWS:
            counts, bin_edges = np.histogram(col_data, bins=50)
            labels = [f"{bin_edges[i]:.2f}" for i in range(len(counts))]
            vc = pd.Series(counts, index=labels)
        return vc, "numeric"
    else:
        vc = col_data.value_counts().head(MAX_CHART_ROWS)
        return vc, "categorical"


# ==============================
# PROMPT — EDA
# ==============================
def build_analysis_prompt(summary, df_head):
    num_stats_str = summary["num_stats"].to_string() if "num_stats" in summary else "N/A"
    cat_stats_str = json.dumps(summary["cat_stats"], indent=2) if "cat_stats" in summary else "N/A"
    corr_str = summary["top_correlations"].to_string() if "top_correlations" in summary else "N/A"

    return f"""
You are a senior data scientist. Analyze this dataset and respond ONLY with a valid JSON object.
No markdown, no backticks, no preamble. Just raw JSON.

Dataset Info:
- Shape: {summary['shape']}
- Data Types: {summary['dtypes'].to_dict()}
- Missing Values: {summary['missing'].to_dict()}
- Missing %: {summary['missing_pct'].to_dict()}
- Duplicate Rows: {summary['duplicates']}
- Numerical Columns: {summary['numerical']}
- Categorical Columns: {summary['categorical']}

Numerical Statistics (mean, std, min, max, skewness, outlier_flag):
{num_stats_str}

Categorical Statistics (unique count, top value, frequency):
{cat_stats_str}

Top Feature Correlations:
{corr_str}

Top 5 rows:
{df_head}

Respond with this exact JSON structure:
{{
  "problem_type": "classification | regression | clustering | unknown",
  "target_column_guess": "column name or null",
  "high_level_summary": "2-3 sentence overview",
  "key_observations": ["observation 1", "observation 2", "observation 3"],
  "preprocessing_steps": ["step 1", "step 2", "step 3"],
  "suggested_models": [
    {{"name": "Model Name", "reason": "why this fits"}},
    {{"name": "Model Name", "reason": "why this fits"}}
  ],
  "risks": ["risk 1", "risk 2"],
  "data_quality_score": 0-100
}}
"""


# ==============================
# PROMPT — TRAINING CODE GENERATION
# ==============================
def build_training_code_prompt(df, summary, model_name, problem_type, target_col):
    col_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample_rows = df.head(3).to_dict(orient="records")

    # Detect time-series: datetime-typed columns OR ts-related column name keywords
    ts_keywords = {"timestamp", "date", "time", "datetime", "period", "hour", "day", "month", "year"}
    col_names_lower = {c.lower() for c in df.columns}
    has_datetime_col = any(str(dtype).startswith("datetime") for dtype in df.dtypes.values)
    has_ts_keyword = bool(col_names_lower & ts_keywords)
    is_timeseries = has_datetime_col or has_ts_keyword

    split_instruction = (
        "IMPORTANT — this dataset appears to be a time series. "
        "Do NOT use random shuffle. Use a chronological split: "
        "train = first 80%% of rows, test = last 20%% of rows (preserve row order). "
        "This prevents data leakage from future rows into the training set."
        if is_timeseries else
        "Split data: 80%% train, 20%% test using train_test_split with random_state=42."
    )

    return f"""
You are an expert ML engineer writing production-quality Python code.

Your task: Write a complete, self-contained Python training script for the dataset described below.

DATASET CONTEXT:
- Shape: {df.shape}
- Columns and dtypes: {json.dumps(col_info)}
- Target column: {target_col}
- Problem type: {problem_type}
- Selected model: {model_name}
- Sample rows (first 3): {json.dumps(sample_rows, default=str)}
- Missing value counts: {summary['missing'].to_dict()}
- Numerical columns: {summary['numerical']}
- Categorical columns: {summary['categorical']}
- Is time-series dataset: {is_timeseries}

STRICT REQUIREMENTS:
1. The dataframe is already loaded and available as variable `df` (pandas DataFrame). Do NOT load any files.
2. Use only these libraries: pandas, numpy, sklearn (any submodule), and optionally xgboost or lightgbm if appropriate.
3. Handle missing values — for time series prefer forward-fill (df.ffill()), otherwise impute or drop.
4. CRITICAL — After ANY operation that may alter the index (dropna, ffill, filtering, concat, merge),
   you MUST call df = df.reset_index(drop=True) immediately afterward.
   Never use df[col][0] or series[0] style integer access — always use .iloc[0] instead.
5. Encode categorical columns using LabelEncoder. Drop or ignore pure ID/index columns.
6. {split_instruction}
7. Train the model and evaluate on the test set.
8. For regression: compute MAE, RMSE, R2, and MAPE (guard against division by zero with a small epsilon).
   For classification: compute accuracy, f1_weighted, precision_weighted, recall_weighted.
9. At the end, populate a dict called `results` with:
   - "model_name": string
   - "metrics": dict of metric_name -> float
   - "feature_importances": dict of feature_name -> importance_value (if model supports it, else empty dict)
   - "train_size": int
   - "test_size": int
   - "split_strategy": "chronological" or "random"
10. Print nothing to stdout (no print statements).
11. Do NOT wrap in functions or classes. Write flat, sequential code.
12. Do NOT include any markdown, backticks, comments explaining the task, or import of dotenv/streamlit/matplotlib.
13. The LAST line of code must be: results = results

Output ONLY the raw Python code. No explanation, no markdown fences.
"""


# ==============================
# EXECUTE GENERATED CODE
# ==============================
def execute_generated_code(code, df):
    """
    Runs LLM-generated code in a controlled namespace with df injected.
    Returns (results_dict, stdout_str, error_str).
    """
    # CRITICAL: __builtins__ must be present so import statements inside exec()
    # resolve installed packages normally. Without it, `import sklearn` raises
    # ModuleNotFoundError even when sklearn is installed.
    import builtins
    # Defensively reset the index so generated code never hits KeyError: 0
    # when the LLM uses series[0] after a dropna/ffill that gaps the index.
    df_clean = df.copy().reset_index(drop=True)
    namespace = {
        "__builtins__": builtins,
        "df": df_clean,
        "pd": pd,
        "np": np,
        "results": {}
    }

    stdout_capture = io.StringIO()
    error_str = None

    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception:
        error_str = traceback.format_exc()

    results = namespace.get("results", {})
    stdout_str = stdout_capture.getvalue()

    return results, stdout_str, error_str


# ==============================
# PARSE LLM JSON
# ==============================
def parse_llm_json(raw):
    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        return None, str(e)


# ==============================
# UI — AI SUMMARY
# ==============================
def render_ai_summary(parsed):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Problem Type", parsed.get("problem_type", "—").capitalize())
    with col2:
        st.metric("Target Column (guessed)", parsed.get("target_column_guess") or "—")
    with col3:
        score = parsed.get("data_quality_score", "—")
        st.metric("Data Quality Score", f"{score}/100" if isinstance(score, int) else "—")

    st.markdown("**Summary**")
    st.write(parsed.get("high_level_summary", "—"))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Key Observations**")
        for obs in parsed.get("key_observations", []):
            st.write(f"- {obs}")
        st.markdown("**Risks**")
        for risk in parsed.get("risks", []):
            st.write(f"- {risk}")

    with col_b:
        st.markdown("**Preprocessing Steps**")
        for i, step in enumerate(parsed.get("preprocessing_steps", []), 1):
            st.write(f"{i}. {step}")
        st.markdown("**Suggested Models**")
        for m in parsed.get("suggested_models", []):
            st.write(f"**{m['name']}** — {m['reason']}")


# ==============================
# UI — TRAINING RESULTS
# ==============================
def render_training_results(results):
    if not results:
        st.warning("No results returned from the generated code.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", results.get("model_name", "—"))
    c2.metric("Train Samples", results.get("train_size", "—"))
    c3.metric("Test Samples", results.get("test_size", "—"))
    c4.metric("Split Strategy", results.get("split_strategy", "—").capitalize())

    # ── Metric diagnostics ──────────────────────────────────────────────────
    metrics = results.get("metrics", {})

    # Normalise key lookup — LLM may return r2, R2, r2_score, R2_Score etc.
    metrics_lower = {k.lower(): float(v) for k, v in metrics.items()}
    r2   = metrics_lower.get("r2") or metrics_lower.get("r2_score")
    mape = metrics_lower.get("mape")

    if r2 is not None:
        if r2 < 0:
            st.error(
                f"**Negative R\u00b2 detected ({r2:.4f})** \u2014 the model is performing worse than a "
                "naive mean baseline. This is a real and honest result, not a bug. Common causes:\n\n"
                "- **Random split on time-series data** \u2014 future rows leak into training, causing "
                "the model to fail on temporally held-out test data. Re-train using a chronological "
                "split (train = first 80%, test = last 20%).\n"
                "- **Underfitting** \u2014 the selected model may be too simple for this data. "
                "Try a tree-based model such as XGBoost or Random Forest.\n"
                "- **Missing lag features** \u2014 demand / load forecasting typically requires lag and "
                "rolling-window features. Consider adding them before training."
            )
        elif r2 < 0.3:
            st.warning(
                f"**Low R\u00b2 ({r2:.4f})** \u2014 the model explains less than 30% of variance in the target. "
                "Consider feature engineering, a stronger model, or verifying the split strategy."
            )

    if mape is not None and mape > 20:
        st.warning(
            f"**High MAPE ({mape:.2f}%)** \u2014 prediction errors are large relative to actual values. "
            "Common with seasonal demand data that lacks lag or calendar features."
        )

    st.markdown("#### Evaluation Metrics")
    if metrics:
        metric_df = pd.DataFrame(
            [(k.upper(), round(float(v), 4)) for k, v in metrics.items()],
            columns=["Metric", "Value"]
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
    else:
        st.info("No metrics returned.")

    fi = results.get("feature_importances", {})
    if fi:
        st.markdown("#### Feature Importances")
        fi_series = pd.Series(fi).sort_values(ascending=False).head(20)
        st.bar_chart(fi_series)


# ==============================
# AI MENTOR CHATBOT LOGIC
# ==============================
def get_chatbot_response(prompt, summary):
    if not NVIDIA_API_KEY:
        return "Your NVIDIA_API_KEY is missing! I am running in low-power mode. Please add your API key to .env to unlock my true LLM capabilities!"

    dataset_context = "No dataset currently uploaded. Ask the user to upload a CSV file."
    
    if summary:
        num = len(summary.get("numerical", []))
        cat = len(summary.get("categorical", []))
        shape = summary.get("shape", [0,0])
        
        # Safely handle missing values which is a pandas series
        try:
            missing_total = int(summary.get("missing", pd.Series()).sum())
        except:
            missing_total = "Unknown"

        dataset_context = (
            f"Dataset Shape: {shape[0]} rows, {shape[1]} columns.\n"
            f"Numerical Columns ({num}): {', '.join(summary.get('numerical', [])[:8])}...\n"
            f"Categorical Columns ({cat}): {', '.join(summary.get('categorical', [])[:8])}...\n"
            f"Total Missing Cells: {missing_total}."
        )

    # Build the Augmented Prompt for Llama 3
    augmented_prompt = f"""You are the NeuroCraft AI Mentor, an expert data science and machine learning assistant.
You are helping a user analyze a dataset and build ML models.

RULES:
1. Be directly helpful, encouraging, and highly concise. Do not write essays.
2. Use Markdown formatting (bolding, bullet points) to make your response readable.
3. Only write code if explicitly asked. Otherwise, focus on conceptual guidance and analysis.
4. If the user asks about their data, reference the context below. If they haven't uploaded data, encourage them to do so via the UI.

CURRENT DATASET CONTEXT:
{dataset_context}

USER QUERY:
{prompt}
"""
    
    # Call the actual LLM API
    return call_nvidia_llm(augmented_prompt, max_tokens=800)


def render_chatbot():
    st.sidebar.markdown("### 🤖 AI Mentor")
    st.sidebar.caption("Powered by Llama 3 (Nvidia API)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to the AI Playground! 🚀 Upload a dataset and I'll analyze it for you in real time."}]
        
    for msg in st.session_state.messages:
        with st.sidebar.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.sidebar.chat_input("Ask about your data or models..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)
            
        with st.sidebar.chat_message("assistant"):
            summary = st.session_state.get("df_summary", None)
            
            with st.spinner("Analyzing..."):
                response = get_chatbot_response(prompt, summary)
            
            message_placeholder = st.empty()
            full_res = ""
            import time
            
            # Simple typing animation
            words = response.split(" ")
            for i, chunk in enumerate(words):
                full_res += chunk + " "
                # Only animate the first 50 words to avoid super long waits for large LLM responses
                if i < 50:
                    message_placeholder.markdown(full_res + "▌")
                    time.sleep(0.02)
                
            message_placeholder.markdown(full_res)
            
        st.session_state.messages.append({"role": "assistant", "content": full_res})


# ==============================
# UI CSS INJECTIONS
# ==============================
def inject_custom_css():
    st.markdown("""
        <style>
        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(15px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .animate-component {
            animation: slideUpFade 0.7s ease-out forwards;
        }
        div[data-testid="stExpander"] {
            background: rgba(30, 41, 59, 0.4) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease;
        }
        div[data-testid="stExpander"]:hover {
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            box-shadow: 0 8px 32px rgba(56, 189, 248, 0.15) !important;
        }
        .stButton button {
            transition: all 0.2s ease-in-out;
            border-radius: 8px !important;
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 16px rgba(56, 189, 248, 0.3);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
        }
        </style>
    """, unsafe_allow_html=True)


# ==============================
# MAIN APP
# ==============================
def explore_data_page():
    inject_custom_css()
    render_chatbot()
    
    st.markdown("<div class='animate-component'>", unsafe_allow_html=True)
    st.title("✨ Interactive AI Playground")
    st.caption("Upload data over the interactive pipeline. Your **AI Mentor** in the sidebar is ready to guide you!")
    st.markdown("</div>", unsafe_allow_html=True)

    if not NVIDIA_API_KEY:
        st.error("NVIDIA_API_KEY not found. Neural generation will disabled, but the Mentor is still active.")

    st.divider()

    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Max 50 MB per file")

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
            return

        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")

        df_hash = get_df_hash(df)

        if st.session_state.get("df_hash") != df_hash:
            for key in ["ai_summary_raw", "ai_summary_parsed", "ai_summary_error",
                        "training_results", "training_code", "training_error"]:
                st.session_state.pop(key, None)
            st.session_state["df_hash"] = df_hash

        summary = analyze_data(df_hash, df)
        st.session_state["df_summary"] = summary

        # ── Interactive Tab Pipeline ──────────────────────────────────────────
        tab_overview, tab_visualize, tab_train = st.tabs([
            "📊 1. Overview & Clean", 
            "📈 2. Deep Dive Visuals", 
            "🧠 3. Train & Evaluate"
        ])
        
        # TAB 1: OVERVIEW
        with tab_overview:
            st.markdown("<div class='animate-component'>", unsafe_allow_html=True)
            st.subheader("Data Overview")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", f"{df.shape[1]:,}")
            c3.metric("Missing Cells", int(summary["missing"].sum()))
            c4.metric("Duplicate Rows", summary["duplicates"])
            
            st.divider()
            st.dataframe(df.head(15), use_container_width=True)
            
            with st.expander("Show Missing Values & Types"):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.caption("Data Types")
                    st.dataframe(summary["dtypes"].reset_index().rename(columns={"index": "Column", 0: "Type"}), use_container_width=True)
                with col_right:
                    st.caption("Missing Values")
                    miss = summary["missing"]
                    st.dataframe(miss[miss > 0].reset_index().rename(columns={"index": "Column", 0: "Missing Count"}), use_container_width=True)
            
            if "top_correlations" in summary:
                with st.expander("Smart Correlations Grid"):
                    st.dataframe(summary["top_correlations"].style.background_gradient(subset=["Correlation"], cmap="Blues"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 2: VISUALIZE
        with tab_visualize:
            st.markdown("<div class='animate-component'>", unsafe_allow_html=True)
            st.subheader("Interactive Feature Deep Dive")
            selected_col = st.selectbox("Select a feature to isolate", df.columns.tolist(), key="deep_dive_col")

            if selected_col:
                col_data = df[selected_col]
                st.markdown(f"#### Analyzing: `{selected_col}`")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique Classes", col_data.nunique())
                c2.metric("Fill Rate", f"{100 - (col_data.isnull().mean() * 100):.1f}%")
                
                if pd.api.types.is_numeric_dtype(col_data):
                    c3.metric("Feature Skewness", f"{col_data.skew():.2f}")
                else:
                    c3.metric("Top Frequency", col_data.mode()[0] if not col_data.mode().empty else "N/A")

                chart_data, chart_type = get_column_chart_data(df_hash, selected_col, str(col_data.dtype), df)

                if chart_data.shape[0] > 0:
                    st.bar_chart(chart_data, height=350)
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 3: TRAIN & EVALUATE
        with tab_train:
            st.markdown("<div class='animate-component'>", unsafe_allow_html=True)
            st.subheader("Neural Code Generator & Evaluator")
            st.caption("Let the LLM write a specialized pipeline for your data, executing it safely in Python.")
            
            with st.expander("Automated Analysis Report", expanded=True):
                if "ai_summary_parsed" not in st.session_state and NVIDIA_API_KEY:
                    with st.spinner("Analyzing semantics via LLM..."):
                        raw = call_nvidia_llm(build_analysis_prompt(summary, df.head().to_string()))
                        parsed, _ = parse_llm_json(raw)
                        if parsed:
                            st.session_state["ai_summary_parsed"] = parsed
                            
                if st.session_state.get("ai_summary_parsed"):
                    render_ai_summary(st.session_state["ai_summary_parsed"])
                elif not NVIDIA_API_KEY:
                    st.warning("NVIDIA API missing. Proceeding with fallback local training mode.")

            parsed_summary = st.session_state.get("ai_summary_parsed", {})
            suggested_models = [m["name"] for m in parsed_summary.get("suggested_models", [])] if parsed_summary else []
            if not suggested_models:
                suggested_models = ["Random Forest", "Logistic Regression", "XGBoost"]

            col_left, col_right = st.columns(2)
            with col_left:
                selected_model = st.selectbox("Select ML Model", options=suggested_models)
            with col_right:
                all_cols = df.columns.tolist()
                guessed = parsed_summary.get("target_column_guess")
                idx = all_cols.index(guessed) if guessed in all_cols else 0
                target_col = st.selectbox("Target Column", options=all_cols, index=idx)

            if st.button("🚀 Generate Code & Train", type="primary"):
                if not NVIDIA_API_KEY:
                    st.error("NVIDIA API Key required to generate code dynamically.")
                else:
                    with st.spinner(f"Writing intelligent pipeline for {selected_model}..."):
                        code_prompt = build_training_code_prompt(df, summary, selected_model, parsed_summary.get("problem_type", "classification"), target_col)
                        raw_code = call_nvidia_llm(code_prompt, max_tokens=2000)
                        
                    clean_code = raw_code.strip()
                    if clean_code.startswith("```"): clean_code = "\n".join(clean_code.split("\n")[1:])
                    if clean_code.endswith("```"): clean_code = "\n".join(clean_code.split("\n")[:-1])
                    st.session_state["training_code"] = clean_code.strip()

                    with st.spinner("Executing pipeline generated code..."):
                        results, _, error_str = execute_generated_code(clean_code.strip(), df)
                        st.session_state["training_results"] = results
                        st.session_state["training_error"] = error_str

            if "training_results" in st.session_state:
                if st.session_state.get("training_error"):
                    st.error("Execution failed.")
                    with st.expander("Error Stacktrace"):
                        st.code(st.session_state["training_error"])
                else:
                    st.success("Model Evaluated Successfully!")
                    render_training_results(st.session_state["training_results"])
                    
                if "training_code" in st.session_state:
                    with st.expander("Review Generated Source Code"):
                        st.code(st.session_state["training_code"], language="python")
                        st.download_button("Download Script", st.session_state["training_code"], file_name="pipeline.py")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="AI Playground", layout="wide")
    explore_data_page()
