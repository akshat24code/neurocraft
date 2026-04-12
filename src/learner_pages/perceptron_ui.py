import streamlit as st
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import warnings
from src.utils.result_interpreter import interpret_results

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GATES = {
    "AND": {"data": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)], "separable": True},
    "OR":  {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)], "separable": True},
    "XOR": {"data": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], "separable": False},
}

# ══════════════════════════════════════════════════════════════════════════════
# DATASET VALIDATION — Full edge case handler
# ══════════════════════════════════════════════════════════════════════════════

class DatasetValidationResult:
    def __init__(self):
        self.valid = False
        self.X = None          # np.ndarray (n_samples, n_features)
        self.y = None          # np.ndarray (n_samples,)
        self.n_features = 0
        self.n_samples = 0
        self.feature_cols = []
        self.target_col = ""
        self.warnings = []     # non-fatal issues shown to user
        self.errors = []       # fatal issues — block training
        self.info = []         # informational notices


def validate_dataset(df, feature_cols, target_col):
    """
    Validate uploaded dataframe against all known edge cases.
    Returns a DatasetValidationResult with rich diagnostics.
    """
    result = DatasetValidationResult()
    result.feature_cols = feature_cols
    result.target_col = target_col

    # ── 1. Empty dataframe ───────────────────────────────────────────────────
    if df.empty:
        result.errors.append("The uploaded CSV is empty.")
        return result

    # ── 2. Feature columns exist ─────────────────────────────────────────────
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        result.errors.append(f"Columns not found in CSV: {missing_cols}")
        return result

    if target_col not in df.columns:
        result.errors.append(f"Target column '{target_col}' not found in CSV.")
        return result

    # ── 3. No feature columns selected ──────────────────────────────────────
    if len(feature_cols) == 0:
        result.errors.append("Please select at least one feature column.")
        return result

    # ── 4. Feature = Target overlap ──────────────────────────────────────────
    if target_col in feature_cols:
        result.errors.append("Target column cannot also be a feature column.")
        return result

    # ── 5. Extract and clean ──────────────────────────────────────────────────
    subset_cols = feature_cols + [target_col]
    df_clean = df[subset_cols].copy()

    # ── 6. Missing values ────────────────────────────────────────────────────
    n_before = len(df_clean)
    df_clean = df_clean.dropna()
    n_after = len(df_clean)
    dropped = n_before - n_after

    if n_after == 0:
        result.errors.append("All rows have missing values. No data to train on.")
        return result
    if dropped > 0:
        result.warnings.append(
            f"{dropped} row(s) with missing values were dropped. "
            f"Proceeding with {n_after} rows."
        )

    # ── 7. Minimum samples ───────────────────────────────────────────────────
    if n_after < 4:
        result.errors.append(
            f"Only {n_after} valid rows found. Need at least 4 samples to train."
        )
        return result

    # ── 8. Non-numeric features ──────────────────────────────────────────────
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            result.errors.append(
                f"Feature column '{col}' contains non-numeric values. "
                f"Perceptron requires numeric inputs. Encode categoricals first."
            )
            return result

    # ── 9. Non-numeric target ────────────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(df_clean[target_col]):
        result.errors.append(
            f"Target column '{target_col}' is non-numeric. "
            f"Please encode class labels as integers (0 and 1)."
        )
        return result

    # ── 10. Target multiclass check ───────────────────────────────────────────
    unique_targets = sorted(df_clean[target_col].unique())
    n_classes = len(unique_targets)

    if n_classes == 1:
        result.errors.append(
            f"Target column has only one unique value ({unique_targets[0]}). "
            f"Need exactly 2 classes for binary classification."
        )
        return result

    if n_classes > 2:
        result.errors.append(
            f"Multiclass target detected ({n_classes} classes: {unique_targets}). "
            f"A single perceptron supports binary classification only. "
            f"Tip: For multiclass problems, use One-vs-Rest with multiple perceptrons "
            f"or the MLP section of this toolbox."
        )
        return result

    # ── 11. Target values not 0/1 — auto remap ────────────────────────────────
    if set(unique_targets) != {0, 1}:
        if set(unique_targets) == {-1, 1}:
            df_clean[target_col] = df_clean[target_col].map({-1: 0, 1: 1})
            result.warnings.append(
                "Target values {-1, 1} automatically remapped to {0, 1}."
            )
        else:
            label_map = {unique_targets[0]: 0, unique_targets[1]: 1}
            df_clean[target_col] = df_clean[target_col].map(label_map)
            result.warnings.append(
                f"Target labels {unique_targets} auto-encoded: "
                f"{unique_targets[0]} → 0, {unique_targets[1]} → 1."
            )

    # ── 12. Class imbalance check ─────────────────────────────────────────────
    class_counts = df_clean[target_col].value_counts()
    minority = class_counts.min()
    majority = class_counts.max()
    imbalance_ratio = majority / minority if minority > 0 else float("inf")

    if imbalance_ratio > 5:
        result.warnings.append(
            f"Severe class imbalance (ratio {imbalance_ratio:.1f}:1). "
            f"Class 0: {class_counts.get(0, 0)}, Class 1: {class_counts.get(1, 0)}. "
            f"Accuracy may be misleading — consider resampling your dataset."
        )
    elif imbalance_ratio > 2:
        result.warnings.append(
            f"Mild class imbalance (ratio {imbalance_ratio:.1f}:1). "
            f"Class 0: {class_counts.get(0, 0)}, Class 1: {class_counts.get(1, 0)}."
        )

    # ── 13. Constant feature columns (zero variance) ──────────────────────────
    for col in feature_cols:
        if df_clean[col].nunique() == 1:
            result.warnings.append(
                f"Feature '{col}' has a constant value ({df_clean[col].iloc[0]}). "
                f"It contributes no information — consider removing it."
            )

    # ── 14. Duplicate rows ────────────────────────────────────────────────────
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        result.warnings.append(
            f"{n_duplicates} duplicate row(s) found and included in training."
        )

    # ── 15. Contradictory labels (same features, different labels) ────────────
    feature_df = df_clean[feature_cols]
    dup_mask = feature_df.duplicated(keep=False)
    if dup_mask.any():
        label_groups = df_clean[dup_mask].groupby(feature_cols)[target_col].nunique()
        contradictions = int((label_groups > 1).sum())
        if contradictions > 0:
            result.warnings.append(
                f"{contradictions} contradictory sample(s): same features mapped to "
                f"different labels. Dataset may not be linearly separable."
            )

    # ── 16. Feature scale warning ─────────────────────────────────────────────
    large_scale_cols = []
    for col in feature_cols:
        col_range = df_clean[col].max() - df_clean[col].min()
        if col_range > 100:
            large_scale_cols.append(col)
    if large_scale_cols:
        result.warnings.append(
            f"Features {large_scale_cols} have large value ranges. "
            f"Consider normalizing (min-max or z-score) for faster convergence."
        )

    # ── 17. Large dataset notice ──────────────────────────────────────────────
    if n_after > 10_000:
        result.info.append(
            f"Large dataset ({n_after:,} rows). Training may take a moment."
        )

    # ── 18. Single feature info ───────────────────────────────────────────────
    if len(feature_cols) == 1:
        result.info.append(
            "Single feature selected. Decision boundary is a threshold on 1 axis."
        )

    # ── 19. High-dimensional notice ───────────────────────────────────────────
    if len(feature_cols) > 2:
        result.info.append(
            f"{len(feature_cols)} features selected. "
            f"Decision boundary cannot be visualized directly in 2D — "
            f"per-sample prediction table will be shown instead."
        )

    # ── All checks passed ─────────────────────────────────────────────────────
    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[target_col].values.astype(int)

    result.valid = True
    result.X = X
    result.y = y
    result.n_features = X.shape[1]
    result.n_samples = X.shape[0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTRON TRAINING — N-feature, vectorized
# ══════════════════════════════════════════════════════════════════════════════

def train_perceptron(X, y, weights, bias, learning_rate, epochs):
    """
    Train a perceptron on N-feature binary data.
    Returns: weights, bias, losses, log_lines, converged, converged_epoch
    """
    w = weights.copy().astype(float)
    b = float(bias)
    losses = []
    log_lines = []
    converged = False
    converged_epoch = -1

    w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
    log_lines.append(f"Initializing...\n   {w_str}  b={b:.4f}\n{'─'*60}\n")

    log_interval = max(1, epochs // 200)

    for epoch in range(epochs):
        total_error = 0
        for xi, yi in zip(X, y):
            y_pred = 1 if (np.dot(w, xi) + b) >= 0 else 0
            error = int(yi) - y_pred
            total_error += abs(error)
            if error != 0:
                w += learning_rate * error * xi
                b += learning_rate * error

        losses.append(total_error)

        if epoch % log_interval == 0 or epoch == epochs - 1:
            w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
            log_lines.append(
                f"Epoch {epoch+1:>4}/{epochs}  |  "
                f"Loss: {total_error:.4f}  |  {w_str}  b={b:.4f}\n"
            )

        if total_error == 0:
            converged = True
            converged_epoch = epoch + 1
            log_lines.append(f"{'─'*60}\n")
            log_lines.append(f"Converged at epoch {converged_epoch}! Loss = 0\n")
            break

    log_lines.append(f"{'─'*60}\n")
    w_str = "  ".join([f"w{i+1}={v:.4f}" for i, v in enumerate(w)])
    log_lines.append(
        f"Training complete. Final Loss: {losses[-1]:.4f}\n"
        f"   {w_str}  b={b:.4f}\n"
    )

    return w, b, losses, log_lines, converged, converged_epoch


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(X, y, w, b):
    preds = (X @ w + b >= 0).astype(int)
    correct = int(np.sum(preds == y))
    return correct, len(y)


def plot_loss_curve(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)), y=losses,
        mode="lines", line=dict(color="#2563EB", width=2.5), name="Total Error",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
    ))
    fig.update_layout(
        title=dict(text="Training Loss per Epoch", font=dict(size=15, color="#111827")),
        xaxis=dict(title="Epoch", gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(title="Total Absolute Error", gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

def plot_loss_surface_3d(X, y, w_final, b_final):
    # Generating meshgrid for w1 and w2 around the final weights
    w1_range = np.linspace(w_final[0] - 2, w_final[0] + 2, 20)
    w2_range = np.linspace(w_final[1] - 2, w_final[1] + 2, 20)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_temp = np.array([W1[i, j], W2[i, j]])
            if len(w_temp) < X.shape[1]:
                # If more than 2 features, pad with final weights
                pad = w_final[2:]
                w_temp = np.concatenate([w_temp, pad])
            
            error = 0
            for xi, yi in zip(X, y):
                pred = 1 if (np.dot(w_temp, xi) + b_final) >= 0 else 0
                error += abs(yi - pred)
            Z[i, j] = error

    fig = go.Figure(data=[go.Surface(z=Z, x=W1, y=W2, colorscale='Viridis', opacity=0.8)])
    
    # Mark the final weight
    fig.add_trace(go.Scatter3d(
        x=[w_final[0]], y=[w_final[1]], z=[0],
        mode='markers', marker=dict(size=8, color='red', symbol='diamond'),
        name='Final Weights'
    ))
    
    fig.update_layout(
        title="3D Loss Surface (w1 vs w2 vs Loss)",
        scene=dict(
            xaxis_title='Weight 1',
            yaxis_title='Weight 2',
            zaxis_title='Loss',
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def plot_decision_boundary_2d(X, y, w, b, feature_cols, title="Decision Boundary"):
    df = pd.DataFrame(X, columns=feature_cols)
    df["Label"] = np.where(y == 0, "Class 0", "Class 1")
    fig = go.Figure()
    for label, color in [("Class 0", "#EF4444"), ("Class 1", "#16A34A")]:
        s = df[df["Label"] == label]
        fig.add_trace(go.Scatter(
            x=s[feature_cols[0]], y=s[feature_cols[1]],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(width=2, color="#111827")),
            name=label
        ))
    w1, w2 = w[0], w[1]
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    if abs(w2) > 1e-9:
        x1_range = np.linspace(x1_min, x1_max, 300)
        x2_boundary = -(w1 * x1_range + b) / w2
        fig.add_trace(go.Scatter(
            x=x1_range, y=x2_boundary, mode="lines",
            line=dict(color="#D97706", width=2.5, dash="dash"),
            name="Decision Boundary"
        ))
    elif abs(w1) > 1e-9:
        fig.add_vline(x=-b/w1, line=dict(color="#D97706", width=2.5, dash="dash"),
                      annotation_text="Boundary")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#111827")),
        xaxis=dict(title=feature_cols[0], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(title=feature_cols[1], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def plot_1d_threshold(X, y, w, b, feature_cols):
    threshold = -b / w[0] if abs(w[0]) > 1e-9 else None
    colors = ["#EF4444" if yi == 0 else "#16A34A" for yi in y]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=np.zeros(len(X)), mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=2, color="#111827")),
        name="Samples"
    ))
    if threshold is not None:
        fig.add_vline(x=threshold, line=dict(color="#D97706", width=2.5, dash="dash"),
                      annotation_text=f"Threshold = {threshold:.4f}",
                      annotation_font_color="#D97706")
    fig.update_layout(
        title=dict(text="1D Decision Threshold", font=dict(size=15, color="#111827")),
        xaxis=dict(title=feature_cols[0], gridcolor="#E5E7EB", linecolor="#D1D5DB", color="#374151"),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def build_prediction_table(X, y, w, b, feature_cols):
    preds = (X @ w + b >= 0).astype(int)
    df = pd.DataFrame(X, columns=feature_cols)
    df["Actual"] = y
    df["Predicted"] = preds
    df["Correct"] = np.where(df["Actual"] == df["Predicted"], "✅", "❌")
    return df


def render_log(placeholder, lines):
    combined = "".join(lines)
    placeholder.markdown(
        f"""<div style="background-color:#0e1117;color:#00ff88;
            font-family:'Courier New',monospace;font-size:12.5px;
            padding:12px 16px;border-radius:8px;height:300px;
            overflow-y:auto;white-space:pre;border:1px solid #2a2a2a;
        ">{combined}</div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "weights": None, "bias": None,
        "losses": [], "trained": False,
        "training_log": [], "converged": False,
        "converged_epoch": -1, "last_gate": None,
        "n_features": 2, "feature_cols": ["X1", "X2"],
        "X_train": None, "y_train": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in ["weights", "bias", "losses", "trained", "training_log",
              "converged", "converged_epoch", "X_train", "y_train"]:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def perceptron_page():
    st.title("Perceptron — Binary Linear Classifier")
    st.caption(
        "Learns a linear boundary via: **ŷ = step(w·x + b)** "
        "and updates: **Δw = η(y−ŷ)x**, **Δb = η(y−ŷ)**"
    )

    tab_theory, tab_math, tab_experiment, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "Analysis"
    ])

    _init_state()

    with tab_theory:
        st.subheader("What is a Perceptron?")
        st.markdown(
            "The **Perceptron** is the simplest form of an artificial neural network. "
            "It is a single-layer, binary linear classifier. "
            "It takes multiple inputs, multiplies them by connection weights, adds a bias, "
            "and passes the sum through a step function to output either 0 or 1.\n\n"
            "**Key Concepts:**\n"
            "- **Linearly Separable**: It can only solve linearly separable problems (like AND, OR). "
            "It fails on non-linear problems like XOR.\n"
            "- **Learning Process**: It learns by updating weights when it makes an error, "
            "pushing the decision boundary iteratively toward the correct classification."
        )

    with tab_math:
        st.subheader("The Mathematics")
        st.markdown(
            r"""
            **1. Weighted Sum ($z$)**
            $$ z = \sum_{i=1}^{n} w_i x_i + b $$

            **2. Activation Function (Step Function)**
            $$ \hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases} $$

            **3. Error Calculation**
            $$ e = y - \hat{y} $$

            **4. Weight Update Rule**
            $$ w_i \leftarrow w_i + \eta \cdot e \cdot x_i $$
            $$ b \leftarrow b + \eta \cdot e $$
            Where $\eta$ is the **Learning Rate**.
            """
        )

    with tab_experiment:
        X, y, feature_cols, gate_name = None, None, ["X1", "X2"], "Custom"
        data_ready = False

        data_source = st.radio("Select Data Source", ["Logic Gate", "Upload CSV"], horizontal=True)

        if data_source == "Logic Gate":
            gate_name = st.selectbox("Select Logic Gate", list(GATES.keys()))
            gate_info = GATES[gate_name]
            raw = gate_info["data"]
            X = np.array([[r[0], r[1]] for r in raw], dtype=float)
            y = np.array([r[2] for r in raw], dtype=int)
            feature_cols = ["X1", "X2"]
            data_ready = True

            if not gate_info["separable"]:
                st.warning(
                    "**XOR is not linearly separable.** A single perceptron cannot "
                    "achieve zero error."
                )

            if st.session_state.last_gate != gate_name:
                st.session_state.last_gate = gate_name
                _reset_state()

            st.dataframe(pd.DataFrame(raw, columns=["X1", "X2", "Output"]), hide_index=True, use_container_width=True)

        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is None:
                st.info("Upload a CSV with numeric features and a binary target.")
            else:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.dataframe(df_uploaded.head(), hide_index=True)
                    all_cols = list(df_uploaded.columns)
                    feature_cols_sel = st.multiselect("Feature columns (X)", all_cols)
                    target_col_sel = st.selectbox("Target column (y)", ["— select —"] + [c for c in all_cols if c not in feature_cols_sel])
                    
                    if feature_cols_sel and target_col_sel != "— select —":
                        vr = validate_dataset(df_uploaded, feature_cols_sel, target_col_sel)
                        if vr.valid:
                            X, y, feature_cols = vr.X, vr.y, vr.feature_cols
                            data_ready = True
                            st.success(f"✅ Ready: {vr.n_samples} samples.")
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")

        if data_ready:
            st.divider()
            n_features = X.shape[1]
            c1, c2 = st.columns(2)
            with c1:
                learning_rate = st.number_input("Learning Rate (η)", value=0.1, step=0.01)
            with c2:
                epochs = st.slider("Maximum Epochs", 10, 2000, 100)

            mode = st.radio("Weight Initialization", ["Random", "Manual"], horizontal=True)
            weights_init = np.zeros(n_features)
            bias_init = 0.0

            if mode == "Manual":
                with st.expander("Manual Weight Input", expanded=True):
                    input_cols = st.columns(min(n_features + 1, 4))
                    for i in range(n_features):
                        weights_init[i] = input_cols[i % len(input_cols)].number_input(f"w{i+1}", value=0.0, step=0.1)
                    bias_init = input_cols[n_features % len(input_cols)].number_input("Bias", value=0.0, step=0.1)

            btn_col, reset_col = st.columns([4, 1])
            with reset_col:
                if st.button("🔄 Reset"):
                    _reset_state()
                    st.rerun()

            log_expander = st.expander("📋 Live Training Log", expanded=False)
            with log_expander:
                log_placeholder = st.empty()

            if st.session_state.training_log:
                render_log(log_placeholder, st.session_state.training_log)

            with btn_col:
                train_clicked = st.button("▶ Train Perceptron", type="primary", use_container_width=True)

            if train_clicked:
                if mode == "Random":
                    w_init = np.random.uniform(-1, 1, n_features)
                    b_init_val = random.uniform(-1, 1)
                else:
                    w_init, b_init_val = weights_init.copy(), float(bias_init)

                w_final, b_final, losses, log_lines, converged, conv_epoch = train_perceptron(X, y, w_init, b_init_val, learning_rate, epochs)
                
                st.session_state.update({
                    "weights": w_final, "bias": b_final, "losses": losses, "trained": True,
                    "training_log": log_lines, "converged": converged, "converged_epoch": conv_epoch,
                    "n_features": n_features, "feature_cols": feature_cols, "X_train": X, "y_train": y
                })
                
                render_log(log_placeholder, log_lines)
                if converged:
                    st.success(f"✅ Converged at epoch **{conv_epoch}**!")
                else:
                    st.warning(f"⚠️ Did not converge. Need more epochs or non-linearly separable.")

    with tab_analysis:
        if not (st.session_state.trained and st.session_state.weights is not None):
            st.info("Train the model in the Experiment tab first to view Analysis.")
        else:
            w, b = st.session_state.weights, st.session_state.bias
            X_tr, y_tr = st.session_state.X_train, st.session_state.y_train
            fcols, n_feat = st.session_state.feature_cols, st.session_state.n_features

            correct, total = compute_accuracy(X_tr, y_tr, w, b)
            accuracy = correct / total * 100

            st.subheader("Result Interpretation & Metrics")
            
            # Advice based on results
            interpret_results("Perceptron", {
                "accuracy": accuracy,
                "converged": st.session_state.converged
            })

            all_metrics = [(f"w{i+1}", f"{w[i]:.4f}") for i in range(n_feat)] + [("Bias", f"{b:.4f}"), ("Accuracy", f"{accuracy:.1f}%")]
            cols = st.columns(len(all_metrics))
            for col, (label, val) in zip(cols, all_metrics):
                col.metric(label, val)

            sub_tabs = st.tabs(["📉 Loss Curve", "📊 Decision Boundary", "⛰️ 3D Loss Surface", "🔍 Predictions"])
            with sub_tabs[0]:
                st.plotly_chart(plot_loss_curve(st.session_state.losses), use_container_width=True)
            with sub_tabs[1]:
                if n_feat == 1:
                    st.plotly_chart(plot_1d_threshold(X_tr, y_tr, w, b, fcols), use_container_width=True)
                elif n_feat == 2:
                    st.plotly_chart(plot_decision_boundary_2d(X_tr, y_tr, w, b, fcols), use_container_width=True)
                else:
                    st.info("Cannot plot boundary for >2 features.")
            with sub_tabs[2]:
                if n_feat >= 2:
                    st.plotly_chart(plot_loss_surface_3d(X_tr, y_tr, w, b), use_container_width=True)
                else:
                    st.info("3D surface requires at least 2 weights.")
            with sub_tabs[3]:
                st.dataframe(build_prediction_table(X_tr, y_tr, w, b, fcols), hide_index=True, use_container_width=True)

                # ══════════════════════════════════════════════════════════════════════════
                # LIVE PREDICTION
                # ══════════════════════════════════════════════════════════════════════════
                st.divider()
                st.subheader("Try a Prediction")
                st.caption("Enter input values to test the trained perceptron.")

                # Step size: binary (gate or binary CSV cols) → 1, continuous CSV → 0.01
                is_binary_data = data_source == "Logic Gate" or (
                    X_tr is not None and np.all(np.isin(X_tr, [0, 1]))
                )

                pred_cols_ui = st.columns(min(n_feat, 4))
                pred_inputs = []
                for i in range(n_feat):
                    col_idx = i % len(pred_cols_ui)
                    if data_source == "Logic Gate":
                        # Hard-lock to 0 or 1 only
                        val = pred_cols_ui[col_idx].selectbox(
                            fcols[i], options=[0, 1], key=f"pi_{i}"
                        )
                    elif is_binary_data:
                        val = pred_cols_ui[col_idx].number_input(
                            fcols[i], value=0.0, min_value=0.0, max_value=1.0,
                            step=1.0, format="%.0f", key=f"pi_{i}"
                        )
                    else:
                        val = pred_cols_ui[col_idx].number_input(
                            fcols[i], value=0.0, step=0.01, format="%.4f", key=f"pi_{i}"
                        )
                    pred_inputs.append(val)

                if st.button("🔍 Predict", type="primary"):
                    x_in = np.array(pred_inputs)
                    ws = float(np.dot(w, x_in) + b)
                    pred = 1 if ws >= 0 else 0
                    color = "#22C55E" if pred == 1 else "#EF4444"
                    st.markdown(
                        f"<h3 style='color:{color}'>Predicted Class: {pred}</h3>",
                        unsafe_allow_html=True
                    )
                    with st.expander("Computation Breakdown"):
                        terms = " + ".join([f"({w[i]:.4f} × {x_in[i]:.4f})" for i in range(n_feat)])
                        st.code(
                            f"weighted_sum = {terms} + ({b:.4f})\n"
                            f"           = {ws:.6f}\n"
                            f"step({'≥0' if ws >= 0 else '<0'})  →  class {pred}",
                            language="text"
                        )