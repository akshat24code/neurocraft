import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# UI BREAKPOINT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_MAX_NEURONS = 6
MANUAL_MAX_INPUTS  = 6

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS  (identical to forward/backward pages)
# ══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "Sigmoid": {
        "fn":    lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "deriv": lambda a: a * (1 - a),
    },
    "ReLU": {
        "fn":    lambda z: np.maximum(0, z),
        "deriv": lambda a: (a > 0).astype(float),
    },
    "Tanh": {
        "fn":    lambda z: np.tanh(z),
        "deriv": lambda a: 1 - a ** 2,
    },
    "Linear": {
        "fn":    lambda z: z,
        "deriv": lambda a: np.ones_like(a),
    },
}


def apply_act(z, name):
    return ACTIVATIONS[name]["fn"](z)


def act_deriv(a, name):
    return ACTIVATIONS[name]["deriv"](a)


# ══════════════════════════════════════════════════════════════════════════════
# SOFTMAX
# ══════════════════════════════════════════════════════════════════════════════

def softmax(z):
    """Works on both (n_samples, n_classes) and (n_classes, n_samples) via axis param."""
    z_s = z - np.max(z, axis=0, keepdims=True)
    e   = np.exp(z_s)
    return e / np.sum(e, axis=0, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def standardize(df_num):
    means = df_num.mean()
    stds  = df_num.std().replace(0, 1)
    return (df_num - means) / stds, means, stds


def preprocess(df, num_cols, cat_cols, means=None, stds=None, dummy_cols=None):
    if num_cols:
        num_df = df[num_cols].astype(float)
        if means is None:
            num_scaled, means, stds = standardize(num_df)
        else:
            num_scaled = (num_df - means) / stds
    else:
        num_scaled = pd.DataFrame(index=df.index)

    if cat_cols:
        cat_df     = df[cat_cols].astype(str)
        cat_dummies = pd.get_dummies(cat_df, drop_first=False)
        if dummy_cols is not None:
            cat_dummies = cat_dummies.reindex(columns=dummy_cols, fill_value=0)
        else:
            dummy_cols = cat_dummies.columns.tolist()
    else:
        cat_dummies = pd.DataFrame(index=df.index)

    return pd.concat([num_scaled, cat_dummies], axis=1), means, stds, dummy_cols


# ══════════════════════════════════════════════════════════════════════════════
# DATASET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(n_rows, n_features, max_rows=20000, max_features=500):
    if n_rows > max_rows:
        return False, f"Too many rows: {n_rows} (max {max_rows})."
    if n_features > max_features:
        return False, f"Too many features after encoding: {n_features} (max {max_features})."
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT  — stable keys, no re-init on rerun
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(in_dim, hidden_sizes, out_dim):
    return "mlp_w_" + str(in_dim) + "_" + "_".join(str(h) for h in hidden_sizes) + "_" + str(out_dim)


def _make_weights(in_dim, hidden_sizes, out_dim):
    """He initialization for ReLU-friendly weights. Returns list of (W, b)."""
    weights, prev = [], in_dim
    for h in hidden_sizes:
        scale = np.sqrt(2.0 / prev)
        weights.append((
            np.random.randn(h, prev) * scale,
            np.zeros((h, 1)),
        ))
        prev = h
    scale = np.sqrt(2.0 / prev)
    weights.append((
        np.random.randn(out_dim, prev) * scale,
        np.zeros((out_dim, 1)),
    ))
    return weights


def _get_weights(in_dim, hidden_sizes, out_dim):
    key = _weight_key(in_dim, hidden_sizes, out_dim)
    if key not in st.session_state:
        st.session_state[key] = _make_weights(in_dim, hidden_sizes, out_dim)
    return [
        (W.copy(), b.copy())
        for W, b in st.session_state[key]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS  — W=(n_out, n_in), Z=W@A+b  (same convention as all pages)
# Convention: X fed as (n_features, n_samples) internally
# ══════════════════════════════════════════════════════════════════════════════

def forward(X_T, weights, hidden_acts, output_act, task_type):
    """
    X_T : (n_features, n_samples)
    Returns layer_A list where A[0]=X_T, and y_pred (n_samples, n_out)
    """
    layer_A = [X_T]
    A = X_T
    for l_idx, (W, b) in enumerate(weights):
        Z = W @ A + b                         # (n_out, n_samples)
        is_out = l_idx == len(weights) - 1
        if is_out:
            if task_type == "multiclass":
                A = softmax(Z)                # softmax over class axis (n_classes, n_samples)
            else:
                A = apply_act(Z, output_act)
        else:
            A = apply_act(Z, hidden_acts[l_idx])
        layer_A.append(A)
    return layer_A


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def backward(weights, layer_A, Y_T, hidden_acts, output_act, task_type, n_samples):
    """
    Y_T : (n_out, n_samples)
    Returns list of (dW, db) per layer.
    """
    grads   = [None] * len(weights)
    y_pred  = layer_A[-1]                     # (n_out, n_samples)

    # Output delta
    if task_type == "multiclass":
        dZ = (y_pred - Y_T) / n_samples       # softmax + CE combined
    else:
        dA  = y_pred - Y_T                    # dL/dA for BCE/MSE
        dZ  = dA * act_deriv(y_pred, output_act)
        dZ  = dZ / n_samples

    for l_idx in range(len(weights) - 1, -1, -1):
        A_prev = layer_A[l_idx]               # (n_in, n_samples)
        W, _   = weights[l_idx]
        dW     = dZ @ A_prev.T                # (n_out, n_in)
        db     = np.sum(dZ, axis=1, keepdims=True)
        grads[l_idx] = (dW, db)

        if l_idx > 0:
            dA_prev = W.T @ dZ                # (n_in, n_samples)
            dZ      = dA_prev * act_deriv(layer_A[l_idx], hidden_acts[l_idx - 1])

    return grads


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(y_pred_T, Y_T, task_type):
    """y_pred_T and Y_T are (n_out, n_samples)."""
    y_pred = y_pred_T.T
    y_true = Y_T.T
    if task_type == "multiclass":
        return float(-np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1)), axis=1)))
    else:
        return float(-np.mean(
            y_true * np.log(np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-9, 1))
        ))


def compute_accuracy(y_pred_T, Y_T, task_type):
    if task_type == "multiclass":
        pred_cls  = np.argmax(y_pred_T, axis=0)
        true_cls  = np.argmax(Y_T, axis=0)
    else:
        pred_cls  = (y_pred_T[0] >= 0.5).astype(int)
        true_cls  = Y_T[0].astype(int)
    return float(np.mean(pred_cls == true_cls)) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS  — light theme throughout
# ══════════════════════════════════════════════════════════════════════════════

def _light_layout(fig, title, xtitle, ytitle):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#111827")),
        xaxis=dict(title=xtitle, gridcolor="#E5E7EB", color="#374151"),
        yaxis=dict(title=ytitle, gridcolor="#E5E7EB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def plot_loss_curve(train_losses, val_losses=None):
    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses, mode="lines",
        line=dict(color="#2563EB", width=2),
        name="Train Loss",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"
    ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_losses, mode="lines",
            line=dict(color="#DC2626", width=2, dash="dash"),
            name="Val Loss"
        ))
    return _light_layout(fig, "Loss Curve", "Epoch", "Loss")

def plot_mlp_network_3d(in_dim, hidden_sizes, out_dim):
    layers = [in_dim] + hidden_sizes + [out_dim]
    max_nodes = max(layers)
    
    x_coords, y_coords, z_coords = [], [], []
    node_colors = []
    
    # Generate node coordinates
    for l_idx, num_nodes in enumerate(layers):
        z = l_idx * 2  # layer depth
        for n_idx in range(num_nodes):
            x = n_idx - (num_nodes - 1) / 2
            y = 0  # 2D projection extruded
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            if l_idx == 0:
                node_colors.append('#3b82f6')
            elif l_idx == len(layers) - 1:
                node_colors.append('#ef4444')
            else:
                node_colors.append('#10b981')

    fig = go.Figure()
    
    # Lines (edges)
    edge_x, edge_y, edge_z = [], [], []
    offset = 0
    layer_offsets = []
    for num_nodes in layers:
        layer_offsets.append(offset)
        offset += num_nodes

    for l_idx in range(len(layers) - 1):
        num_curr = layers[l_idx]
        num_next = layers[l_idx + 1]
        start_idx = layer_offsets[l_idx]
        next_idx = layer_offsets[l_idx + 1]
        
        for i in range(num_curr):
            for j in range(num_next):
                edge_x.extend([x_coords[start_idx + i], x_coords[next_idx + j], None])
                edge_y.extend([y_coords[start_idx + i], y_coords[next_idx + j], None])
                edge_z.extend([z_coords[start_idx + i], z_coords[next_idx + j], None])
                
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='rgba(156, 163, 175, 0.2)', width=1), hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=8, color=node_colors, line=dict(width=1, color='white')),
        name="Neurons"
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(projection=dict(type='orthographic'))
        ),
        title="3D MLP Architecture",
        showlegend=False, margin=dict(l=0, r=0, b=0, t=30), height=400
    )
    return fig


def plot_accuracy_curve(train_accs, val_accs=None):
    fig = go.Figure()
    epochs = list(range(1, len(train_accs) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=train_accs, mode="lines",
        line=dict(color="#16A34A", width=2),
        name="Train Accuracy"
    ))
    if val_accs:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_accs, mode="lines",
            line=dict(color="#D97706", width=2, dash="dash"),
            name="Val Accuracy"
        ))
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return _light_layout(fig, "Training Accuracy (%)", "Epoch", "Accuracy (%)")


def plot_confusion_matrix(y_true, y_pred, class_labels):
    n = len(class_labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[str(c) for c in class_labels],
        y=[str(c) for c in class_labels],
        colorscale="Blues",
        showscale=True,
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=14, color="#111827")),
        xaxis=dict(title="Predicted", color="#374151"),
        yaxis=dict(title="Actual", color="#374151", autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOG RENDERER
# ══════════════════════════════════════════════════════════════════════════════

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
        "mlp_log":            [],
        "mlp_trained":        False,
        "mlp_train_losses":   [],
        "mlp_val_losses":     [],
        "mlp_train_accs":     [],
        "mlp_val_accs":       [],
        "mlp_weights":        None,
        "mlp_task_type":      None,
        "mlp_class_labels":   None,
        "mlp_feature_cols":   None,
        "mlp_num_cols":       [],
        "mlp_cat_cols":       [],
        "mlp_cat_values":     {},
        "mlp_means":          None,
        "mlp_stds":           None,
        "mlp_dummy_cols":     None,
        "mlp_last_source":    None,
        "mlp_pred_result":    None,
        "mlp_hidden_acts":    None,
        "mlp_output_act":     None,
        "mlp_in_dim":         None,
        "mlp_out_dim":        None,
        "mlp_hidden_sizes":   None,
        "mlp_final_train_acc": None,
        "mlp_final_val_acc":  None,
        "mlp_final_loss":     None,
        "mlp_y_true_cls":     None,
        "mlp_y_pred_cls":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_train_state():
    train_keys = [
        "mlp_log", "mlp_trained", "mlp_train_losses", "mlp_val_losses",
        "mlp_train_accs", "mlp_val_accs", "mlp_weights", "mlp_pred_result",
        "mlp_final_train_acc", "mlp_final_val_acc", "mlp_final_loss",
        "mlp_y_true_cls", "mlp_y_pred_cls",
    ]
    for k in train_keys:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def mlp_page():
    st.title("Multi-Layer Perceptron (MLP)")
    st.caption(
        "Full training loop: forward pass → loss → backprop → weight update. "
        "Supports binary and multiclass classification with train/validation split."
    )

    tab_theory, tab_math, tab_experiment, tab_3d, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "3D Visualization", "Analysis"
    ])

    _init_state()

    with tab_theory:
        st.subheader("What is a Multi-Layer Perceptron?")
        st.markdown(
            "An MLP is a fully connected class of feedforward artificial neural network (ANN). "
            "It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.\n\n"
            "Except for the input nodes, each node is a neuron that uses a nonlinear activation function. "
            "MLP utilizes a supervised learning technique called backpropagation for training.\n\n"
            "**Key Use Cases:**\n"
            "- Classification tasks (e.g. distinguishing irises)\n"
            "- Learning non-linear representations (e.g. the XOR logic gate)"
        )

    with tab_math:
        st.subheader("The Mathematics")
        st.markdown(
            r"""
            **1. Forward Propagation**
            For each hidden layer \(l\):
            $$ Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]} $$
            $$ A^{[l]} = g^{[l]}(Z^{[l]}) $$

            **2. Softmax Output (for Multiclass classification)**
            $$ \hat{y}_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

            **3. Loss Functions**
            - **Binary Cross-Entropy (BCE):**
            $$ L = -\frac{1}{m} \sum_{i=1}^m [ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) ] $$
            - **Categorical Cross-Entropy (CCE):**
            $$ L = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{i,k} \log(\hat{y}_{i,k}) $$

            **4. Backpropagation Update**
            $$ W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W} $$
            """
        )

    with tab_experiment:
        data_source = st.radio("Data Source", ["Logic Gate", "Upload CSV"], horizontal=True)

        if st.session_state.mlp_last_source != data_source:
            st.session_state.mlp_last_source = data_source
            _reset_train_state()

        GATES = {
            "AND": [(0,0,0),(0,1,0),(1,0,0),(1,1,1)],
            "OR":  [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],
            "XOR": [(0,0,0),(0,1,1),(1,0,1),(1,1,0)],
        }

        X = Y = None
        data_ready  = False
        task_type   = None
        class_labels = None
        feature_cols = None
        num_cols     = []
        cat_cols     = []
        cat_values   = {}
        means = stds = dummy_cols = None

        if data_source == "Logic Gate":
            gate = st.selectbox("Logic Gate", list(GATES.keys()))
            raw  = GATES[gate]
            X    = np.array([[r[0], r[1]] for r in raw], dtype=float)
            Y    = np.array([[r[2]] for r in raw], dtype=float)
            task_type    = "binary"
            class_labels = [0, 1]
            feature_cols = ["X1", "X2"]
            data_ready   = True

            st.subheader(f"{gate} Truth Table")
            st.dataframe(pd.DataFrame(raw, columns=["X1", "X2", "Output"]), hide_index=True, use_container_width=True)
        else:
            csv_src = st.radio("CSV Source", ["Upload CSV", "Use Sample Iris Dataset"], horizontal=True)
            df_raw = None

            if csv_src == "Use Sample Iris Dataset":
                try:
                    df_raw = pd.read_csv("data/IRIS.csv")
                    st.info("Loaded: IRIS.csv")
                except FileNotFoundError:
                    st.error("data/IRIS.csv not found.")
            else:
                up = st.file_uploader("Upload CSV", type=["csv"])
                if up:
                    try:
                        df_raw = pd.read_csv(up)
                    except Exception as e:
                        st.error(f"Could not parse CSV: {e}")

            if df_raw is not None:
                if df_raw.empty:
                    st.error("Uploaded CSV is empty.")
                elif len(df_raw.columns) < 2:
                    st.error("CSV must have at least 2 columns.")
                else:
                    st.subheader("Data Preview")
                    st.dataframe(df_raw.head(10), hide_index=True, use_container_width=True)

                    with st.expander("Dataset Statistics", expanded=False):
                        st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
                        st.dataframe(df_raw.describe().round(4), use_container_width=True)

                    target_col = st.selectbox("Target column", ["— select —"] + list(df_raw.columns))

                    if target_col != "— select —":
                        df_model = df_raw.dropna(subset=[target_col]).copy()
                        feature_cols = [c for c in df_model.columns if c != target_col]

                        if not feature_cols:
                            st.error("No feature columns remain.")
                        else:
                            y_raw = df_model[target_col]
                            X_df  = df_model[feature_cols]

                            unique_tgts = y_raw.dropna().unique()
                            n_classes   = len(unique_tgts)

                            if n_classes < 2:
                                st.error("Target must have at least 2 unique classes.")
                            elif n_classes > 50:
                                st.error("Target has too many classes (>50).")
                            else:
                                task_type = "binary" if n_classes == 2 else "multiclass"
                                st.info(f"Task: **{task_type}** | Classes: **{n_classes}** | Samples: **{len(df_model)}**")

                                num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
                                cat_cols = [c for c in feature_cols if c not in num_cols]
                                for c in cat_cols:
                                    cat_values[c] = X_df[c].astype(str).unique().tolist()

                                try:
                                    X_feat, means, stds, dummy_cols = preprocess(X_df, num_cols, cat_cols)
                                except MemoryError:
                                    st.error("Out of memory during preprocessing.")
                                    X_feat = None

                                if X_feat is not None:
                                    ok, msg = validate_dataset(X_feat.shape[0], X_feat.shape[1])
                                    if not ok:
                                        st.error(msg)
                                        X_feat = None
                                
                                if X_feat is not None:
                                    if task_type == "binary":
                                        if set(unique_tgts).issubset({0, 1}):
                                            Y = y_raw.astype(int).to_numpy().reshape(-1, 1)
                                            class_labels = [0, 1]
                                        else:
                                            enc, class_labels = pd.factorize(y_raw)
                                            Y = enc.reshape(-1, 1)
                                            class_labels = list(class_labels)
                                    else:
                                        enc, class_labels = pd.factorize(y_raw)
                                        Y = np.eye(len(class_labels))[enc]
                                        class_labels = list(class_labels)

                                    X = X_feat.to_numpy(dtype=float)
                                    data_ready = True
                                    st.success(f"Dataset ready — {X.shape[0]} samples, {X.shape[1]} features")

        if data_ready:
            st.divider()
            st.subheader("Network Architecture")

            in_dim  = X.shape[1]
            out_dim = Y.shape[1] if Y.ndim == 2 else 1

            c1, c2 = st.columns(2)
            with c1:
                n_hidden_layers = st.slider("Hidden layers", 1, 5, 1)
            with c2:
                st.metric("Input features", in_dim)

            hidden_sizes = []
            st.caption("Neurons per hidden layer:")
            ncols = st.columns(min(n_hidden_layers, 5))
            for l in range(n_hidden_layers):
                n = ncols[l % 5].slider(f"Layer {l+1}", 1, 64, 4, key=f"mlp_hl_{l}")
                hidden_sizes.append(n)

            arch_str = f"Input({in_dim}) → " + " → ".join([f"Hidden{i+1}({h})" for i, h in enumerate(hidden_sizes)]) + f" → Output({out_dim})"
            st.markdown(f"`{arch_str}`")

            st.divider()
            st.subheader("Activation Functions")

            a1, a2 = st.columns(2)
            with a1:
                same_act = st.checkbox("Same activation for all hidden layers", value=True)

            hidden_acts = []
            if same_act:
                with a2:
                    h_act = st.selectbox("Hidden activation", list(ACTIVATIONS.keys()), index=0)
                hidden_acts = [h_act] * n_hidden_layers
            else:
                act_cols = st.columns(min(n_hidden_layers, 5))
                for l in range(n_hidden_layers):
                    a = act_cols[l % 5].selectbox(f"Layer {l+1}", list(ACTIVATIONS.keys()), index=0, key=f"mlp_act_{l}")
                    hidden_acts.append(a)

            if task_type == "multiclass":
                st.info("Output activation: **Softmax** (fixed for multiclass)")
                output_act = "Softmax"
            else:
                out_act_col, _ = st.columns(2)
                with out_act_col:
                    output_act = st.selectbox("Output activation", ["Sigmoid", "Linear"], index=0)

            st.divider()
            st.subheader("Hyperparameters")

            h1, h2, h3, h4 = st.columns(4)
            with h1:
                learning_rate = st.number_input("Learning Rate (η)", value=0.1, max_value=1.0, step=0.01)
            with h2:
                epochs = st.slider("Epochs", 100, 10000, 1000, step=100)
            with h3:
                val_split = st.slider("Validation split %", 0, 40, 20, step=5) / 100
            with h4:
                early_stop_patience = st.number_input("Early stop patience", value=50, max_value=500, step=10)

            n_samples = X.shape[0]
            if val_split > 0 and n_samples >= 10:
                n_val = max(1, int(n_samples * val_split))
                idx = np.random.RandomState(42).permutation(n_samples)
                val_idx, train_idx = idx[:n_val], idx[n_val:]
                X_train, Y_train = X[train_idx], Y[train_idx]
                X_val, Y_val = X[val_idx], Y[val_idx]
                st.caption(f"Train: {len(train_idx)} samples · Val: {len(val_idx)} samples")
            else:
                X_train, Y_train = X, Y
                X_val = Y_val = None
                if val_split > 0:
                    st.warning("Dataset too small for validation split — training on all data.")

            st.divider()
            btn_col, reset_col = st.columns([4, 1])
            with reset_col:
                if st.button("Reset", use_container_width=True):
                    _reset_train_state()
                    st.rerun()

            log_exp = st.expander("Training Log", expanded=False)
            with log_exp:
                log_ph = st.empty()

            if st.session_state.mlp_log:
                render_log(log_ph, st.session_state.mlp_log)

            with btn_col:
                train_clicked = st.button("Train MLP", type="primary", use_container_width=True)

            if train_clicked:
                X_tr_T, Y_tr_T = X_train.T, Y_train.T
                X_vl_T, Y_vl_T = (X_val.T, Y_val.T) if X_val is not None else (None, None)

                wkey = _weight_key(in_dim, hidden_sizes, out_dim)
                st.session_state[wkey] = _make_weights(in_dim, hidden_sizes, out_dim)
                weights = _get_weights(in_dim, hidden_sizes, out_dim)

                log_lines = []
                train_losses, val_losses, train_accs, val_accs = [], [], [], []
                best_val_loss = float("inf")
                patience_ctr = 0
                best_weights = None

                def log(line=""):
                    log_lines.append(line + "\n")
                    render_log(log_ph, log_lines)

                log("MLP TRAINING")
                log_interval = max(1, epochs // 200)

                for epoch in range(epochs):
                    layer_A = forward(X_tr_T, weights, hidden_acts, output_act, task_type)
                    loss = compute_loss(layer_A[-1], Y_tr_T, task_type)
                    acc = compute_accuracy(layer_A[-1], Y_tr_T, task_type)

                    grads = backward(weights, layer_A, Y_tr_T, hidden_acts, output_act, task_type, X_train.shape[0])
                    for l_idx, ((W, b), (dW, db)) in enumerate(zip(weights, grads)):
                        weights[l_idx] = (W - learning_rate * dW, b - learning_rate * db)

                    train_losses.append(loss); train_accs.append(acc)

                    v_loss_val = None
                    if X_vl_T is not None:
                        vl_A = forward(X_vl_T, weights, hidden_acts, output_act, task_type)
                        v_loss_val = compute_loss(vl_A[-1], Y_vl_T, task_type)
                        val_losses.append(v_loss_val)
                        val_accs.append(compute_accuracy(vl_A[-1], Y_vl_T, task_type))

                        if early_stop_patience > 0:
                            if v_loss_val < best_val_loss - 1e-6:
                                best_val_loss, patience_ctr, best_weights = v_loss_val, 0, [(W.copy(), b.copy()) for W, b in weights]
                            else:
                                patience_ctr += 1
                            if patience_ctr >= early_stop_patience:
                                weights = best_weights
                                log(f"Early stop at epoch {epoch+1}")
                                train_losses, val_losses, train_accs, val_accs = train_losses[:epoch+1], val_losses[:epoch+1], train_accs[:epoch+1], val_accs[:epoch+1]
                                break

                    if epoch % log_interval == 0 or epoch == epochs - 1:
                        v_str = f"  Val Loss: {v_loss_val:.6f}" if v_loss_val else ""
                        log(f"Epoch {epoch+1:>5}/{epochs}  |  Train Loss: {loss:.6f}  Acc: {acc:.1f}%{v_str}")

                final_layer_A = forward(X_tr_T, weights, hidden_acts, output_act, task_type)
                final_train_acc = compute_accuracy(final_layer_A[-1], Y_tr_T, task_type)
                final_val_acc = compute_accuracy(forward(X_vl_T, weights, hidden_acts, output_act, task_type)[-1], Y_vl_T, task_type) if X_vl_T is not None else None

                y_pred_cls = np.argmax(final_layer_A[-1], axis=0).tolist() if task_type == "multiclass" else (final_layer_A[-1][0] >= 0.5).astype(int).tolist()
                y_true_cls = np.argmax(Y_tr_T, axis=0).tolist() if task_type == "multiclass" else Y_tr_T[0].astype(int).tolist()

                log(f"\nDone. Final Train Loss: {train_losses[-1]:.6f} Acc: {final_train_acc:.1f}%")

                st.session_state.update({
                    "mlp_log": log_lines, "mlp_trained": True, "mlp_train_losses": train_losses, "mlp_val_losses": val_losses,
                    "mlp_train_accs": train_accs, "mlp_val_accs": val_accs, "mlp_weights": weights, "mlp_task_type": task_type,
                    "mlp_class_labels": class_labels, "mlp_feature_cols": feature_cols, "mlp_num_cols": num_cols,
                    "mlp_cat_cols": cat_cols, "mlp_cat_values": cat_values, "mlp_means": means, "mlp_stds": stds,
                    "mlp_dummy_cols": dummy_cols, "mlp_hidden_acts": hidden_acts, "mlp_output_act": output_act,
                    "mlp_in_dim": in_dim, "mlp_out_dim": out_dim, "mlp_hidden_sizes": hidden_sizes,
                    "mlp_final_train_acc": final_train_acc, "mlp_final_val_acc": final_val_acc, "mlp_final_loss": train_losses[-1],
                    "mlp_y_true_cls": y_true_cls, "mlp_y_pred_cls": y_pred_cls
                })

                if final_train_acc == 100.0:
                    st.success("Converged! Train Accuracy: 100%")
                else:
                    st.info(f"Training complete. Train Accuracy: {final_train_acc:.1f}%")

    with tab_3d:
        if not st.session_state.mlp_trained:
            st.info("Train the MLP in the Experiment tab first to view 3D insights.")
        else:
            in_dim = st.session_state.mlp_in_dim
            out_dim = st.session_state.mlp_out_dim
            hidden_sizes = st.session_state.mlp_hidden_sizes
            
            st.subheader("Network Architecture Topology")
            st.caption("A 3D node-link representation of your Multi-Layer Perceptron.")
            st.plotly_chart(plot_mlp_network_3d(in_dim, hidden_sizes, out_dim), use_container_width=True)
            
            if in_dim == 2:
                st.subheader("Decision Boundary")
                st.caption("For 2D inputs, we can visualize the learned decision boundary.")
                # We can do a quick scatter plot
                # (You could expand this to a full 3D loss surface or contour plot as needed later)
                st.info("The decision boundary is complex for MLPs. View the Analysis tab for the Confusion Matrix!")

    with tab_analysis:
        if not st.session_state.mlp_trained:
            st.info("Train the MLP in the Experiment tab first.")
        else:
            st.subheader("Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Train Loss", f"{st.session_state.mlp_final_loss:.4f}")
            m2.metric("Train Accuracy", f"{st.session_state.mlp_final_train_acc:.1f}%")
            if st.session_state.mlp_final_val_acc is not None:
                m3.metric("Val Accuracy", f"{st.session_state.mlp_final_val_acc:.1f}%")
                m4.metric("Final Val Loss", f"{st.session_state.mlp_val_losses[-1]:.4f}")
            else:
                m3.metric("Val Accuracy", "—")
                m4.metric("Final Val Loss", "—")

            sub_tab_loss, sub_tab_acc, sub_tab_cm, sub_tab_pred = st.tabs(["Loss Curve", "Accuracy Curve", "Confusion Matrix", "Prediction"])

            with sub_tab_loss:
                st.plotly_chart(plot_loss_curve(st.session_state.mlp_train_losses, st.session_state.mlp_val_losses or None), use_container_width=True)

            with sub_tab_acc:
                st.plotly_chart(plot_accuracy_curve(st.session_state.mlp_train_accs, st.session_state.mlp_val_accs or None), use_container_width=True)

            with sub_tab_cm:
                st.caption("Computed on training data.")
                st.plotly_chart(plot_confusion_matrix(st.session_state.mlp_y_true_cls, st.session_state.mlp_y_pred_cls, st.session_state.mlp_class_labels), use_container_width=True)

            with sub_tab_pred:
                st.caption("Enter feature values to get a prediction from the trained MLP.")
                weights, h_acts, o_act, t_type = st.session_state.mlp_weights, st.session_state.mlp_hidden_acts, st.session_state.mlp_output_act, st.session_state.mlp_task_type
                c_labels, f_cols, s_num_cols, s_cat_cols = st.session_state.mlp_class_labels, st.session_state.mlp_feature_cols, st.session_state.mlp_num_cols, st.session_state.mlp_cat_cols
                s_cat_vals, s_means, s_stds, s_dummy = st.session_state.mlp_cat_values, st.session_state.mlp_means, st.session_state.mlp_stds, st.session_state.mlp_dummy_cols

                if st.session_state.mlp_last_source == "Logic Gate":
                    pcols = st.columns(2)
                    X_test = np.array([[pcols[0].selectbox("X1", [0, 1]), pcols[1].selectbox("X2", [0, 1])]], dtype=float)
                else:
                    input_vals = {}
                    pcols = st.columns(min(len(f_cols), 4))
                    for i, col in enumerate(f_cols):
                        if col in s_cat_cols:
                            input_vals[col] = pcols[i % 4].selectbox(col, s_cat_vals.get(col, []))
                        else:
                            input_vals[col] = pcols[i % 4].number_input(col, value=float(s_means[col]) if s_means is not None and col in s_means else 0.0, step=0.1)
                    X_test = preprocess(pd.DataFrame([input_vals]), s_num_cols, s_cat_cols, s_means, s_stds, s_dummy)[0].to_numpy(dtype=float)

                if st.button("Predict", type="primary"):
                    layer_A = forward(X_test.T, weights, h_acts, o_act, t_type)
                    y_out = layer_A[-1]

                    if t_type == "multiclass":
                        probs = softmax(y_out).flatten()
                        pred_label = c_labels[int(np.argmax(probs))]
                        st.success(f"Predicted Class: **{pred_label}**")
                        st.dataframe(pd.DataFrame({"Class": [str(c) for c in c_labels], "Probability": [f"{p:.4f}" for p in probs]}), hide_index=True, use_container_width=True)
                    else:
                        prob = float(y_out[0][0])
                        st.markdown(f"<h3 style='color:{'#16A34A' if prob >= 0.5 else '#DC2626'}'>Predicted: {int(prob >= 0.5)}</h3>", unsafe_allow_html=True)
                        st.caption(f"Raw output: {prob:.4f}  (threshold 0.5)")

                    with st.expander("Computation Breakdown"):
                        A = X_test.T
                        for l_idx, (W, b) in enumerate(weights):
                            Z = W @ A + b
                            A = softmax(Z) if (t_type == "multiclass" and l_idx == len(weights) - 1) else apply_act(Z, o_act if l_idx == len(weights) - 1 else h_acts[l_idx])
                            lbl = "Output" if l_idx == len(weights) - 1 else f"Hidden {l_idx+1}"
                            st.markdown(f"**{lbl}:** Z={np.round(Z.flatten()[:4],4).tolist()}... → A={np.round(A.flatten()[:4],4).tolist()}...")