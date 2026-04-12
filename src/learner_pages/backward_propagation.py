import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# UI BREAKPOINT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DIAGRAM_MAX_NODES  = 6
DIAGRAM_MAX_LAYERS = 6
MANUAL_MAX_NEURONS = 6
MANUAL_MAX_INPUTS  = 6

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS + DERIVATIVES
# ══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "Sigmoid": {
        "fn":      lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "deriv":   lambda a: a * (1 - a),          # derivative w.r.t. A (post-activation)
        "formula": "sigmoid(z) = 1/(1+e^-z)",
        "deriv_formula": "sigmoid'(a) = a*(1-a)",
    },
    "ReLU": {
        "fn":      lambda z: np.maximum(0, z),
        "deriv":   lambda a: (a > 0).astype(float), # derivative w.r.t. A
        "formula": "ReLU(z) = max(0, z)",
        "deriv_formula": "ReLU'(a) = 1 if a>0 else 0",
    },
    "Tanh": {
        "fn":      lambda z: np.tanh(z),
        "deriv":   lambda a: 1 - a ** 2,
        "formula": "tanh(z) = (e^z-e^-z)/(e^z+e^-z)",
        "deriv_formula": "tanh'(a) = 1 - a^2",
    },
    "Linear": {
        "fn":      lambda z: z,
        "deriv":   lambda a: np.ones_like(a),
        "formula": "f(z) = z",
        "deriv_formula": "f'(z) = 1",
    },
}


def apply_activation(z, name):
    return ACTIVATIONS[name]["fn"](z)


def activation_deriv(a, name):
    """Derivative of activation w.r.t. pre-activation Z, expressed in terms of A."""
    return ACTIVATIONS[name]["deriv"](a)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS + DERIVATIVES
# ══════════════════════════════════════════════════════════════════════════════

LOSSES = {
    "MSE": {
        "fn":      lambda y_pred, y_true: 0.5 * np.mean((y_pred - y_true) ** 2),
        "deriv":   lambda y_pred, y_true: y_pred - y_true,
        "formula": "L = 0.5 * (y_pred - y_true)^2",
    },
    "Binary Cross-Entropy": {
        "fn":      lambda y_pred, y_true: -np.mean(
            y_true * np.log(np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-9, 1))
        ),
        "deriv":   lambda y_pred, y_true: (
            -(y_true / np.clip(y_pred, 1e-9, 1)) +
            (1 - y_true) / np.clip(1 - y_pred, 1e-9, 1)
        ),
        "formula": "L = -(y*log(y_pred) + (1-y)*log(1-y_pred))",
    },
    "MAE": {
        "fn":      lambda y_pred, y_true: np.mean(np.abs(y_pred - y_true)),
        "deriv":   lambda y_pred, y_true: np.sign(y_pred - y_true),
        "formula": "L = |y_pred - y_true|",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PASS  (identical convention to forward_propagation.py)
# ══════════════════════════════════════════════════════════════════════════════

def forward_pass(X, weights, hidden_acts, output_act):
    """
    Z = W @ A_prev + b  for each layer.
    Returns layer_Z, layer_A  where layer_A[0] = X.
    """
    layer_Z, layer_A, A_prev = [], [X], X
    for l_idx, (W, b) in enumerate(weights):
        Z   = W @ A_prev + b
        act = output_act if l_idx == len(weights) - 1 else hidden_acts[l_idx]
        A   = apply_activation(Z, act)
        layer_Z.append(Z)
        layer_A.append(A)
        A_prev = A
    return layer_Z, layer_A


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def backward_pass(weights, layer_A, layer_Z, y_true, hidden_acts, output_act, loss_fn):
    """
    Full backpropagation through all layers.

    Convention:
        dL_dZ[l] = dL_dA[l] * activation_deriv(A[l])
        dL_dW[l] = dL_dZ[l] @ A[l-1].T
        dL_db[l] = dL_dZ[l]
        dL_dA[l-1] = W[l].T @ dL_dZ[l]

    Returns
    -------
    grads : list of dicts per layer (output → input order reversed to natural order)
        Each dict: { dL_dZ, dL_dW, dL_db, dL_dA_prev }
    dL_dA_output : gradient of loss w.r.t. output activation
    """
    n_layers = len(weights)
    grads    = [None] * n_layers

    # ── Output layer ──────────────────────────────────────────────────────────
    y_pred      = layer_A[-1]
    dL_dA_out   = LOSSES[loss_fn]["deriv"](y_pred, np.array([[y_true]]))
    dA_dZ_out   = activation_deriv(layer_A[-1], output_act)
    dL_dZ_out   = dL_dA_out * dA_dZ_out
    dL_dW_out   = dL_dZ_out @ layer_A[-2].T
    dL_db_out   = dL_dZ_out.copy()
    dL_dA_prev  = weights[-1][0].T @ dL_dZ_out

    grads[-1] = {
        "dL_dA":      dL_dA_out,
        "dA_dZ":      dA_dZ_out,
        "dL_dZ":      dL_dZ_out,
        "dL_dW":      dL_dW_out,
        "dL_db":      dL_db_out,
        "dL_dA_prev": dL_dA_prev,
    }

    # ── Hidden layers (right to left) ─────────────────────────────────────────
    dL_dA_curr = dL_dA_prev
    for l_idx in range(n_layers - 2, -1, -1):
        act        = hidden_acts[l_idx]
        dA_dZ      = activation_deriv(layer_A[l_idx + 1], act)
        dL_dZ      = dL_dA_curr * dA_dZ
        dL_dW      = dL_dZ @ layer_A[l_idx].T
        dL_db      = dL_dZ.copy()
        dL_dA_prev = weights[l_idx][0].T @ dL_dZ

        grads[l_idx] = {
            "dL_dA":      dL_dA_curr,
            "dA_dZ":      dA_dZ,
            "dL_dZ":      dL_dZ,
            "dL_dW":      dL_dW,
            "dL_db":      dL_db,
            "dL_dA_prev": dL_dA_prev,
        }
        dL_dA_curr = dL_dA_prev

    return grads, dL_dA_out


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT FLOW DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def draw_gradient_flow(layer_sizes, layer_labels, grads, layer_A):
    """
    Visualize gradient magnitude flowing backwards through the network.
    Node color intensity = |dL/dZ| magnitude (darker = larger gradient).
    Arrow direction is right-to-left to show backprop flow.
    """
    n_layers  = len(layer_sizes)
    max_nodes = max(layer_sizes)
    COLLAPSE  = DIAGRAM_MAX_NODES

    node_size = max(16, min(34, int(150 / max(max_nodes, 1))))
    font_size = max(7,  min(10, int(node_size * 0.27)))
    y_margin  = 0.06
    x_margin  = 0.07

    height = max(300, max_nodes * (node_size + 12) + 100)
    width  = max(420, n_layers * 155)

    layer_xs = (
        np.linspace(x_margin, 1 - x_margin, n_layers).tolist()
        if n_layers > 1 else [0.5]
    )

    # Compute per-neuron gradient magnitude for coloring
    # grads list is indexed [0..n_hidden_layers, output]
    # layer_A[0]=X, layer_A[1..n_hidden], layer_A[-1]=output
    grad_magnitudes = []  # per layer (None for input)
    grad_magnitudes.append(None)  # input layer has no gradient
    for l_idx in range(len(grads)):
        if grads[l_idx] is not None:
            grad_magnitudes.append(np.abs(grads[l_idx]["dL_dZ"]).flatten())
        else:
            grad_magnitudes.append(None)

    # Build node positions
    node_positions = []
    for l_idx, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
        positions = []
        if n_nodes <= COLLAPSE:
            ys = np.linspace(1 - y_margin, y_margin, n_nodes) if n_nodes > 1 else [0.5]
            for n_idx, y in enumerate(ys):
                positions.append((lx, float(y), n_idx, False))
        else:
            show_top = 3
            ys = np.linspace(1 - y_margin, y_margin, show_top + 2)
            for n_idx in range(show_top):
                positions.append((lx, float(ys[n_idx]), n_idx, False))
            positions.append((lx, float(ys[show_top]), -1, True))   # ellipsis
            positions.append((lx, float(ys[show_top+1]), n_nodes-1, False))
        node_positions.append(positions)

    palette_hidden = [
        ("#8B5CF6", "#6D28D9"),
        ("#F59E0B", "#B45309"),
        ("#EC4899", "#9D174D"),
        ("#06B6D4", "#0E7490"),
    ]

    fig = go.Figure()

    # Edges (light, background)
    for l_idx in range(n_layers - 1):
        for (x0, y0, _, ell_s) in node_positions[l_idx]:
            for (x1, y1, _, ell_d) in node_positions[l_idx + 1]:
                if ell_s or ell_d:
                    continue
                fig.add_shape(
                    type="line",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    xref="paper", yref="paper",
                    line=dict(color="#E2E8F0", width=0.7),
                    layer="below",
                )

    # Gradient flow arrows (output → input, right to left)
    # Draw on top of edges, between layer centers
    for l_idx in range(n_layers - 1, 0, -1):
        x_src = layer_xs[l_idx]
        x_dst = layer_xs[l_idx - 1]
        y_mid = 0.5
        # Only draw if gradient exists for this layer
        if l_idx <= len(grads) and grads[l_idx - 1] is not None:
            mag = float(np.mean(np.abs(grads[l_idx - 1]["dL_dA_prev"])))
            arrow_color = "#EF4444" if mag > 0.1 else "#F97316" if mag > 0.01 else "#FCD34D"
            # Arrow: drawn as a shape from src to dst (paper coords)
            fig.add_annotation(
                x=x_dst + (x_src - x_dst) * 0.35,
                y=y_mid + 0.06 * (l_idx % 2),
                ax=60,   # pixel offset rightward (points back toward src)
                ay=0,
                xref="paper", yref="paper",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=arrow_color,
                text=f"d={mag:.3f}",
                font=dict(size=8, color=arrow_color),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=arrow_color,
                borderwidth=1,
            )

    # Nodes
    for l_idx, positions in enumerate(node_positions):
        for (nx, ny, n_idx, is_ellipsis) in positions:
            if is_ellipsis:
                fig.add_trace(go.Scatter(
                    x=[nx], y=[ny], mode="text",
                    text=["..."], textfont=dict(size=14, color="#94A3B8"),
                    showlegend=False, hoverinfo="skip",
                ))
                continue

            # Gradient-colored nodes
            if l_idx == 0:
                fill, border = "#3B82F6", "#1D4ED8"
                label = f"x{n_idx+1}"
                val_str = ""
            elif l_idx == n_layers - 1:
                fill, border = "#10B981", "#047857"
                label = "y"
                mag = float(np.abs(grads[-1]["dL_dZ"].flatten()[0])) if grads[-1] else 0
                val_str = f"\ndZ={mag:.3f}"
            else:
                base_fill, border = palette_hidden[(l_idx - 1) % len(palette_hidden)]
                label = f"h{n_idx+1}"
                # Gradient magnitude → color intensity
                g_arr = grad_magnitudes[l_idx]
                if g_arr is not None and n_idx < len(g_arr):
                    mag = g_arr[n_idx]
                    # interpolate alpha: low grad = light, high grad = saturated
                    val_str = f"\ndZ={mag:.3f}"
                    fill = base_fill
                else:
                    fill = base_fill
                    val_str = ""

            display = f"{label}{val_str}"
            fig.add_trace(go.Scatter(
                x=[nx], y=[ny],
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=fill,
                    line=dict(color=border, width=1.5),
                ),
                text=[display],
                textposition="middle center",
                textfont=dict(size=font_size, color="white", family="monospace"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Layer labels
    for lx, lbl in zip(layer_xs, layer_labels):
        fig.add_annotation(
            x=lx, y=-0.06,
            text=f"<b>{lbl}</b>",
            showarrow=False,
            font=dict(size=10, color="#374151"),
            xanchor="center",
            xref="paper", yref="paper",
        )

    # Legend for arrows
    fig.add_annotation(
        x=0.99, y=0.99,
        text="← gradient flow",
        showarrow=False,
        font=dict(size=9, color="#EF4444"),
        xanchor="right",
        xref="paper", yref="paper",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#EF4444",
        borderwidth=1,
    )

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=30, b=55),
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[-0.1, 1.12]),
        plot_bgcolor="#F9FAFB",
        paper_bgcolor="#F9FAFB",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT MAGNITUDE BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradient_bars(grads, all_labels):
    """Bar chart of mean |dL/dW| per layer — shows vanishing/exploding gradients."""
    labels, magnitudes = [], []
    for l_idx, g in enumerate(grads):
        if g is not None:
            lbl = all_labels[l_idx + 1]   # skip input label
            mag = float(np.mean(np.abs(g["dL_dW"])))
            labels.append(lbl)
            magnitudes.append(mag)

    colors = ["#EF4444" if m > 1.0 else "#F59E0B" if m > 0.1 else "#3B82F6"
              for m in magnitudes]

    fig = go.Figure(go.Bar(
        x=labels, y=magnitudes,
        marker=dict(color=colors, line=dict(color="#1E293B", width=1)),
        text=[f"{m:.4f}" for m in magnitudes],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Mean |dL/dW| per Layer", font=dict(size=14, color="#111827")),
        xaxis=dict(title="Layer", gridcolor="#E5E7EB", color="#374151"),
        yaxis=dict(title="Gradient Magnitude", gridcolor="#E5E7EB", color="#374151"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F9FAFB",
        font=dict(color="#374151"),
        margin=dict(t=50, b=40, l=50, r=20),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text="Red = exploding (>1.0)  |  Yellow = healthy (>0.1)  |  Blue = potentially vanishing (<0.1)",
        showarrow=False, font=dict(size=9, color="#6B7280"), xanchor="center",
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
        "bp_log":         [],
        "bp_computed":    False,
        "bp_grads":       None,
        "bp_layer_Z":     None,
        "bp_layer_A":     None,
        "bp_weights_old": None,
        "bp_weights_new": None,
        "bp_loss":        None,
        "bp_n_inputs":    2,
        "bp_hidden_sizes":[2],
        "bp_input_vals":  None,
        "bp_y_true":      1.0,
        "bp_all_labels":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in [k for k in st.session_state if k.startswith("bp_")]:
        del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT  (same stable-key pattern as forward_propagation.py)
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(n_inputs, hidden_sizes):
    return "bp_w_" + str(n_inputs) + "_" + "_".join(str(h) for h in hidden_sizes)


def _make_weights(n_inputs, hidden_sizes):
    weights, in_sz = [], n_inputs
    for h in hidden_sizes:
        weights.append((
            np.random.uniform(-1, 1, (h, in_sz)),
            np.random.uniform(-1, 1, (h, 1)),
        ))
        in_sz = h
    weights.append((
        np.random.uniform(-1, 1, (1, in_sz)),
        np.array([[np.random.uniform(-1, 1)]]),
    ))
    return weights


def _get_weights(n_inputs, hidden_sizes):
    key = _weight_key(n_inputs, hidden_sizes)
    if key not in st.session_state:
        st.session_state[key] = _make_weights(n_inputs, hidden_sizes)
    return st.session_state[key]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def backward_propagation_page():
    st.title("Backward Propagation")
    st.caption(
        "How a network computes gradients and updates weights using the chain rule. "
    )

    tab_theory, tab_math, tab_experiment, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "Analysis"
    ])

    _init_state()

    with tab_theory:
        st.subheader("What is Backward Propagation?")
        st.markdown(
            "Backward Propagation is the algorithm used to train neural networks. "
            "After a Forward Pass calculates the output and the loss (error), "
            "backpropagation measures how much each weight contributed to that error.\n\n"
            "**Key Concepts:**\n"
            "1. **Gradient**: A vector that gives the direction of steepest ascent of the loss function.\n"
            "2. **Chain Rule**: Calculus rule used to compute gradients layer by layer from output back to input.\n"
            "3. **Weight Update**: Adjusting the weights in the opposite direction of the gradient to reduce error."
        )

    with tab_math:
        st.subheader("The Mathematics")
        st.markdown(
            r"""
            **1. Chain Rule for Gradients**
            $$ \frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial Z^{[l]}} \cdot (A^{[l-1]})^T $$
            $$ \frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial Z^{[l]}} $$

            **2. Error Term per Layer**
            For the output layer:
            $$ \frac{\partial L}{\partial Z^{[L]}} = \nabla_A L \odot g'(Z^{[L]}) $$
            For hidden layers:
            $$ \frac{\partial L}{\partial Z^{[l]}} = \left( (W^{[l+1]})^T \cdot \frac{\partial L}{\partial Z^{[l+1]}} \right) \odot g'(Z^{[l]}) $$

            **3. Gradient Descent Update**
            $$ W_{new} = W_{old} - \eta \frac{\partial L}{\partial W} $$
            """
        )

    with tab_experiment:
        st.subheader("Network Architecture")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_inputs = st.slider("Input features", 1, 20, 2)
        with c2:
            n_hidden_layers = st.slider("Hidden layers", 1, 5, 1)

        st.caption("Neurons per hidden layer:")
        hidden_sizes = []
        ncols = st.columns(min(n_hidden_layers, 5))
        for l in range(n_hidden_layers):
            n = ncols[l % 5].slider(f"Layer {l+1}", 1, 20, 2, key=f"bp_hl_{l}")
            hidden_sizes.append(n)

        all_sizes  = [n_inputs] + hidden_sizes + [1]
        all_labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)] + ["Output"]

        arch_str = " → ".join([f"**{lbl}** ({sz})" for lbl, sz in zip(all_labels, all_sizes)])
        st.markdown(arch_str)

        st.divider()
        st.subheader("Input and Target")
        in_cols = st.columns(min(n_inputs + 1, 5))
        X_vals  = []
        for i in range(n_inputs):
            val = in_cols[i % 4].number_input(f"x{i+1}", value=round(0.3 + i * 0.15, 2), step=0.1, key=f"bp_x{i}")
            X_vals.append(val)
        y_true = in_cols[min(n_inputs, 4)].number_input("Target y", value=1.0, step=0.1, key="bp_ytrue")
        X = np.array(X_vals).reshape(-1, 1)

        st.divider()
        st.subheader("Hyperparameters")
        h1, h2, h3 = st.columns(3)
        with h1:
            learning_rate = st.number_input("Learning Rate (η)", value=0.1, max_value=1.0, step=0.01)
        with h2:
            loss_fn = st.selectbox("Loss Function", list(LOSSES.keys()))

        st.divider()
        st.subheader("Activation Functions")
        same_act = st.checkbox("Same activation for all hidden layers", value=True)
        hidden_acts = []

        if same_act:
            a1, a2 = st.columns(2)
            with a1:
                act = st.selectbox("Hidden layers", list(ACTIVATIONS.keys()), index=0)
            hidden_acts = [act] * n_hidden_layers
        else:
            act_cols = st.columns(min(n_hidden_layers, 5))
            for l in range(n_hidden_layers):
                a = act_cols[l % 5].selectbox(f"Layer {l+1}", list(ACTIVATIONS.keys()), index=0, key=f"bp_act_{l}")
                hidden_acts.append(a)

        o1, o2 = st.columns(2)
        with o1:
            output_act = st.selectbox("Output layer", list(ACTIVATIONS.keys()), index=3)

        st.divider()
        st.subheader("Initial Weights")
        can_manual = (n_inputs <= MANUAL_MAX_INPUTS and all(h <= MANUAL_MAX_NEURONS for h in hidden_sizes))
        mode = st.radio("Mode", ["Random", "Manual"] if can_manual else ["Random"], horizontal=True)

        weights = _get_weights(n_inputs, hidden_sizes)

        if mode == "Random":
            if st.button("🎲 Randomize Weights"):
                st.session_state[_weight_key(n_inputs, hidden_sizes)] = _make_weights(n_inputs, hidden_sizes)
                st.rerun()

            weights = _get_weights(n_inputs, hidden_sizes)
            with st.expander("Current Weights (read-only)", expanded=False):
                for l_idx, (W, b) in enumerate(weights):
                    st.dataframe(pd.DataFrame(W), use_container_width=True)
        else:
            weights_manual = []
            in_sz = n_inputs
            for l_idx, h_sz in enumerate(hidden_sizes):
                W, b = np.zeros((h_sz, in_sz)), np.zeros((h_sz, 1))
                with st.expander(f"Hidden Layer {l_idx+1}", expanded=True):
                    for j in range(h_sz):
                        row = st.columns(in_sz + 1)
                        for i in range(in_sz):
                            W[j, i] = row[i].number_input(f"W[{j+1},{i+1}]", value=0.5 if i==j else 0.0, step=0.1, key=f"bp_mw_{l_idx}_{j}_{i}")
                        b[j, 0] = row[in_sz].number_input(f"b[{j+1}]", value=0.0, step=0.1, key=f"bp_mb_{l_idx}_{j}")
                weights_manual.append((W, b))
                in_sz = h_sz
            W_o, b_o = np.zeros((1, in_sz)), np.zeros((1, 1))
            with st.expander("Output Layer", expanded=True):
                out_row = st.columns(in_sz + 1)
                for j in range(in_sz):
                    W_o[0, j] = out_row[j].number_input(f"W_o[{j+1}]", value=1.0, step=0.1, key=f"bp_mwo_{j}")
                b_o[0, 0] = out_row[in_sz].number_input("b_o", value=0.0, step=0.1, key="bp_mbo")
            weights_manual.append((W_o, b_o))
            weights = weights_manual

        st.divider()
        btn_col, reset_col = st.columns([4, 1])
        with reset_col:
            if st.button("Reset"):
                _reset_state()
                st.rerun()

        log_exp = st.expander("Computation Log", expanded=False)
        with log_exp:
            log_ph = st.empty()

        if st.session_state.bp_log:
            render_log(log_ph, st.session_state.bp_log)

        with btn_col:
            run_clicked = st.button("Run Backward Propagation", type="primary", use_container_width=True)

        if run_clicked:
            log_lines = []
            def log(line=""):
                log_lines.append(line + "\n")
                render_log(log_ph, log_lines)

            log("FORWARD PASS")
            layer_Z, layer_A = forward_pass(X, weights, hidden_acts, output_act)
            y_pred = layer_A[-1][0][0]
            loss   = LOSSES[loss_fn]["fn"](layer_A[-1], np.array([[y_true]]))
            log(f"Loss computed: {loss:.4f}")

            log("BACKWARD PASS")
            grads, dL_dA_out = backward_pass(weights, layer_A, layer_Z, y_true, hidden_acts, output_act, loss_fn)
            log("Gradients computed.")

            log("WEIGHT UPDATES")
            weights_new = []
            for l_idx, ((W, b), g) in enumerate(zip(weights, grads)):
                W_new = W - learning_rate * g["dL_dW"]
                b_new = b - learning_rate * g["dL_db"]
                weights_new.append((W_new, b_new))
            log("Updated weights.")

            st.session_state.update({
                "bp_log": log_lines, "bp_computed": True, "bp_grads": grads, 
                "bp_layer_Z": layer_Z, "bp_layer_A": layer_A, "bp_weights_old": weights, 
                "bp_weights_new": weights_new, "bp_loss": float(loss), "bp_n_inputs": n_inputs,
                "bp_hidden_sizes": hidden_sizes, "bp_input_vals": X_vals, "bp_y_true": y_true,
                "bp_all_labels": all_labels
            })

    with tab_analysis:
        if not st.session_state.bp_computed:
            st.info("Run Backward Propagation in the Experiment tab first.")
        else:
            grads, layer_A, weights_old, weights_new = st.session_state.bp_grads, st.session_state.bp_layer_A, st.session_state.bp_weights_old, st.session_state.bp_weights_new
            loss_val, s_n_inp, s_hid, s_y_true = st.session_state.bp_loss, st.session_state.bp_n_inputs, st.session_state.bp_hidden_sizes, st.session_state.bp_y_true
            s_all_labels = st.session_state.bp_all_labels
            s_all_sizes  = [s_n_inp] + s_hid + [1]
            y_pred_val = float(layer_A[-1][0][0])

            st.subheader("Result Interpretation & Analysis")
            st.write(
                "You just observed a single step of gradient descent. "
                "The loss was calculated and mapped backward through the network to update all weights."
            )
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Loss", f"{loss_val:.6f}")
            m2.metric("y_pred", f"{y_pred_val:.4f}")
            m3.metric("y_true", f"{s_y_true:.4f}")
            m4.metric("Error", f"{y_pred_val - s_y_true:+.4f}")

            sub_tabs = st.tabs(["Gradient Flow", "Gradient Magnitudes", "3D Optimization Path", "Weight Updates"])

            with sub_tabs[0]:
                st.caption("Arrows show gradient flow direction. Node labels show dL/dZ.")
                if len(s_all_sizes) <= DIAGRAM_MAX_LAYERS:
                    st.plotly_chart(draw_gradient_flow(s_all_sizes, s_all_labels, grads, layer_A), use_container_width=True)
                else:
                    st.info("Diagram skipped — exceeds limits.")

            with sub_tabs[1]:
                st.caption("Mean |dL/dW| per layer: Red (>1.0), Yellow (>0.1), Blue (<0.1)")
                st.plotly_chart(plot_gradient_bars(grads, s_all_labels), use_container_width=True)

            with sub_tabs[2]:
                st.caption("3D view of the first two weights from the first hidden layer (or output) vs. Loss.")
                # We'll visualize W[0,0] and W[0,1] of the first layer
                w_idx = 0
                w_orig = weights_old[0][0]
                w_new = weights_new[0][0]
                
                # Create a local grid around the weight
                w1_base, w2_base = w_orig[0,0], w_orig[0,1]
                w1_vals = np.linspace(w1_base - 1, w1_base + 1, 20)
                w2_vals = np.linspace(w2_base - 1, w2_base + 1, 20)
                W1, W2 = np.meshgrid(w1_vals, w2_vals)
                # Mock loss surface: parabolic around the target
                Z = (W1 - (w1_base - 0.5))**2 + (W2 - (w2_base - 0.5))**2 + (loss_val * 0.5)
                
                fig3d = go.Figure(data=[go.Surface(z=Z, x=W1, y=W2, colorscale='Viridis', opacity=0.7)])
                
                # Plot the update step
                fig3d.add_trace(go.Scatter3d(
                    x=[w1_base, w_new[0,0]],
                    y=[w2_base, w_new[0,1]],
                    z=[loss_val, loss_val * 0.9], # simulated loss drop
                    mode='lines+markers',
                    line=dict(color='red', width=5),
                    marker=dict(size=5, color=['blue', 'red']),
                    name='Update Step'
                ))
                
                fig3d.update_layout(
                    scene=dict(xaxis_title="Weight 1", yaxis_title="Weight 2", zaxis_title="Loss"),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=500
                )
                st.plotly_chart(fig3d, use_container_width=True)

            with sub_tabs[2]:
                st.caption("Weight values before and after one backward pass.")
                for l_idx, ((W_old, b_old), (W_new, b_new)) in enumerate(zip(weights_old, weights_new)):
                    lbl = "Output Layer" if l_idx == len(weights_old) - 1 else f"Hidden Layer {l_idx+1}"
                    with st.expander(lbl, expanded=(l_idx == 0)):
                        before_df = pd.DataFrame(W_old)
                        after_df = pd.DataFrame(W_new)
                        delta_df = (after_df.astype(float) - before_df.astype(float)).round(4)
                        c1, c2, c3 = st.columns(3)
                        c1.caption("Before"); c1.dataframe(before_df.round(4))
                        c2.caption("After"); c2.dataframe(after_df.round(4))
                        c3.caption("Delta (Δ)"); c3.dataframe(delta_df)