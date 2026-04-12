import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
# UI BREAKPOINT CONSTANTS  (limits on rendering, NOT on math)
# ══════════════════════════════════════════════════════════════════════════════

DIAGRAM_MAX_NODES   = 6   # nodes per layer — beyond this, layer is collapsed
DIAGRAM_MAX_LAYERS  = 6   # total layers    — beyond this, diagram is skipped
MANUAL_MAX_NEURONS  = 6   # hidden neurons  — beyond this, manual mode disabled
MANUAL_MAX_INPUTS   = 6   # input features  — beyond this, manual mode disabled

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "Sigmoid": {
        "fn":      lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "formula": "sigmoid(z) = 1 / (1 + e^-z)",
    },
    "ReLU": {
        "fn":      lambda z: np.maximum(0, z),
        "formula": "ReLU(z) = max(0, z)",
    },
    "Tanh": {
        "fn":      lambda z: np.tanh(z),
        "formula": "tanh(z) = (e^z - e^-z) / (e^z + e^-z)",
    },
    "Linear": {
        "fn":      lambda z: z,
        "formula": "f(z) = z  (no activation)",
    },
}


def apply_activation(z, name):
    return ACTIVATIONS[name]["fn"](z)


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK DIAGRAM  — multi-layer, adaptive sizing, collapse for large layers
# ══════════════════════════════════════════════════════════════════════════════

def _node_label(l_idx, n_idx, n_layers):
    if l_idx == 0:
        return f"x{n_idx+1}"
    elif l_idx == n_layers - 1:
        return "y"
    else:
        return f"h{n_idx+1}"


def draw_network(layer_sizes, layer_labels, layer_vals=None):
    """
    Draw a fully-connected network for any number of layers.
    - Node size and canvas height scale with max nodes per layer
    - Layers beyond DIAGRAM_MAX_NODES are collapsed (first 3 + ... + last)
    - Canvas width scales with number of layers
    """
    n_layers  = len(layer_sizes)
    max_nodes = max(layer_sizes)

    # Adaptive sizing
    node_size = max(16, min(36, int(160 / max(max_nodes, 1))))
    font_size = max(7,  min(11, int(node_size * 0.28)))
    y_margin  = 0.06
    x_margin  = 0.07

    height = max(300, max_nodes * (node_size + 12) + 100)
    width  = max(420, n_layers * 150)

    layer_xs = (
        np.linspace(x_margin, 1 - x_margin, n_layers).tolist()
        if n_layers > 1 else [0.5]
    )

    # Build node list per layer: (x, y, label, val, is_ellipsis)
    node_positions = []
    COLLAPSE = DIAGRAM_MAX_NODES

    for l_idx, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
        vals = (layer_vals[l_idx] if layer_vals is not None else None)
        positions = []

        if n_nodes <= COLLAPSE:
            ys = np.linspace(1 - y_margin, y_margin, n_nodes) if n_nodes > 1 else [0.5]
            for n_idx, y in enumerate(ys):
                label = _node_label(l_idx, n_idx, n_layers)
                val   = vals[n_idx] if vals is not None and n_idx < len(vals) else None
                positions.append((lx, float(y), label, val, False))
        else:
            show_top = 3
            ys = np.linspace(1 - y_margin, y_margin, show_top + 2)
            for n_idx in range(show_top):
                label = _node_label(l_idx, n_idx, n_layers)
                val   = vals[n_idx] if vals is not None else None
                positions.append((lx, float(ys[n_idx]), label, val, False))
            # Ellipsis
            positions.append((lx, float(ys[show_top]), "...", None, True))
            # Last node
            label = _node_label(l_idx, n_nodes - 1, n_layers)
            val   = vals[n_nodes - 1] if vals is not None else None
            positions.append((lx, float(ys[show_top + 1]), label, val, False))

        node_positions.append(positions)

    # Color palette
    palette = [
        ("#3B82F6", "#1D4ED8"),   # input  — blue
        ("#8B5CF6", "#6D28D9"),   # hidden — purple
        ("#F59E0B", "#B45309"),   # hidden — amber
        ("#EC4899", "#9D174D"),   # hidden — pink
        ("#06B6D4", "#0E7490"),   # hidden — cyan
        ("#84CC16", "#3F6212"),   # hidden — lime
        ("#10B981", "#047857"),   # output — green
    ]

    fig = go.Figure()

    # Edges
    for l_idx in range(n_layers - 1):
        for (x0, y0, _, _, ell_s) in node_positions[l_idx]:
            for (x1, y1, _, _, ell_d) in node_positions[l_idx + 1]:
                if ell_s or ell_d:
                    continue
                fig.add_shape(
                    type="line",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    xref="paper", yref="paper",
                    line=dict(color="#CBD5E1", width=0.8),
                    layer="below",
                )

    # Nodes
    for l_idx, positions in enumerate(node_positions):
        if l_idx == 0:
            fill, border = palette[0]
        elif l_idx == n_layers - 1:
            fill, border = palette[-1]
        else:
            fill, border = palette[1 + ((l_idx - 1) % (len(palette) - 2))]

        for (nx, ny, label, val, is_ellipsis) in positions:
            if is_ellipsis:
                fig.add_trace(go.Scatter(
                    x=[nx], y=[ny],
                    mode="text",
                    text=["..."],
                    textfont=dict(size=14, color="#94A3B8"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                continue

            display = label if val is None else f"{label}\n{val:.3f}"
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

    # Layer labels at bottom
    for lx, lbl in zip(layer_xs, layer_labels):
        fig.add_annotation(
            x=lx, y=-0.05,
            text=f"<b>{lbl}</b>",
            showarrow=False,
            font=dict(size=10, color="#374151"),
            xanchor="center",
            xref="paper", yref="paper",
        )

    # Collapsed layer badges
    for l_idx, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
        if n_nodes > COLLAPSE:
            fig.add_annotation(
                x=lx, y=1.08,
                text=f"n={n_nodes}",
                showarrow=False,
                font=dict(size=9, color="#6B7280"),
                xanchor="center",
                xref="paper", yref="paper",
                bgcolor="#F3F4F6",
                bordercolor="#D1D5DB",
                borderwidth=1,
            )

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=30, b=55),
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[-0.1, 1.15]),
        plot_bgcolor="#F9FAFB",
        paper_bgcolor="#F9FAFB",
    )
    return fig


def plot_fwd_network_3d(in_dim, hidden_sizes, layer_A):
    layers = [in_dim] + hidden_sizes + [1]
    
    x_coords, y_coords, z_coords = [], [], []
    node_colors = []
    node_sizes = []
    hover_texts = []
    
    for l_idx, num_nodes in enumerate(layers):
        z = l_idx * 2
        activations = layer_A[l_idx].flatten() if layer_A is not None else np.zeros(num_nodes)
        
        for n_idx in range(num_nodes):
            x = n_idx - (num_nodes - 1) / 2
            y = 0
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
            act_val = float(activations[n_idx]) if n_idx < len(activations) else 0.0
            hover_texts.append(f"Layer {l_idx} Node {n_idx}<br>Activation: {act_val:.4f}")
            node_sizes.append(max(6, 6 + abs(act_val) * 15))  # Scale size with activation
            
            # Color intensity based on activation
            alpha = max(0.2, min(1.0, abs(act_val)))
            if l_idx == 0:
                node_colors.append(f'rgba(59, 130, 246, {alpha})') # Blue
            elif l_idx == len(layers) - 1:
                node_colors.append(f'rgba(239, 68, 68, {alpha})') # Red
            else:
                node_colors.append(f'rgba(16, 185, 129, {alpha})') # Green

    fig = go.Figure()
    
    # Edges
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
        mode='markers', text=hover_texts, hoverinfo='text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')),
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(projection=dict(type='orthographic'))
        ),
        title="3D Flow (Node Size/Color = Activation magnitude)",
        showlegend=False, margin=dict(l=0, r=0, b=0, t=30), height=400
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
        "fp_log":          [],
        "fp_computed":     False,
        "fp_layer_Z":      None,
        "fp_layer_A":      None,
        "fp_n_inputs":     2,
        "fp_hidden_sizes": [2],
        "fp_input_vals":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state():
    for k in [k for k in st.session_state if k.startswith("fp_")]:
        del st.session_state[k]
    _init_state()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _weight_key(n_inputs, hidden_sizes):
    return "fp_w_" + str(n_inputs) + "_" + "_".join(str(h) for h in hidden_sizes)


def _make_weights(n_inputs, hidden_sizes):
    """Returns list of (W, b) for all layers including output."""
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
# FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def forward_pass(X, weights, hidden_acts, output_act):
    """
    Full forward pass through all layers.
    Returns layer_Z (list), layer_A (list, A[0]=X).
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
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def forward_propagation_page():
    st.title("Forward Propagation")
    st.caption(
        "How inputs flow layer-by-layer through a network to produce output. "
        "No weight updates — pure inference. "
    )

    tab_theory, tab_math, tab_experiment, tab_3d, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "3D Visualization", "Analysis"
    ])

    _init_state()

    with tab_theory:
        st.subheader("What is Forward Propagation?")
        st.markdown(
            "Forward Propagation is the core operation of a neural network during inference. "
            "Data flows from the input layer, through hidden layers, to the output layer.\n\n"
            "At each neuron, two things happen:\n"
            "1. **Linear Transformation**: A weighted sum is computed from all incoming connections plus a bias.\n"
            "2. **Non-linear Activation**: This sum is passed through an activation function to introduce non-linearity."
        )

    with tab_math:
        st.subheader("The Mathematics")
        st.markdown(
            r"""
            For a given layer $l$, let $W^{[l]}$ be the weight matrix, $b^{[l]}$ the bias vector, and $A^{[l-1]}$ the activation from the previous layer (or $X$ for layer 1).

            **1. Weighted Sum ($Z^{[l]}$)**
            $$ Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]} $$

            **2. Activation ($A^{[l]}$)**
            $$ A^{[l]} = g(Z^{[l]}) $$
            where $g()$ is the assigned activation function (Sigmoid, ReLU, Tanh, etc).
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
            n = ncols[l % 5].slider(f"Layer {l+1}", 1, 20, 2, key=f"fp_hl_{l}")
            hidden_sizes.append(n)

        all_sizes  = [n_inputs] + hidden_sizes + [1]
        all_labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden_layers)] + ["Output"]

        arch_str = " → ".join([f"**{lbl}** ({sz})" for lbl, sz in zip(all_labels, all_sizes)])
        st.markdown(arch_str)

        can_draw = len(all_sizes) <= DIAGRAM_MAX_LAYERS
        if can_draw:
            st.plotly_chart(draw_network(all_sizes, all_labels), use_container_width=True, key="fp_arch_preview")
            if max(all_sizes) > DIAGRAM_MAX_NODES:
                st.caption(f"Layers with > {DIAGRAM_MAX_NODES} neurons collapsed.")
        else:
            st.info(f"Diagram skipped — > {DIAGRAM_MAX_LAYERS} layers.")

        st.divider()
        st.subheader("Input Values")
        in_cols = st.columns(min(n_inputs, 5))
        X_vals = []
        for i in range(n_inputs):
            val = in_cols[i % 5].number_input(f"x{i+1}", value=round(0.3 + i * 0.15, 2), step=0.1, key=f"fp_x{i}")
            X_vals.append(val)
        X = np.array(X_vals).reshape(-1, 1)

        st.divider()
        st.subheader("Activation Functions")
        same_act = st.checkbox("Same activation for all hidden layers", value=True)
        hidden_acts = []

        if same_act:
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                act = st.selectbox("Hidden layers", list(ACTIVATIONS.keys()), index=0)
            hidden_acts = [act] * n_hidden_layers
        else:
            act_cols = st.columns(min(n_hidden_layers, 5))
            for l in range(n_hidden_layers):
                a = act_cols[l % 5].selectbox(f"Layer {l+1}", list(ACTIVATIONS.keys()), index=0, key=f"fp_act_{l}")
                hidden_acts.append(a)

        out_col1, out_col2 = st.columns(2)
        with out_col1:
            output_act = st.selectbox("Output layer", list(ACTIVATIONS.keys()), index=3)

        st.divider()
        st.subheader("Weights")

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
                            W[j, i] = row[i].number_input(f"W[{j+1},{i+1}]", value=0.5 if i==j else 0.0, step=0.1, key=f"mw_{l_idx}_{j}_{i}")
                        b[j, 0] = row[in_sz].number_input(f"b[{j+1}]", value=0.0, step=0.1, key=f"mb_{l_idx}_{j}")
                weights_manual.append((W, b))
                in_sz = h_sz
            W_o, b_o = np.zeros((1, in_sz)), np.zeros((1, 1))
            with st.expander("Output Layer", expanded=True):
                out_row = st.columns(in_sz + 1)
                for j in range(in_sz):
                    W_o[0, j] = out_row[j].number_input(f"W_o[{j+1}]", value=1.0, step=0.1, key=f"mwo_{j}")
                b_o[0, 0] = out_row[in_sz].number_input("b_o", value=0.0, step=0.1, key="mbo")
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

        if st.session_state.fp_log:
            render_log(log_ph, st.session_state.fp_log)

        with btn_col:
            run_clicked = st.button("Run Forward Propagation", type="primary", use_container_width=True)

        if run_clicked:
            log_lines = []
            def log(line=""):
                log_lines.append(line + "\n")
                render_log(log_ph, log_lines)

            log("FORWARD PROPAGATION")
            layer_Z, layer_A = forward_pass(X, weights, hidden_acts, output_act)
            log("Running Layers...")
            for idx in range(len(weights)):
                log(f"Layer {idx+1} computed.")
            log(f"Final Output: {layer_A[-1][0][0]:.6f}")

            st.session_state.update({
                "fp_log": log_lines, "fp_computed": True, "fp_layer_Z": layer_Z, "fp_layer_A": layer_A,
                "fp_n_inputs": n_inputs, "fp_hidden_sizes": hidden_sizes, "fp_input_vals": X_vals
            })

    with tab_3d:
        if not st.session_state.fp_computed:
            st.info("Run the Forward Propagation in the Experiment tab first.")
        else:
            in_dim = st.session_state.fp_n_inputs
            h_sizes = st.session_state.fp_hidden_sizes
            layer_A = st.session_state.fp_layer_A
            st.subheader("3D Network Flow")
            st.caption("Neurons light up and scale based on their activation values.")
            
            can_draw = (in_dim <= 10 and all(h <= 10 for h in h_sizes))
            if can_draw:
                st.plotly_chart(plot_fwd_network_3d(in_dim, h_sizes, layer_A), use_container_width=True)
            else:
                st.warning("Network is too large to render efficiently in 3D.")

    with tab_analysis:
        if not st.session_state.fp_computed:
            st.info("Run the Forward Propagation in the Experiment tab first.")
        else:
            st.subheader("Analysis & Interpretation")
            layer_Z, layer_A = st.session_state.fp_layer_Z, st.session_state.fp_layer_A
            inp_vals = st.session_state.fp_input_vals
            s_n_inp = st.session_state.fp_n_inputs
            s_hid = st.session_state.fp_hidden_sizes
            s_all_sz = [s_n_inp] + s_hid + [1]
            s_all_lbl = ["Input"] + [f"Hidden {i+1}" for i in range(len(s_hid))] + ["Output"]

            st.success("Successfully computed forward propagation!")
            st.write(
                "Notice how the inputs change at every layer. The chosen activations map the resulting values. "
                "Because there are no updates or learning involved, this is simply the network predicting an output "
                "from a given input vector using static weights."
            )

            diagram_vals = [inp_vals]
            for i in range(len(s_hid)):
                diagram_vals.append(layer_A[i + 1].flatten().tolist())
            diagram_vals.append([float(layer_A[-1][0][0])])

            if len(s_all_sz) <= DIAGRAM_MAX_LAYERS:
                st.markdown("**Network Outputs (Per layer)**")
                st.plotly_chart(draw_network(s_all_sz, s_all_lbl, layer_vals=diagram_vals), use_container_width=True)

            st.markdown("**Layer-by-Layer Output**")
            tab_names = [f"Hidden {i+1}" for i in range(len(s_hid))] + ["Output"]
            tabs = st.tabs(tab_names)

            for l_idx, tab in enumerate(tabs):
                with tab:
                    Z, A = layer_Z[l_idx], layer_A[l_idx + 1]
                    is_out = l_idx == len(tab_names) - 1

                    if is_out:
                        c1, c2 = st.columns(2)
                        c1.metric("Z", f"{Z[0,0]:.4f}")
                        c2.metric("Output", f"{A[0,0]:.4f}")
                    else:
                        st.caption(f"{Z.shape[0]} neurons")
                        cols = st.columns(min(Z.shape[0], 5))
                        for j in range(Z.shape[0]):
                            with cols[j % 5]:
                                st.metric(f"Z{j+1}", f"{Z[j,0]:.4f}")
                                st.metric(f"A{j+1}", f"{A[j,0]:.4f}")