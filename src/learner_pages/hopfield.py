import streamlit as st
import numpy as np
import random
import time

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    CANVAS_AVAILABLE = False

# ==========================================
# 🧠 HOPFIELD CORE LOGIC (Self-Contained)
# ==========================================

def calculate_weight_matrix(patterns, method="hebbian"):
    """
    Calculate weight matrix using Hebbian or Pseudoinverse rule.
    Pseudoinverse is more robust for correlated patterns (like letters).
    """
    P = patterns.T  # Shape: (neurons, num_patterns)
    N = P.shape[0]
    
    if method == "pseudoinverse":
        # W = P * (P^T * P)^-1 * P^T
        # More stable for non-orthogonal patterns
        try:
            P_inv = np.linalg.pinv(P)
            W = P @ P_inv
        except np.linalg.LinAlgError:
            W = (P @ P.T) / N
    else:
        # Standard Hebbian: W = (1/N) * sum(p_i * p_i^T)
        W = (P @ P.T) / N
        
    np.fill_diagonal(W, 0)
    return W

def apply_noise(pattern, noise_level):
    """Flip random pixels based on noise_level (0.0 to 1.0)."""
    noisy = pattern.copy()
    n_pixels = len(pattern)
    n_flip = int(n_pixels * noise_level)
    indices = np.random.choice(n_pixels, n_flip, replace=False)
    noisy[indices] *= -1
    return noisy

def async_recall(input_pattern, W, max_epochs=20):
    """Asynchronous update: update one random neuron at a time."""
    s = np.array(input_pattern).flatten().astype(float)
    N = len(s)
    
    for _ in range(max_epochs):
        indices = list(range(N))
        np.random.shuffle(indices)
        changed = False
        for i in indices:
            activation = np.dot(W[i], s)
            new_val = 1 if activation >= 0 else -1
            if s[i] != new_val:
                s[i] = new_val
                changed = True
        
        # Check stability
        if not changed:
            break
            
    return s

def calculate_energy(state, W):
    """Energy E = -0.5 * s^T * W * s"""
    return -0.5 * float(state @ W @ state)


# ==========================================
# DATASET PREPARATION - IDEAL PATTERNS
# ==========================================

GRID_SIZE = 15
NEURONS = GRID_SIZE * GRID_SIZE

LETTER_BITMAPS_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _bitmap_to_pattern(bitmap_rows):
    """Convert a 5x7 bitmap to a centered 15x15 bipolar pattern."""
    small = np.array([[1 if ch == "1" else -1 for ch in row] for row in bitmap_rows], dtype=int)
    # Enlarge each pixel to improve stroke visibility on the 15x15 canvas.
    scaled = np.kron(small, np.ones((2, 2), dtype=int))
    grid = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)
    start_row = (GRID_SIZE - scaled.shape[0]) // 2
    start_col = (GRID_SIZE - scaled.shape[1]) // 2
    grid[start_row:start_row + scaled.shape[0], start_col:start_col + scaled.shape[1]] = scaled
    return grid.flatten()


def _build_alphabet_patterns():
    labels = sorted(LETTER_BITMAPS_5X7.keys())
    patterns = np.array([_bitmap_to_pattern(LETTER_BITMAPS_5X7[label]) for label in labels])
    return labels, patterns


def _shift_bipolar_grid(grid, dr, dc):
    """Shift a bipolar grid without wrap-around; uncovered pixels become background (-1)."""
    shifted = np.full_like(grid, -1)

    if dr >= 0:
        src_r = slice(0, GRID_SIZE - dr)
        dst_r = slice(dr, GRID_SIZE)
    else:
        src_r = slice(-dr, GRID_SIZE)
        dst_r = slice(0, GRID_SIZE + dr)

    if dc >= 0:
        src_c = slice(0, GRID_SIZE - dc)
        dst_c = slice(dc, GRID_SIZE)
    else:
        src_c = slice(-dc, GRID_SIZE)
        dst_c = slice(0, GRID_SIZE + dc)

    shifted[dst_r, dst_c] = grid[src_r, src_c]
    return shifted


def _dilate_bipolar_grid(grid):
    """Thicken foreground strokes by one pixel neighborhood."""
    ink = grid == 1
    padded = np.pad(ink, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(ink)

    for r in range(3):
        for c in range(3):
            dilated |= padded[r:r + GRID_SIZE, c:c + GRID_SIZE]

    return np.where(dilated, 1, -1)


def _build_classifier_variants(base_patterns):
    """Create style variants per letter for robust classification."""
    variants = []
    variant_label_idx = []
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for label_idx, flat_pattern in enumerate(base_patterns):
        grid = flat_pattern.reshape(GRID_SIZE, GRID_SIZE)
        pool = [grid]
        pool.extend(_shift_bipolar_grid(grid, dr, dc) for dr, dc in shifts)

        thick = _dilate_bipolar_grid(grid)
        pool.append(thick)
        pool.extend(_shift_bipolar_grid(thick, dr, dc) for dr, dc in shifts)

        seen = set()
        for v in pool:
            key = v.tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(v.flatten())
            variant_label_idx.append(label_idx)

    return np.array(variants, dtype=int), np.array(variant_label_idx, dtype=int)


def _aggregate_variant_scores(variant_scores, variant_to_label, num_labels):
    """Reduce per-variant scores to per-letter scores using max pooling per label."""
    label_scores = np.full(num_labels, -np.inf, dtype=np.float64)
    for idx, score in enumerate(variant_scores):
        label_idx = int(variant_to_label[idx])
        if score > label_scores[label_idx]:
            label_scores[label_idx] = score
    return label_scores


def _center_ink(grid):
    """Center the drawn foreground pixels to better match stored templates."""
    ink = grid == 1
    if not np.any(ink):
        return grid

    rows = np.where(np.any(ink, axis=1))[0]
    cols = np.where(np.any(ink, axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    glyph = grid[r0:r1 + 1, c0:c1 + 1]

    centered = np.full_like(grid, -1)
    gh, gw = glyph.shape
    start_row = (GRID_SIZE - gh) // 2
    start_col = (GRID_SIZE - gw) // 2
    centered[start_row:start_row + gh, start_col:start_col + gw] = glyph
    return centered


def _canvas_to_flat_bipolar(image_data):
    """Convert 150x150 canvas data into a centered 15x15 bipolar input pattern."""
    if image_data is None:
        return None

    gray = image_data[:, :, 0].astype(np.float32)
    # Average-pool each 10x10 patch to get a 15x15 grid from the 150x150 canvas.
    pooled = gray.reshape(GRID_SIZE, 10, GRID_SIZE, 10).mean(axis=(1, 3))

    # Lower grayscale means stronger ink stroke (black stroke on white background).
    bipolar_grid = np.where(pooled < 220.0, 1, -1)
    centered = _center_ink(bipolar_grid)
    return centered.flatten()


def _shape_similarity_scores(state, patterns):
    """Foreground-aware similarity that is less biased by matching background pixels."""
    s = np.where(np.asarray(state) >= 0, 1, -1).reshape(1, -1)
    p = np.where(np.asarray(patterns) >= 0, 1, -1)

    s_ink = s == 1
    p_ink = p == 1

    intersection = np.sum(p_ink & s_ink, axis=1).astype(np.float64)
    union = np.sum(p_ink | s_ink, axis=1).astype(np.float64)
    iou = intersection / np.maximum(union, 1.0)

    s_ink_count = float(np.sum(s_ink))
    precision = intersection / max(s_ink_count, 1.0)
    p_ink_count = np.sum(p_ink, axis=1).astype(np.float64)
    recall = intersection / np.maximum(p_ink_count, 1.0)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-9)

    dot = (p @ s.ravel()) / s.shape[1]
    dot01 = (dot + 1.0) / 2.0

    # Row/column projections capture global glyph structure.
    p_grid = p.reshape(p.shape[0], GRID_SIZE, GRID_SIZE)
    s_grid = s.reshape(GRID_SIZE, GRID_SIZE)
    p_ink_float = (p_grid == 1).astype(np.float64)
    s_ink_float = (s_grid == 1).astype(np.float64)

    p_row = np.sum(p_ink_float, axis=2)
    s_row = np.sum(s_ink_float, axis=1)
    row_dot = p_row @ s_row
    row_norm = np.linalg.norm(p_row, axis=1) * max(np.linalg.norm(s_row), 1e-9)
    row_sim = row_dot / np.maximum(row_norm, 1e-9)

    p_col = np.sum(p_ink_float, axis=1)
    s_col = np.sum(s_ink_float, axis=0)
    col_dot = p_col @ s_col
    col_norm = np.linalg.norm(p_col, axis=1) * max(np.linalg.norm(s_col), 1e-9)
    col_sim = col_dot / np.maximum(col_norm, 1e-9)

    return 0.35 * iou + 0.20 * f1 + 0.10 * dot01 + 0.20 * row_sim + 0.15 * col_sim


def _classify_with_consensus(input_state, recalled_state, classifier_patterns, classifier_variant_to_label, ideal_labels):
    """Combine raw-input and recalled-shape evidence for more reliable predictions."""
    input_variant_scores = _shape_similarity_scores(input_state, classifier_patterns)
    recall_variant_scores = _shape_similarity_scores(recalled_state, classifier_patterns)

    input_scores = _aggregate_variant_scores(
        input_variant_scores,
        classifier_variant_to_label,
        len(ideal_labels),
    )
    recall_scores = _aggregate_variant_scores(
        recall_variant_scores,
        classifier_variant_to_label,
        len(ideal_labels),
    )

    agreement = float(np.mean(np.asarray(input_state) == np.asarray(recalled_state)))
    recall_weight = 0.15 + 0.25 * agreement
    input_weight = 1.0 - recall_weight

    combined_scores = (input_weight * input_scores) + (recall_weight * recall_scores)
    best_idx = int(np.argmax(combined_scores))

    sorted_scores = np.sort(combined_scores)
    if sorted_scores.size > 1:
        margin = float(sorted_scores[-1] - sorted_scores[-2])
    else:
        margin = float(sorted_scores[-1])

    return best_idx, combined_scores, input_scores, recall_scores, margin, agreement


# Initialize A-Z patterns once globally
IDEAL_LABELS, IDEAL_PATTERNS = _build_alphabet_patterns()
W_MATRIX = calculate_weight_matrix(IDEAL_PATTERNS, method="pseudoinverse")
CLASSIFIER_VARIANTS, CLASSIFIER_V_TO_L = _build_classifier_variants(IDEAL_PATTERNS)

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

def hopfield_page():
    """Main function for the Hopfield Network module."""
    
    st.markdown('<div class="nc-section-label">Interactive Pattern Recognition</div>', unsafe_allow_html=True)
    st.title("🧠 Hopfield Network OCR")
    
    st.markdown("""
    **Draw a letter (A to Z)** in the white canvas. The Hopfield network reconstructs 
    patterns from noisy or incomplete sketches, demonstrating **associative memory** in action.
    """)
    
    if not CANVAS_AVAILABLE:
        st.error("⚠️ `streamlit-drawable-canvas` is missing. Install it to use the interactive lab.")
        st.code("pip install streamlit-drawable-canvas")
        return
    
    # Initialize session state for persistent results
    if 'hop_canvas_data' not in st.session_state:
        st.session_state.hop_canvas_data = None
    
    # UI Layout: Three functional columns
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    
    # ==========================================
    # COLUMN 1: INPUT (DRAWING)
    # ==========================================
    with col1:
        st.subheader("1️⃣ Draw Here")
        st.caption("Sketch an uppercase alphabet (A-Z)")

        # Injected CSS to highlight the canvas area
        st.markdown("""
        <style>
        iframe[title*="canvas"] {
            border: 3px solid #67e8f9 !important;
            border-radius: 12px;
            background: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=15,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=150,
            height=150,
            drawing_mode="freedraw",
            key="hop_ocr_canvas",
        )
        
        if canvas_result.image_data is not None:
            st.session_state.hop_canvas_data = canvas_result.image_data
        
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()

    # ==========================================
    # COLUMN 2: TUNING & DYNAMICS
    # ==========================================
    with col2:
        st.subheader("2️⃣ Tuning")
        st.caption("Adjust noise and recall depth")
        
        noise_level = st.slider("🎲 Noise Level (%)", 0, 50, 10, step=5)
        recall_steps = st.slider("⏳ Iterations", 5, 120, 30, step=5, help="Update epochs for async neuron updates.")
        recall_trials = st.slider("🔄 Retries", 1, 10, 5, help="Runs multiple recalls to find the most stable attractor.")

        if st.session_state.hop_canvas_data is not None:
            raw_input = _canvas_to_flat_bipolar(st.session_state.hop_canvas_data)
            if raw_input is not None:
                # Apply simulated noise for visualization
                noisy_view = apply_noise(raw_input, noise_level / 100.0)
                grid_img = np.where(noisy_view.reshape(GRID_SIZE, GRID_SIZE) == 1, 0, 255).astype(np.uint8)
                view_img = np.kron(grid_img, np.ones((8, 8))).astype(np.uint8)
                st.image(view_img, caption=f"Noisy input ({noise_level}%)", width=150)

    # ==========================================
    # COLUMN 3: RECONSTRUCTION (OUTPUT)
    # ==========================================
    with col3:
        st.subheader("3️⃣ Reconstruction")
        st.caption("Network's stable attractor")
        
        predict_btn = st.button("🔍 Predict Letter", type="primary", use_container_width=True)
        
        if predict_btn:
            if st.session_state.hop_canvas_data is None:
                st.error("❌ Draw something first!")
            else:
                raw_input = _canvas_to_flat_bipolar(st.session_state.hop_canvas_data)
                if raw_input is None or np.sum(raw_input == 1) < 4:
                    st.error("❌ Sketch is too small or blank.")
                else:
                    noisy_input = apply_noise(raw_input, noise_level / 100.0)
                    
                    # Perform multiple trials of asynchronous recall
                    results = []
                    with st.spinner("Network recalling..."):
                        for _ in range(recall_trials):
                            candidate = async_recall(noisy_input, W_MATRIX, max_epochs=recall_steps)
                            idx, combined, _, _, margin, agreement = _classify_with_consensus(
                                noisy_input, candidate, CLASSIFIER_VARIANTS, CLASSIFIER_V_TO_L, IDEAL_LABELS
                            )
                            results.append({
                                'candidate': candidate, 'idx': idx, 'score': combined[idx],
                                'margin': margin, 'agreement': agreement, 'energy': calculate_energy(candidate, W_MATRIX)
                            })
                    
                    # Select best trial based on consensus score and stability
                    best_trial = max(results, key=lambda x: (x['score'], x['margin'], -x['energy']))
                    
                    # Display the reconstructed pattern
                    out_grid = np.where(best_trial['candidate'].reshape(GRID_SIZE, GRID_SIZE) == 1, 0, 255).astype(np.uint8)
                    out_img = np.kron(out_grid, np.ones((8, 8))).astype(np.uint8)
                    st.image(out_img, caption="Reconstructed Output", width=150)
                    
                    conf = float(best_trial['score'])
                    st.success(f"### Result: **{IDEAL_LABELS[best_trial['idx']]}**")
                    st.caption(f"Confidence: {conf*100:.1f}% | Energy: {best_trial['energy']:.3f}")

                    if conf < 0.65:
                        st.warning("Low confidence. Try a cleaner drawing or more iterations.")

    # ==========================================
    # EDUCATIONAL DEEP DIVE
    # ==========================================
    st.divider()
    with st.expander("📚 How Associative Memory Works"):
        st.markdown("""
        ### From Hebbian Rule to Attractors
        1. **Storage**: The network 'memorizes' the clean A-Z patterns by building a large weight matrix where correlations between pixels are strong.
        2. **Energy Surfaces**: Each memorized pattern becomes a **valley** in a 256-dimensional energy landscape.
        3. **Dynamic Recovery**: When you draw a noisy letter, the system starts on a high ridge and 'rolls down' the gradient into the nearest valley (attractor).
        4. **Fault Tolerance**: This is why Hopfield networks are used for error correction—they can recreate a whole signal from just a few correct fragments.
        """)

if __name__ == "__main__":
    hopfield_page()
