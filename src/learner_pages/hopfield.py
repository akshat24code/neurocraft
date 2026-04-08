import streamlit as st
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image

class HopfieldNetwork:
    def __init__(self, size):
        self.num_neuron = size
        self.W = np.zeros((size, size))

    def train(self, train_data):
        train_data = [np.array(d) for d in train_data]
        self.num_neuron = train_data[0].size
        
        # The Hebbian (Mean-Centered) rule has a mathematical capacity limit of ~0.14 * N.
        # In this 49-neuron grid, storing 20 pattern variations completely breaks Hebbian math 
        # (causing memories like 6, 3, and 8 to merge together).
        # To fix this, we strictly use the Pseudo-Inverse (Projection) rule which has 
        # a perfect recall capacity limit of up to N memories (49).
        P_mat = np.array(train_data).T
        pinv_P = np.linalg.pinv(P_mat)
        self.W = np.dot(P_mat, pinv_P)
        # CRITICAL: Do NOT zero the diagonal for Pseudo-Inverse matrices, or it shatters orthogonal projection!

    def predict(self, data, num_iter=10, threshold=0):
        # We process a single vector here for the UI
        s = data.copy().astype(float)
        for _ in range(num_iter):
            # Synchronous updates are drastically more reliable for Pseudo-Inverse matrices
            s_new = np.sign(np.dot(self.W, s) - threshold)
            s_new[s_new == 0] = 1
            if np.array_equal(s, s_new):
                break # Converged
            s = s_new
        return s

def get_7x7_patterns():
    # 1: Black (+1), -1: White (-1)
    def P(matrix_str):
        return [1 if c == '1' else -1 for c in matrix_str.replace('\n', '').replace(' ', '')]
    
    # Store multiple variations per letter
    patterns = {
        "A": [
            P("0 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1"),
            P("0 0 1 1 1 0 0 0 1 1 0 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1")
        ],
        "B": [
            P("1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0"),
            P("1 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 0 0")
        ],
        "C": [
            P("0 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 1 1"),
            P("0 0 1 1 1 1 1 0 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 1")
        ],
        "D": [
            P("1 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 0 0"),
            P("1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0")
        ],
        "E": [
            P("1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1"),
            P("1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0")
        ]
    }
    return patterns

def get_7x7_number_patterns():
    def P(matrix_str):
        cleaned = matrix_str.replace('\n', '').replace(' ', '')
        if len(cleaned) != 49: print(f"Error: Size is {len(cleaned)}")
        return [1 if c == '1' else -1 for c in cleaned]
    
    patterns = {
        "0": [
            P("""
            0 0 1 1 1 0 0
            0 1 1 0 1 1 0
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 0 1 1 0
            0 0 1 1 1 0 0
            """),
            P("""
            0 1 1 1 1 1 0
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 1 1 1 0
            """)
        ],
        "1": [
            P("""
            0 0 0 1 1 0 0
            0 0 1 1 1 0 0
            0 1 1 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 1 1 1 1 0
            """),
            P("""
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            0 0 0 1 1 0 0
            """)
        ],
        "2": [
            P("""
            0 1 1 1 1 0 0
            1 1 0 0 1 1 0
            0 0 0 0 1 1 0
            0 0 0 1 1 0 0
            0 0 1 1 0 0 0
            0 1 1 0 0 0 0
            1 1 1 1 1 1 1
            """),
            P("""
            0 0 1 1 1 1 0
            0 1 1 0 0 1 1
            0 0 0 0 1 1 0
            0 0 0 1 1 0 0
            0 0 1 1 0 0 0
            0 1 1 0 0 0 0
            1 1 1 1 1 1 1
            """)
        ],
        "3": [
            P("""
            1 1 1 1 1 1 0
            0 0 0 0 1 1 0
            0 0 0 1 1 0 0
            0 0 1 1 1 0 0
            0 0 0 0 1 1 0
            1 1 0 0 1 1 0
            0 1 1 1 1 0 0
            """),
            P("""
            0 1 1 1 1 1 0
            0 0 0 0 1 1 0
            0 0 0 1 1 1 0
            0 0 0 0 0 1 1
            0 0 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 1 1 1 0
            """)
        ],
        "4": [
            P("""
            0 0 0 0 1 1 0
            0 0 0 1 1 1 0
            0 0 1 1 1 1 0
            0 1 1 0 1 1 0
            1 1 1 1 1 1 1
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            """),
            P("""
            0 0 0 1 1 0 0
            0 0 1 1 1 0 0
            0 1 1 0 1 1 0
            1 1 0 0 1 1 0
            1 1 1 1 1 1 1
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            """)
        ],
        "5": [
            P("""
            1 1 1 1 1 1 1
            1 1 0 0 0 0 0
            1 1 1 1 1 0 0
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            1 1 0 0 1 1 0
            0 1 1 1 1 0 0
            """),
            P("""
            1 1 1 1 1 1 0
            1 1 0 0 0 0 0
            1 1 1 1 1 1 0
            0 0 0 0 0 1 1
            0 0 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 1 1 1 0
            """)
        ],
        "6": [
            P("""
            0 0 1 1 1 0 0
            0 1 1 0 0 0 0
            1 1 0 0 0 0 0
            1 1 1 1 1 0 0
            1 1 0 0 1 1 0
            1 1 0 0 1 1 0
            0 1 1 1 1 0 0
            """),
            P("""
            0 0 1 1 1 1 0
            0 1 1 0 0 0 0
            1 1 0 0 0 0 0
            1 1 1 1 1 1 0
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 1 1 1 0
            """)
        ],
        "7": [
            P("""
            1 1 1 1 1 1 1
            0 0 0 0 1 1 0
            0 0 0 1 1 0 0
            0 0 1 1 0 0 0
            0 1 1 0 0 0 0
            0 1 1 0 0 0 0
            0 1 1 0 0 0 0
            """),
            P("""
            1 1 1 1 1 1 1
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            0 0 0 0 1 1 0
            """)
        ],
        "8": [
            P("""
            0 1 1 1 1 0 0
            1 1 0 0 1 1 0
            1 1 0 0 1 1 0
            0 1 1 1 1 0 0
            1 1 0 0 1 1 0
            1 1 0 0 1 1 0
            0 1 1 1 1 0 0
            """),
            P("""
            0 0 1 1 1 0 0
            0 1 1 0 1 1 0
            0 1 1 0 1 1 0
            0 0 1 1 1 0 0
            0 1 1 0 1 1 0
            0 1 1 0 1 1 0
            0 0 1 1 1 0 0
            """)
        ],
        "9": [
            P("""
            0 0 1 1 1 0 0
            0 1 1 0 1 1 0
            0 1 1 0 1 1 0
            0 0 1 1 1 1 1
            0 0 0 0 0 1 1
            0 0 0 0 0 1 1
            0 0 0 1 1 1 0
            """),
            P("""
            0 1 1 1 1 1 0
            1 1 0 0 0 1 1
            1 1 0 0 0 1 1
            0 1 1 1 1 1 1
            0 0 0 0 0 1 1
            0 0 0 0 0 1 1
            0 0 1 1 1 0 0
            """)
        ]
    }
    return patterns

def downsample_12x12_to_7x7(grid_12x12):
    out = np.zeros((7, 7), dtype=int)
    for i in range(7):
        for j in range(7):
            r_start = int(np.round(i * 12 / 7))
            r_end = int(np.round((i + 1) * 12 / 7))
            c_start = int(np.round(j * 12 / 7))
            c_end = int(np.round((j + 1) * 12 / 7))
            
            if r_end == r_start: r_end += 1
            if c_end == c_start: c_end += 1
            
            block = grid_12x12[r_start:r_end, c_start:c_end]
            if block.size == 0:
                out[i, j] = -1
            else:
                active_pct = np.sum(block == 1) / block.size
                out[i, j] = 1 if active_pct > 0.3 else -1
    return out.flatten()

def render_grid_preview(matrix, rows, cols, cell_size=20, active_color="#67e8f9", title=None, highlight_diff=None):
    r_rows = []
    matrix_reshaped = matrix.reshape(rows, cols)
    if highlight_diff is not None:
        diff_reshaped = highlight_diff.reshape(rows, cols)
    
    for r in range(rows):
        r_cells = []
        for c in range(cols):
            val = matrix_reshaped[r,c]
            is_diff = highlight_diff is not None and diff_reshaped[r,c] != val
            
            if is_diff:
                color = "#ef4444" if val == 1 else "rgba(239, 68, 68, 0.4)"
            else:
                color = active_color if val == 1 else "rgba(255,255,255,0.05)"
                
            r_cells.append(f"<div style='width:{cell_size}px; height:{cell_size}px; border:1px solid rgba(255,255,255,0.05); background-color:{color}; border-radius:3px;'></div>")
        r_rows.append(f"<div style='display:flex; justify-content:center; gap:2px;'>{''.join(r_cells)}</div>")
    
    title_html = f"<div style='text-align:center; font-weight:bold; margin-bottom:5px; color:#94a9c7;'>{title}</div>" if title else ""
    return f"<div>{title_html}<div style='display:flex; flex-direction:column; align-items:center; gap:2px;'>{''.join(r_rows)}</div></div>"

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def hopfield_page():
    st.markdown('<div class="nc-section-label">Memory Lab V4</div>', unsafe_allow_html=True)
    st.title("Hopfield Network (7x7 Precision Lab)")
    st.caption("Accurately recalls alphabet letters or numbers using 7x7 downsampling, asynchronous updates, and mean-centered Hebbian learning.")

    # Select Mode: Letters or Numbers
    mode = st.radio("Select Training Memory:", ["Letters (A-E)", "Numbers (0-9)"], horizontal=True)

    if mode == "Letters (A-E)":
        patterns_dict = get_7x7_patterns()
    else:
        patterns_dict = get_7x7_number_patterns()

    # Prepare Patterns & Train Network
    all_patterns = []
    for var_list in patterns_dict.values():
        all_patterns.extend(var_list)
        
    hn = HopfieldNetwork(49)
    hn.train(all_patterns)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("1. Drawing Board")
        stroke_width = st.slider("Brush Size", 5, 40, 20)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#0f172a",
            background_color="#f8fafc",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="hopfield_7x7_canvas",
            update_streamlit=True,
        )

        grid_12x12 = np.full((12, 12), -1)
        if canvas_result.image_data is not None:
            # We strictly analyze the Red channel to separate #0f172a (dark stroke) from #f8fafc (light bg)
            img_arr = canvas_result.image_data[:, :, 0]
            
            # Step 1: Strict Binarization BEFORE any resizing to trap thin pen lines
            # If red value < 100, it safely belongs to the dark stroke
            is_stroke = img_arr < 100 
            
            # Step 2: Convert to 0/255 and use BOX resampling to preserve true coverage area
            mask_img = Image.fromarray((is_stroke * 255).astype(np.uint8))
            mask_img = mask_img.resize((12, 12), resample=Image.Resampling.BOX)
            
            # Box resampling perfectly averages the pixels, so values represent % density.
            # > 15 out of 255 represents > 5% stroke area density. If yes, mark active (+1)!
            grid_12x12 = np.where(np.array(mask_img) > 15, 1, -1)

        st.markdown("**Original Drawing (Processed 12x12)**")
        st.markdown(render_grid_preview(grid_12x12, 12, 12, cell_size=16), unsafe_allow_html=True)

    with col2:
        st.subheader("2. Target Recall Process")
        
        grid_7x7 = downsample_12x12_to_7x7(grid_12x12)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(render_grid_preview(grid_7x7, 7, 7, cell_size=24, active_color="#fbbf24", title="Downsampled Input (7x7)"), unsafe_allow_html=True)
        
        if st.button("🧠 Recall Memory", type="primary", use_container_width=True):
            if np.all(grid_7x7 == -1):
                st.warning("Please draw a letter on the left to recall!")
            else:
                with st.spinner("Asynchronously Converging..."):
                    reconstructed = hn.predict(grid_7x7, num_iter=20)
                    
                    scores = []
                    for letter, variations in patterns_dict.items():
                        best_sim_for_letter = -1
                        best_var_pattern = None
                        for p in variations:
                            sim = cosine_similarity(reconstructed, np.array(p))
                            if sim > best_sim_for_letter:
                                best_sim_for_letter = sim
                                best_var_pattern = p
                        
                        confidence = (best_sim_for_letter + 1.0) / 2.0
                        scores.append((letter, confidence, best_var_pattern))
                    
                    scores.sort(key=lambda x: x[1], reverse=True)
                    best_match, best_score, best_pattern = scores[0]

                with c2:
                    st.markdown(render_grid_preview(reconstructed, 7, 7, cell_size=24, active_color="#34d399", title="Reconstructed (7x7)", highlight_diff=grid_7x7), unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="nc-panel" style="text-align: center; margin-top: 1.5rem; padding: 1.5rem;">
                        <div class="nc-section-label">Closest Match from Memory</div>
                        <h1 style="font-size: 5rem; margin: 0; color: #34d399;">{best_match}</h1>
                        <p style="font-weight: bold; color: #94a9c7;">Prediction Confidence: {best_score*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption("Top 3 Matches Ranked")
                for l, s, _ in scores[:3]:
                    st.progress(max(0.0, min(1.0, s)), text=f"{l} - {s*100:.1f}% Match")

        else:
            with c2:
                st.markdown(render_grid_preview(np.full(49,-1), 7, 7, cell_size=24, active_color="#34d399", title="Reconstructed (7x7)"), unsafe_allow_html=True)

    st.divider()
    with st.expander("Explore Memory Patterns (7x7)"):
        st.info("The Hopfield network is trained on these multi-variation 7x7 patterns using Hebbian Learning.")
        
        num_cols = 5
        for i, (l, variations) in enumerate(patterns_dict.items()):
            if i % num_cols == 0:
                p_cols = st.columns(num_cols)
                
            with p_cols[i % num_cols]:
                st.markdown(f"<div style='text-align:center; font-weight:bold; margin-bottom:10px;'>Pattern {l}</div>", unsafe_allow_html=True)
                for var_idx, p in enumerate(variations):
                    st.markdown(render_grid_preview(np.array(p), 7, 7, cell_size=12, active_color="#22c55e", title=f"Variation {var_idx+1}"), unsafe_allow_html=True)
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    hopfield_page()
