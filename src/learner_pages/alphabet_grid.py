import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


DRAW_ROWS = 14
DRAW_COLS = 10
TEMPLATE_ROWS = 7
TEMPLATE_COLS = 5
CANVAS_HEIGHT = 420
CANVAS_WIDTH = 300
TEMPLATE_INDEX = [f"R{i + 1}" for i in range(TEMPLATE_ROWS)]
TEMPLATE_COLUMNS = [f"C{j + 1}" for j in range(TEMPLATE_COLS)]
DRAW_INDEX = [f"R{i + 1}" for i in range(DRAW_ROWS)]
DRAW_COLUMNS = [f"C{j + 1}" for j in range(DRAW_COLS)]

LETTER_PATTERNS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
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


def _pattern_to_matrix(pattern_rows: list[str]) -> np.ndarray:
    return np.array([[int(cell) for cell in row] for row in pattern_rows], dtype=int)


def _matrix_frame(matrix: np.ndarray, index_labels: list[str], column_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix.astype(int), index=index_labels, columns=column_labels)


def _template_vectors() -> dict[str, np.ndarray]:
    return {letter: _pattern_to_matrix(pattern).flatten() for letter, pattern in LETTER_PATTERNS.items()}


def _resize_to_grid(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    resized = image.resize((cols, rows), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=float) / 255.0
    return (arr >= 0.20).astype(int)


def _compress_drawn_grid(draw_grid: np.ndarray) -> np.ndarray:
    compressed = np.zeros((TEMPLATE_ROWS, TEMPLATE_COLS), dtype=int)
    for row in range(TEMPLATE_ROWS):
        for col in range(TEMPLATE_COLS):
            block = draw_grid[row * 2:(row + 1) * 2, col * 2:(col + 1) * 2]
            compressed[row, col] = 1 if np.mean(block) >= 0.25 else 0
    return compressed


def _recognize_letter(grid: np.ndarray) -> tuple[str | None, pd.DataFrame]:
    vector = grid.flatten()
    if int(vector.sum()) == 0:
        return None, pd.DataFrame(columns=["Letter", "Match %", "Pixel Differences"])

    scores = []
    for letter, template in _template_vectors().items():
        matches = int(np.sum(vector == template))
        differences = int(np.sum(np.abs(vector - template)))
        match_pct = 100.0 * matches / len(template)
        scores.append((letter, match_pct, differences))

    results = pd.DataFrame(scores, columns=["Letter", "Match %", "Pixel Differences"])
    results = results.sort_values(["Match %", "Pixel Differences", "Letter"], ascending=[False, True, True]).reset_index(drop=True)
    return str(results.iloc[0]["Letter"]), results


def _render_pixel_preview(matrix: np.ndarray, cell_size: int = 28) -> None:
    rows = []
    for row in matrix:
        cells = []
        for value in row:
            bg = "#22c55e" if int(value) else "#0f172a"
            border = "#86efac" if int(value) else "#334155"
            cells.append(
                f"<div style='width:{cell_size}px;height:{cell_size}px;border-radius:6px;background:{bg};border:1px solid {border};'></div>"
            )
        rows.append(f"<div style='display:flex;gap:4px;margin-bottom:4px;'>{''.join(cells)}</div>")

    st.markdown(
        "<div style='padding:14px;border-radius:18px;border:1px solid rgba(148,163,184,0.25);"
        "background:linear-gradient(180deg, rgba(15,23,42,0.9), rgba(17,24,39,0.96));display:inline-block;'>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def _canvas_to_draw_grid(image_data) -> np.ndarray:
    if image_data is None:
        return np.zeros((DRAW_ROWS, DRAW_COLS), dtype=int)

    rgba = np.asarray(image_data)
    alpha = rgba[:, :, 3] > 0
    rgb_dark = np.mean(rgba[:, :, :3], axis=2) < 250
    mask = np.logical_or(alpha, rgb_dark).astype(np.uint8)
    return _resize_to_grid(mask, DRAW_ROWS, DRAW_COLS)


def alphabet_grid_page() -> None:
    st.title("Alphabet Grid Recognition")
    st.caption(
        "Draw an alphabet letter with your mouse on the canvas. The app converts your drawing into a grid "
        "and recognizes the closest alphabet pattern."
    )

    st.divider()
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Canvas Workspace")
        st.caption("Draw using your mouse. Use the clear button below the canvas whenever you want to start over.")

        stroke_width = st.slider("Brush size", 8, 28, 16)
        canvas_result = st_canvas(
            fill_color="rgba(34, 197, 94, 0.95)",
            stroke_width=stroke_width,
            stroke_color="#22c55e",
            background_color="#0f172a",
            update_streamlit=True,
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="alphabet_canvas",
        )

        draw_grid = _canvas_to_draw_grid(canvas_result.image_data)

        st.markdown("**Detected Drawing Grid (14 x 10)**")
        _render_pixel_preview(draw_grid, cell_size=18)
        st.dataframe(_matrix_frame(draw_grid, DRAW_INDEX, DRAW_COLUMNS), use_container_width=True)

    compact_grid = _compress_drawn_grid(draw_grid)
    recognized_letter, ranking = _recognize_letter(compact_grid)

    with right:
        st.subheader("Recognition Result")
        st.metric("Filled Cells", int(draw_grid.sum()))

        if recognized_letter is None:
            st.info("Draw a letter on the canvas and the recognizer will predict it here.")
        else:
            top_match = ranking.iloc[0]
            st.metric("Predicted Letter", recognized_letter)
            st.metric("Match Score", f"{top_match['Match %']:.1f}%")
            st.metric("Pixel Differences", str(int(top_match["Pixel Differences"])))

            st.markdown("**Top 5 Matches**")
            st.dataframe(ranking.head(5), use_container_width=True, hide_index=True)

            st.markdown("**Recognition Grid (7 x 5)**")
            _render_pixel_preview(compact_grid, cell_size=28)
            st.dataframe(_matrix_frame(compact_grid, TEMPLATE_INDEX, TEMPLATE_COLUMNS), use_container_width=True)

            st.markdown("**Template For Predicted Letter**")
            template_matrix = _pattern_to_matrix(LETTER_PATTERNS[recognized_letter])
            _render_pixel_preview(template_matrix, cell_size=28)
            st.dataframe(_matrix_frame(template_matrix, TEMPLATE_INDEX, TEMPLATE_COLUMNS), use_container_width=True)

    st.divider()
    st.subheader("How Recognition Works")
    st.markdown(
        "1. You draw freely on the canvas with the mouse.\n"
        "2. The canvas is converted into a 14 x 10 binary drawing grid.\n"
        "3. That drawing is compressed into a 7 x 5 recognition matrix.\n"
        "4. The recognition matrix is compared with stored A-Z templates.\n"
        "5. The closest matching template is returned as the recognized letter."
    )
