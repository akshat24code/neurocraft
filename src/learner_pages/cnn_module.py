import streamlit as st

def cnn_page():
    st.markdown('<div class="nc-section-label">Vision Lab</div>', unsafe_allow_html=True)
    st.title("Convolutional Neural Networks (CNN)")
    st.caption("Understand how filters, convolutions, and pooling layers extract visual features for image classification.")

    tab_theory, tab_math, tab_experiment, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "Analysis"
    ])

    with tab_theory:
        st.subheader("What is a CNN?")
        st.markdown(
            "Convolutional Neural Networks (CNNs) are specialized neural networks for processing data that has "
            "a known, grid-like topology, such as images (2D grid of pixels).\n\n"
            "Key components include:\n"
            "- **Convolutional Layers:** Apply filters to extract spatial features (edges, textures).\n"
            "- **Pooling Layers:** Downsample spatial dimensions to reduce computation and add translational invariance.\n"
            "- **Fully Connected Layers:** At the end of the network, they interpret the extracted features to classify the image."
        )

    with tab_math:
        st.subheader("The Mathematics of Convolution")
        st.markdown(
            r"""
            **1. Discrete 2D Convolution**
            For an image $I$ and a filter (kernel) $K$ of size $m \times n$, the convolution operation is defined as:
            $$ S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n) $$
            
            **2. Activation (ReLU)**
            $$ f(x) = \max(0, x) $$
            
            **3. Max Pooling**
            Downsamples the feature map by taking the maximum value over a spatial window (e.g., $2 \times 2$):
            $$ P(i, j) = \max_{(m,n) \in \text{window}} S(i+m, j+n) $$
            """
        )

    with tab_experiment:
        st.info("Interactive CNN experiment and visualization will be integrated here.")
        # Placeholders for future CNN visualizer
        st.markdown("**Load Image:** Upload or select a sample image.")
        st.markdown("**Network Architecture:** Select number of Conv layers, kernel size, and pooling type.")
        st.button("Run Forward Pass", disabled=True)

    with tab_analysis:
        st.info("Analyze the feature maps and class predictions here.")
        # Placeholders for analysis
        st.markdown("**Feature Maps:** View how the image is transformed at each layer.")

if __name__ == "__main__":
    cnn_page()
