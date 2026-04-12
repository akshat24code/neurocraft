import streamlit as st

def rnn_page():
    st.markdown('<div class="nc-section-label">Sequence Models</div>', unsafe_allow_html=True)
    st.title("Recurrent Neural Networks (RNN)")
    st.caption("Visualize how RNNs process sequential data, maintain hidden states, and handle time-series information.")

    tab_theory, tab_math, tab_experiment, tab_analysis = st.tabs([
        "Theory", "Math", "Experiment", "Analysis"
    ])

    with tab_theory:
        st.subheader("What is an RNN?")
        st.markdown(
            "Recurrent Neural Networks (RNNs) are designed specifically to handle sequential data like text, speech, or time-series. "
            "Unlike traditional feed-forward networks, RNNs maintain a hidden state (or memory) that captures information from previous timesteps.\n\n"
            "Challenges with standard RNNs include the **vanishing gradient problem**, which makes it difficult to learn long-term dependencies, "
            "leading to the development of LSTMs and GRUs."
        )

    with tab_math:
        st.subheader("RNN Forward Propagation")
        st.markdown(
            r"""
            At each timestep $t$, the RNN takes the current input $x_t$ and the previous hidden state $h_{t-1}$ to compute the new hidden state $h_t$:

            $$ h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h) $$
            
            The output $y_t$ at timestep $t$ can be computed as:
            
            $$ y_t = W_{yh} h_t + b_y $$
            
            Where:
            - $W_{hx}$: Weights for input
            - $W_{hh}$: Weights for recurrent hidden state
            - $W_{yh}$: Weights for output
            """
        )

    with tab_experiment:
        st.info("Interactive RNN visualization will be integrated here.")
        st.markdown("**Sequence Input:** Enter a sequence of data points or text.")
        st.markdown("**Unroll Network:** Visualize the network unrolled across timesteps.")
        st.button("Simulate Sequence Processing", disabled=True)

    with tab_analysis:
        st.info("Analyze hidden states and predictions across timesteps.")
        st.markdown("**Hidden State Evolution:** See how the hidden state changes as new data is processed.")

if __name__ == "__main__":
    rnn_page()
