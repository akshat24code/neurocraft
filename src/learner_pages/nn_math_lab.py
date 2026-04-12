import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def linear_algebra_lab(explain_mode=True):
    st.markdown("### Linear Algebra Lab: Vector Transformations")
    if explain_mode:
        with st.expander("📚 Concept: Linear Transformations", expanded=True):
            st.write("""
            In Neural Networks, weight matrices $W$ act as linear transformations. 
            Multiplying a vector $x$ by $W$ can rotate, scale, or shear the vector in space.
            """)
            st.latex(r"y = Wx = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Vector Input")
        x1 = st.slider("x1", -5.0, 5.0, 1.0)
        x2 = st.slider("x2", -5.0, 5.0, 1.0)
        x = np.array([x1, x2])
        
        st.subheader("Matrix W")
        r1c1 = st.number_input("W[0,0]", value=1.0)
        r1c2 = st.number_input("W[0,1]", value=0.0)
        r2c1 = st.number_input("W[1,0]", value=0.0)
        r2c2 = st.number_input("W[1,1]", value=1.0)
        W = np.array([[r1c1, r1c2], [r2c1, r2c2]])

    with col2:
        v_transformed = W @ x
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector x')
        ax.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='red', label='Transformed Vector Wx')
        
        limit = max(np.abs(x).max(), np.abs(v_transformed).max()) + 1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        st.latex(f"Wx = {v_transformed}")

def calculus_lab(explain_mode=True):
    st.markdown("### Calculus Lab: Derivatives & Tangents")
    if explain_mode:
        with st.expander("📚 Concept: The Derivative", expanded=True):
            st.write("""
            The derivative $\frac{df}{dx}$ represents the instantaneous rate of change or the slope of the tangent line at a point.
            In AI, we use derivatives (gradients) to understand how to change weights to reduce error.
            """)

    col1, col2 = st.columns([1, 2])
    with col1:
        func_name = st.selectbox("Select Function f(x)", ["x^2", "sin(x)", "x^3"])
        point = st.slider("Select Point x", -5.0, 5.0, 0.0)
        
        if func_name == "x^2":
            f = lambda x: x**2
            df = lambda x: 2*x
        elif func_name == "sin(x)":
            f = lambda x: np.sin(x)
            df = lambda x: np.cos(x)
        else:
            f = lambda x: x**3
            df = lambda x: 3*x**2

    with col2:
        x_range = np.linspace(-6, 6, 200)
        y_range = f(x_range)
        slope = df(point)
        tangent = slope * (x_range - point) + f(point)
        
        fig, ax = plt.subplots()
        ax.plot(x_range, y_range, label=f"f(x) = {func_name}", color='black')
        ax.plot(x_range, tangent, '--', label="Tangent Line", color='red')
        ax.scatter([point], [f(point)], color='red', s=100)
        ax.set_ylim(min(y_range)-2, max(y_range)+2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        st.metric("Slope (f'(x))", f"{slope:.4f}")

def probability_lab(explain_mode=True):
    st.markdown("### Probability Lab: Normal Distribution")
    if explain_mode:
        with st.expander("📚 Concept: Normal Distribution", expanded=True):
            st.write("""
            The Gaussian distribution is fundamental in ML. $\mu$ is the mean, and $\sigma$ is the standard deviation.
            """)
            st.latex(r"P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}")

    col1, col2 = st.columns([1, 2])
    with col1:
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0)
        sigma = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0)
    with col2:
        x = np.linspace(-10, 10, 500)
        p = stats.norm.pdf(x, mu, sigma)
        fig, ax = plt.subplots()
        ax.plot(x, p, 'b-', lw=3)
        ax.fill_between(x, p, alpha=0.2, color='blue')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def optimization_lab(explain_mode=True):
    st.markdown("### Optimization Lab: Gradient Descent Simulator")
    if explain_mode:
        with st.expander("📚 Concept: Gradient Descent", expanded=True):
            st.write("""
            We move weights in the opposite direction of the gradient to find the minimum of the loss function.
            """)
            st.latex(r"x_{new} = x_{old} - \eta \cdot \nabla f(x)")

    col1, col2 = st.columns([1, 2])
    with col1:
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
        start_x = st.slider("Initial Value", -10.0, 10.0, 8.0)
        steps = st.slider("Steps", 1, 50, 10)
        f = lambda x: x**2
        df = lambda x: 2*x
    with col2:
        history_x = [start_x]
        curr_x = start_x
        for _ in range(steps):
            curr_x = curr_x - lr * df(curr_x)
            history_x.append(curr_x)
        x_range = np.linspace(-11, 11, 400)
        y_range = f(x_range)
        history_y = [f(px) for px in history_x]
        fig, ax = plt.subplots()
        ax.plot(x_range, y_range, 'k-', alpha=0.3)
        ax.scatter(history_x, history_y, color='red', s=40)
        ax.plot(history_x, history_y, 'r--', lw=1, alpha=0.6)
        st.pyplot(fig)
        st.write(f"Final Value after {steps} steps: **{curr_x:.6f}**")
