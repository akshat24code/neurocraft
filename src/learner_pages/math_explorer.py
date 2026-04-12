import streamlit as st
import numpy as np
import plotly.graph_objects as go

def plot_activation(fn_name):
    x = np.linspace(-10, 10, 200)
    if fn_name == 'Sigmoid':
        y = 1 / (1 + np.exp(-x))
    elif fn_name == 'ReLU':
        y = np.maximum(0, x)
    elif fn_name == 'Tanh':
        y = np.tanh(x)
    else:
        y = x

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=fn_name, line=dict(color="#67e8f9", width=3)))
    fig.update_layout(
        title=f"{fn_name} Function",
        xaxis_title="Input (z)",
        yaxis_title="Output (a)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig

def plot_loss(loss_type):
    if loss_type == 'MSE':
        y_true = 0
        y_hat = np.linspace(-2, 2, 100)
        loss = (y_true - y_hat) ** 2
        title = "MSE Loss (Target = 0)"
    else:
        y_true = 1
        y_hat = np.linspace(0.01, 0.99, 100)
        loss = - (y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
        title = "Binary Cross-Entropy Loss (Target = 1)"
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_hat, y=loss, mode='lines', name=loss_type, line=dict(color="#f59e0b", width=3)))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted (ŷ)",
        yaxis_title="Loss (L)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig

def plot_gradient_descent():
    w = np.linspace(-3, 3, 100)
    loss = w ** 2
    
    # Simulate points for gradient descent path
    w_steps = [-2.5, -1.8, -1.0, -0.4, -0.1, 0]
    loss_steps = [ww**2 for ww in w_steps]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=loss, mode='lines', name="Loss Landscape", line=dict(color="rgba(147, 197, 253, 0.5)", width=2)))
    fig.add_trace(go.Scatter(x=w_steps, y=loss_steps, mode='markers+lines', name="Weight Updates", 
                             marker=dict(color="#34d399", size=10), line=dict(color="#34d399", width=2, dash='dash')))
    fig.update_layout(
        title="Gradient Descent Optimization",
        xaxis_title="Weight (w)",
        yaxis_title="Loss (L)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig

def math_explorer_page():
    st.markdown('<div class="nc-section-label">Math Engine</div>', unsafe_allow_html=True)
    st.title("Math Explorer")
    st.caption("Deep dive into the fundamental mathematics powering modern neural networks.")

    tabs = st.tabs(["Linear Algebra", "Perceptron", "Activation Functions", "Loss Functions", "Gradient Descent", "Backpropagation", "Hopfield Network"])

    with tabs[0]:
        st.header("Linear Algebra (Vectors & Matrices)")
        st.markdown(
            "### Definition\n"
            "Linear algebra is the branch of mathematics concerning linear equations and their representations through matrices and vectors. "
            "In neural networks, the data and network parameters are stored as highly dimensional matrices (tensors)."
        )
        st.markdown(
            "### Step-by-Step Explanation\n"
            "Each neuron processes multiple inputs. Instead of looping through each input, neural networks compute them all at once using matrix multiplication.\n\n"
            "**Key Formula:**\n"
            r"$$ z = \sum_{i=1}^{n} w_i x_i + b $$"
        )
        st.info(
            "**Use-case in NNs:** Forward propagation relies heavily on matrix multiplication to map inputs from one layer to the next. "
            "Graphics Processing Units (GPUs) are specifically designed to perform these matrix operations rapidly in parallel!"
        )
        with st.expander("Interactive Demo: Matrix Multiplication"):
            st.markdown("Multiply a 1x2 weight matrix with a 2x1 input vector:")
            c1, c2 = st.columns(2)
            w1 = c1.number_input("Weight 1", value=0.5)
            w2 = c1.number_input("Weight 2", value=-0.8)
            x1 = c2.number_input("Input 1", value=1.0)
            x2 = c2.number_input("Input 2", value=2.0)
            b = st.number_input("Bias", value=0.1)
            z = (w1 * x1) + (w2 * x2) + b
            st.success(f"Output (z) = {w1}*{x1} + {w2}*{x2} + {b} = **{z:.4f}**")

    with tabs[1]:
        st.header("Perceptron (The Linear Unit)")
        st.markdown(
            "The Rosenblatt Perceptron represents the simplest learning machine. "
            "It learns a separating hyperplane in $N$-dimensional space."
        )
        st.latex(r"f(x) = \text{sgn}(\mathbf{w} \cdot \mathbf{x} + b)")
        st.info("**Convergence Theorem:** If the data is linearly separable, the Perceptron is guaranteed to find a solution in finite steps!")

    with tabs[1]:
        st.header("Activation Functions")
        st.markdown(
            "### Definition\n"
            "Activation functions introduce non-linearity to the network, enabling it to learn complex patterns. Without them, a neural network, no matter how deep, would just act as a single linear layer."
        )
        st.markdown(
            "### Step-by-Step Explanation\n"
            "We apply the activation function $f$ to the weighted sum $z$ from the linear step: $a = f(z)$\n\n"
            "**Key Formula (Sigmoid):**\n"
            r"$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$"
        )
        st.info("**Use-case in NNs:** ReLU is used in almost all hidden layers for faster and more stable learning (avoids vanishing gradients), while Sigmoid/Softmax is used in output layers to generate probabilities.")
        
        act_sel = st.selectbox("Visualize Function", ["Sigmoid", "ReLU", "Tanh"])
        st.plotly_chart(plot_activation(act_sel), use_container_width=True)

    with tabs[2]:
        st.header("Loss Functions")
        st.markdown(
            "### Definition\n"
            "A loss function evaluates how well the algorithm models your dataset. It outputs a higher number when predictions are bad and a lower number when they match the actual labels."
        )
        st.markdown(
            "### Step-by-Step Explanation\n"
            "We compare the prediction $\hat{y}$ to the true label $y$.\n\n"
            "**Key Formula (Mean Squared Error):**\n"
            r"$$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$"
        )
        st.info("**Use-case in NNs:** Mean Squared Error (MSE) is used for Regression. Cross-Entropy is used for Classification.")
        
        loss_sel = st.selectbox("Visualize Function", ["MSE", "Binary Cross-Entropy"])
        st.plotly_chart(plot_loss(loss_sel), use_container_width=True)
        
        with st.expander("3D Loss Surface Concept"):
            st.markdown("In higher dimensions, the loss landscape looks like a terrain with mountains and valleys.")
            # Simple 3D mesh
            x = np.linspace(-2, 2, 30)
            y = np.linspace(-2, 2, 30)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2
            fig3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
            fig3d.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig3d)

    with tabs[3]:
        st.header("Gradient Descent")
        st.markdown(
            "### Definition\n"
            "Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent (the negative gradient)."
        )
        st.markdown(
            "### Step-by-Step Explanation\n"
            "We compute the gradient $\\frac{\\partial L}{\\partial w}$, which points in the direction of steepest ascent. We subtract a small fraction of this gradient from our weights to descend towards the minimum error.\n\n"
            "**Key Formula:**\n"
            r"$$ w_{new} = w_{old} - \eta \frac{\partial L}{\partial w_{old}} $$"
            "Where $\eta$ is the **learning rate**."
        )
        st.info("**Use-case in NNs:** This is the core engine of \"learning.\" Every time you train an epoch, the optimizer (like SGD or Adam) executes this formula for millions of parameters.")
        
        st.plotly_chart(plot_gradient_descent(), use_container_width=True)

    with tabs[4]:
        st.header("Backpropagation (Chain Rule)")
        st.markdown(
            "### Definition\n"
            "Backpropagation calculates the gradient of the loss function with respect to the weights of the network. It uses the mathematical *Chain Rule* to propagate the error backwards from output to input."
        )
        st.markdown(
            "### Step-by-Step Explanation\n"
            "If $L$ depends on $a$, $a$ depends on $z$, and $z$ depends on $w$, we multiply the partial derivatives together.\n\n"
            "**Key Formula:**\n"
            r"$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} $$"
        )
        st.info("**Use-case in NNs:** Allows us to calculate exactly how much each neuron in the early layers contributed to the final mistake, so we can adjust its weights appropriately.")

        st.markdown("**(Interactive Backprop Example)**")
        st.latex(r"y = 1 \quad \text{(Target)}")
        st.latex(r"w = 0.5, \quad x = 1.0, \quad b = 0")
        st.latex(r"z = w \cdot x + b = 0.5")
        st.latex(r"a = \sigma(0.5) = 0.6225")
        st.markdown(r"**Update Step:** Assuming $\frac{\partial L}{\partial w} = -0.09$, if $\eta = 1.0$, $w_{new} = 0.5 - 1.0 \times (-0.09) = 0.59$. The weight increases to bump the output closer to 1!")

    with tabs[6]:
        st.header("Hopfield Networks (Associative Memory)")
        st.markdown(
            "Hopfield networks use an **Energy Function** (Lyapunov function) to move towards stable patterns."
        )
        st.latex(r"E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j + \sum_i \theta_i s_i")
        st.info("**Hebb's Rule:** 'Neurons that fire together, wire together'. This simple idea allows the network to store and recall memories based on incomplete cues.")

if __name__ == "__main__":
    math_explorer_page()
