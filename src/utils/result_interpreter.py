import streamlit as st
import numpy as np

def interpret_results(module_name, metrics):
    """
    Provides intelligent suggestions based on training results.
    metrics: dict containing relevant values like 'accuracy', 'loss', 'converged', etc.
    """
    st.markdown("### 🧠 Intelligent Analysis")
    
    advice = []
    
    if module_name == "Perceptron":
        acc = metrics.get('accuracy', 0)
        converged = metrics.get('converged', False)
        
        if converged:
            advice.append("✅ **Perfect Linear Separation:** Your data is linearly separable. The perceptron found a boundary that perfectly divides the classes.")
        elif acc > 80:
            advice.append("⚠️ **High Accuracy but No Convergence:** The classes might be slightly overlapping or the learning rate is too high, causing the boundary to 'jump' over the solution.")
            advice.append("💡 **Try:** Reducing the learning rate or increasing epochs.")
        else:
            advice.append("❌ **Poor Separation:** This dataset likely has non-linear patterns (like XOR). A single perceptron cannot solve this.")
            advice.append("💡 **Suggestion:** Move to the **Multi-Layer Perceptron (MLP)** module for non-linear boundaries.")

    elif module_name == "MLP":
        loss_history = metrics.get('loss_history', [])
        final_acc = metrics.get('accuracy', 0)
        
        if len(loss_history) > 1:
            if loss_history[-1] > loss_history[0]:
                advice.append("🛑 **Divergence Detected:** Loss is increasing! This usually happens with a learning rate that is too high.")
                advice.append("💡 **Try:** Lowering the learning rate by a factor of 10.")
            elif loss_history[-1] == loss_history[-2] and final_acc < 90:
                advice.append("📉 **Plateau reached:** The model has stopped learning. It might be stuck in a local minimum.")
                advice.append("💡 **Try:** Adding more hidden neurons or using a different activation function (like ReLU).")

    elif module_name == "Hopfield":
        energy_history = metrics.get('energy_history', [])
        if energy_history and energy_history[-1] > energy_history[0]:
            advice.append("⚠️ **Unstable Dynamics:** Energy should typically decrease. Check your weight matrix symmetry.")
        else:
            advice.append("✅ **Stable Attraction:** The network converged to a local energy minimum (an 'attractor').")

    # Display the advice in a nice box
    for item in advice:
        st.write(item)
    
    if not advice:
        st.write("Keep experimenting! Adjust hyperparameters to see how they impact the learning dynamics.")
