# NeuroCraft - Hopfield Memory Lab

A modern web application demonstrating **Associative Memory** using a **Hopfield Neural Network**.

---

## 🎨 Features
- **12x12 Interactive Grid:** Sketch letters A, B, C, D, or E smoothly using mouse drag.
- **Predictive Recognition:** The Hopfield network reconstructs the stored pattern from your noisy or incomplete drawing.
- **Modern UI:** Built with React, Tailwind CSS, and Framer Motion for premium aesthetics.
- **Confidence Scoring:** Understand how closely your sketch matches the network's memories.
- **Reconstruction Preview:** Visualize the final stable state (convergence) of the neural network.

---

## 🛠️ Tech Stack
- **Frontend:** React.js (Vite), Tailwind CSS, Framer Motion, Lucide icons, Axios.
- **Backend:** Python Flask, NumPy (Hopfield Network logic).

---

## 🚀 How to Run

### 1. Start the Backend
Navigate to the `backend` folder and follow these steps:
```bash
cd neurocraft-hopfield/backend
pip install -r requirements.txt
python app.py
```
The server will start at `http://localhost:5000`.

### 2. Start the Frontend
Navigate to the `frontend` folder and follow these steps:
```bash
cd neurocraft-hopfield/frontend
npm install
npm run dev
```
The application will be available at `http://localhost:5173`.

---

## 🔬 How it Works
The Hopfield Network uses **Hebbian Learning** to store patterns (A, B, C, D, E). When you provide an input, the network iteratively updates its state to minimize its "Energy" until it finds a stable state corresponding to one of the stored memories.

- **Black Cell:** represented as `+1`
- **White Cell:** represented as `-1`
- **Weight Matrix:** calculated via outer products of trained patterns.
- **Convergence:** the network converges to the nearest local energy minimum.
