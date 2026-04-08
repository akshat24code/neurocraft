import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { 
  Eraser, 
  Pencil, 
  RotateCcw, 
  Cpu, 
  Sparkles, 
  Dice5,
  BrainCircuit,
  Target,
  ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const GRID_SIZE = 12;
const API_URL = 'http://localhost:5000/predict';

const App = () => {
  const [grid, setGrid] = useState(Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0)));
  const [isDrawing, setIsDrawing] = useState(false);
  const [eraseMode, setEraseMode] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [reconstructed, setReconstructed] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);

  // Mouse handling
  const handleMouseDown = (row, col) => {
    setIsDrawing(true);
    updatePixel(row, col);
  };

  const handleMouseEnter = (row, col) => {
    if (isDrawing) {
      updatePixel(row, col);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const updatePixel = (row, col) => {
    const newGrid = [...grid];
    newGrid[row][col] = eraseMode ? 0 : 1;
    setGrid(newGrid);
  };

  const clearGrid = () => {
    setGrid(Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0)));
    setPrediction(null);
    setReconstructed(null);
    setConfidence(null);
  };

  const addNoise = () => {
    const newGrid = grid.map(row => 
      row.map(cell => (Math.random() < 0.1 ? (cell === 1 ? 0 : 1) : cell))
    );
    setGrid(newGrid);
  };

  const predictLetter = async () => {
    setLoading(true);
    try {
      const response = await axios.post(API_URL, { matrix: grid });
      setPrediction(response.data.letter);
      setReconstructed(response.data.reconstructed);
      setConfidence(response.data.similarity);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Backend not found! Make sure the Python Flask server is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pb-20">
      {/* Navbar/Header */}
      <nav className="glass sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-neuro-600 p-2 rounded-xl text-white">
            <BrainCircuit size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">NeuroCraft</h1>
            <p className="text-xs font-semibold text-neuro-600 uppercase tracking-widest">Hopfield Memory Lab</p>
          </div>
        </div>
        <div className="hidden md:flex gap-6 text-sm font-medium text-slate-500">
           <a href="#about" className="hover:text-neuro-600 transition-colors">How it works</a>
           <a href="#github" className="hover:text-neuro-600 transition-colors">Source</a>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 mt-12 grid grid-cols-1 lg:grid-cols-12 gap-10">
        
        {/* Control Panel - Left Side (Col: 3) */}
        <div className="lg:col-span-3 space-y-6">
          <div className="glass rounded-2xl p-6 space-y-6">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Cpu size={20} className="text-neuro-600" />
              Controls
            </h3>
            
            <div className="grid gap-3">
              <button 
                onClick={() => setEraseMode(!eraseMode)}
                className={`flex items-center justify-center gap-3 p-3 rounded-xl transition-all border ${
                  eraseMode 
                  ? 'bg-rose-50 border-rose-200 text-rose-600 shadow-sm' 
                  : 'bg-white border-slate-200 text-slate-600 hover:border-neuro-300'
                }`}
              >
                {eraseMode ? <Eraser size={20} /> : <Pencil size={20} />}
                <span className="font-semibold">{eraseMode ? "Eraser Active" : "Brush Active"}</span>
              </button>

              <button 
                onClick={addNoise}
                className="flex items-center justify-center gap-3 p-3 rounded-xl bg-white border border-slate-200 text-slate-600 hover:border-neuro-300 transition-all active:bg-slate-50"
              >
                <Dice5 size={20} />
                <span className="font-semibold">Add Noise</span>
              </button>

              <button 
                onClick={clearGrid}
                className="flex items-center justify-center gap-3 p-3 rounded-xl bg-rose-50 border border-rose-100 text-rose-600 hover:bg-rose-100 transition-all"
              >
                <RotateCcw size={20} />
                <span className="font-semibold">Clear Grid</span>
              </button>
            </div>

            <div className="pt-4 border-t border-slate-100">
              <button 
                onClick={predictLetter}
                disabled={loading}
                className="w-full flex items-center justify-center gap-3 p-4 rounded-xl bg-neuro-600 text-white font-bold shadow-lg shadow-neuro-200 hover:bg-neuro-700 transition-all disabled:opacity-50"
              >
                {loading ? <div className="w-5 h-5 border-2 border-white/50 border-t-white rounded-full animate-spin" /> : <Sparkles size={20} />}
                {loading ? "Recognizing..." : "Predict Letter"}
              </button>
            </div>
          </div>

          {/* Pattern Stats/Info */}
          <div className="glass rounded-2xl p-6 bg-gradient-to-br from-neuro-600 to-neuro-800 text-white shadow-xl shadow-neuro-200">
            <h4 className="text-white/80 text-xs font-bold uppercase tracking-wider mb-2">Stored Patterns</h4>
            <div className="text-3xl font-black tracking-tighter flex gap-2">
              {['A', 'B', 'C', 'D', 'E'].map(l => <span key={l} className="opacity-50 hover:opacity-100 cursor-default">{l}</span>)}
            </div>
            <p className="mt-4 text-xs text-white/70 leading-relaxed">
              Associative memory allows the network to recall a stored pattern even if the input is distorted or noisy.
            </p>
          </div>
        </div>

        {/* Drawing Board - Center (Col: 6) */}
        <div className="lg:col-span-6 flex flex-col items-center">
          <div 
            className="bg-white p-3 rounded-[2rem] shadow-2xl border-8 border-slate-200/50 touch-none select-none"
            onMouseLeave={handleMouseUp}
          >
            <div 
              className="grid grid-cols-12 gap-[1px] bg-slate-100 rounded-lg overflow-hidden"
              style={{ width: 'min(90vw, 480px)', height: 'min(90vw, 480px)' }}
            >
              {grid.map((row, rowIndex) => 
                row.map((cell, colIndex) => (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                    onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                    onMouseUp={handleMouseUp}
                    className={`pixel transition-none ${cell === 1 ? 'pixel-black' : 'pixel-white hover:bg-slate-100'}`}
                  />
                ))
              )}
            </div>
          </div>
          <p className="mt-6 text-sm text-slate-400 font-medium flex items-center gap-2 italic">
            <MousePointer2 size={14} />
            Click and drag to sketch a letter
          </p>
        </div>

        {/* Results - Right Side (Col: 3) */}
        <div className="lg:col-span-3 space-y-6">
          <AnimatePresence mode="wait">
            {prediction ? (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Prediction Result */}
                <div className="glass rounded-2xl p-6 text-center">
                  <div className="flex justify-center mb-4">
                    <div className="w-16 h-16 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center animate-bounce">
                      <Target size={32} />
                    </div>
                  </div>
                  <h4 className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-1">Top Prediction</h4>
                  <div className="text-8xl font-black text-slate-900 leading-none mb-4">
                    {prediction}
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden mb-2">
                    <motion.div 
                      className="bg-emerald-500 h-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence * 100}%` }}
                    />
                  </div>
                  <p className="text-sm font-bold text-slate-600">
                    Confidence: <span className={confidence > 0.8 ? 'text-emerald-600' : 'text-amber-600'}>
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </p>
                </div>

                {/* Reconstructed Preview */}
                {reconstructed && (
                  <div className="glass rounded-2xl p-6">
                    <h4 className="text-slate-500 text-xs font-bold uppercase tracking-wider mb-4 flex items-center gap-2">
                      <RotateCcw size={14} />
                      Convergence Result
                    </h4>
                    <div className="grid grid-cols-12 gap-[1px] bg-slate-100 rounded-md overflow-hidden bg-white border border-slate-200">
                      {reconstructed.map((row) => 
                        row.map((cell, i) => (
                          <div
                            key={i}
                            className={`w-full aspect-square ${cell === 1 ? 'bg-slate-800' : 'bg-white'}`}
                          />
                        ))
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            ) : (
              <div className="glass rounded-2xl p-8 text-center bg-slate-50/50 border-dashed border-2 border-slate-200">
                 <div className="w-16 h-16 mx-auto bg-slate-200/50 text-slate-400 rounded-full flex items-center justify-center mb-4">
                  <Cpu size={24} />
                </div>
                <h4 className="text-slate-800 font-bold mb-2">Awaiting Input</h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Draw a letter (A, B, C, D, or E) and click Predict to see associative memory in action.
                </p>
              </div>
            )}
          </AnimatePresence>
        </div>

      </main>

      {/* Footer / About */}
      <section className="max-w-4xl mx-auto px-6 mt-20" id="about">
        <div className="p-8 rounded-[2rem] bg-slate-900 text-white relative overflow-hidden">
          <div className="relative z-10">
            <h2 className="text-3xl font-bold mb-4">The Hopfield Network</h2>
            <div className="grid md:grid-cols-2 gap-8 text-slate-300 text-sm leading-relaxed">
              <div>
                <p className="mb-4">
                  A Hopfield network is a form of recurrent artificial neural network that serves as a content-addressable memory system.
                </p>
                <p>
                  It doesn't "recognize" features like a CNN. Instead, it recovers stored patterns using "Energy Minimization".
                </p>
              </div>
              <ul className="space-y-3">
                <li className="flex gap-2">
                  <div className="mt-1 flex-shrink-0 w-4 h-4 rounded-full bg-neuro-500" />
                  <span><strong>Hebbian Learning:</strong> Neurons that fire together, wire together.</span>
                </li>
                <li className="flex gap-2">
                  <div className="mt-1 flex-shrink-0 w-4 h-4 rounded-full bg-neuro-500" />
                  <span><strong>Associative Memory:</strong> Recall even from incomplete data.</span>
                </li>
              </ul>
            </div>
          </div>
          {/* Abstract pattern decoration */}
          <div className="absolute -right-20 -bottom-20 w-80 h-80 bg-neuro-600/20 rounded-full blur-3xl" />
        </div>
      </section>
    </div>
  );
};

// Added missing Icon for clarity
const MousePointer2 = ({ size, className }) => (
  <svg 
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
  >
    <path d="m22 2-7 20-4-9-9-4Z" />
    <path d="M6 6l.01 0" />
  </svg>
);

export default App;
