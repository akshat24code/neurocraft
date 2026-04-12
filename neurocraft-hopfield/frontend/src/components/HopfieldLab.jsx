import React, { useState } from 'react';
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
  MousePointer2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const GRID_SIZE = 12;
const API_URL = 'http://localhost:5000/predict';

const HopfieldLab = () => {
  const [grid, setGrid] = useState(Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0)));
  const [isDrawing, setIsDrawing] = useState(false);
  const [eraseMode, setEraseMode] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [reconstructed, setReconstructed] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);

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
    <div className="flex-1 min-h-screen pl-64 bg-cyber-bg transition-all duration-500 p-10">
       <div className="max-w-6xl mx-auto space-y-10">
          <div className="flex items-center gap-4">
            <div className="bg-cyber-cyan/20 p-3 rounded-2xl text-cyber-cyan border border-cyber-cyan/30">
              <BrainCircuit size={32} className="drop-shadow-[0_0_8px_rgba(0,242,255,0.5)]" />
            </div>
            <div>
              <h1 className="text-4xl font-black tracking-tight text-white">Hopfield Memory Lab</h1>
              <p className="text-sm font-bold text-cyber-cyan uppercase tracking-[0.2em] mt-1">Associative Neural Network</p>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-10">
             {/* Left Column: Controls */}
             <div className="col-span-3 space-y-6">
                <div className="glass-card rounded-2xl p-6 space-y-6">
                  <h3 className="text-xs font-black uppercase tracking-widest text-slate-400 flex items-center gap-2">
                    <Cpu size={16} className="text-cyber-cyan" />
                    Lab Controls
                  </h3>
                  
                  <div className="grid gap-4">
                    <button 
                      onClick={() => setEraseMode(!eraseMode)}
                      className={`flex items-center justify-center gap-3 py-4 rounded-xl transition-all border font-black uppercase text-[10px] tracking-widest ${
                        eraseMode 
                        ? 'bg-rose-500/10 border-rose-500/50 text-rose-500' 
                        : 'bg-white/5 border-white/10 text-white hover:border-cyber-cyan/50 hover:bg-white/10'
                      }`}
                    >
                      {eraseMode ? <Eraser size={18} /> : <Pencil size={18} />}
                      <span>{eraseMode ? "Eraser Mode" : "Brush Mode"}</span>
                    </button>

                    <button 
                      onClick={addNoise}
                      className="flex items-center justify-center gap-3 py-4 rounded-xl bg-white/5 border border-white/10 text-white font-black uppercase text-[10px] tracking-widest hover:border-cyber-cyan/50 hover:bg-white/10 transition-all"
                    >
                      <Dice5 size={18} />
                      <span>Inject Noise</span>
                    </button>

                    <button 
                      onClick={clearGrid}
                      className="flex items-center justify-center gap-3 py-4 rounded-xl bg-white/5 border border-white/10 text-slate-500 font-black uppercase text-[10px] tracking-widest hover:text-rose-400 transition-all"
                    >
                      <RotateCcw size={18} />
                      <span>Reset Canvas</span>
                    </button>
                  </div>

                  <div className="pt-4 border-t border-white/5">
                    <button 
                      onClick={predictLetter}
                      disabled={loading}
                      className="w-full flex items-center justify-center gap-3 py-4 rounded-xl bg-cyber-cyan text-black font-black uppercase text-[10px] tracking-widest hover:shadow-[0_0_20px_rgba(0,242,255,0.5)] transition-all disabled:opacity-50"
                    >
                      {loading ? <div className="w-4 h-4 border-2 border-black/50 border-t-black rounded-full animate-spin" /> : <Sparkles size={18} />}
                      {loading ? "Computing..." : "Recover Pattern"}
                    </button>
                  </div>
                </div>
             </div>

             {/* Middle Column: Canvas */}
             <div className="col-span-6 flex flex-col items-center">
                <div 
                  className="bg-black/50 p-4 rounded-[2.5rem] shadow-2xl border border-white/10 backdrop-blur-xl relative"
                  onMouseLeave={handleMouseUp}
                >
                  <div className="absolute inset-0 bg-cyber-cyan/5 blur-[100px] rounded-full pointer-events-none"></div>
                  <div 
                    className="grid grid-cols-12 gap-[1px] bg-white/5 rounded-2xl overflow-hidden relative z-10"
                    style={{ width: '480px', height: '480px' }}
                  >
                    {grid.map((row, rowIndex) => 
                      row.map((cell, colIndex) => (
                        <div
                          key={`${rowIndex}-${colIndex}`}
                          onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                          onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                          onMouseUp={handleMouseUp}
                          className={`w-full aspect-square cursor-crosshair transition-all duration-75 ${
                            cell === 1 ? 'bg-cyber-cyan shadow-[inset_0_0_10px_rgba(0,242,255,0.5)]' : 'bg-transparent hover:bg-white/5'
                          }`}
                        />
                      ))
                    )}
                  </div>
                </div>
                <p className="mt-8 text-[10px] text-slate-500 font-black uppercase tracking-[0.3em] flex items-center gap-3">
                  <div className="w-1.5 h-1.5 bg-cyber-cyan rounded-full animate-pulse"></div>
                  Neural Input Buffer
                </p>
             </div>

             {/* Right Column: Visualization */}
             <div className="col-span-3 space-y-6">
                <AnimatePresence mode="wait">
                  {prediction ? (
                    <motion.div 
                      className="space-y-6"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                    >
                      <div className="glass-card rounded-2xl p-8 text-center relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyber-cyan to-transparent"></div>
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4">Network Convergence</p>
                        <div className="text-9xl font-black text-white leading-none mb-6 glow-text-cyan">
                          {prediction}
                        </div>
                        <div className="space-y-2">
                           <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                             <span>Similarity Index</span>
                             <span className="text-cyber-cyan italic">{(confidence * 100).toFixed(1)}%</span>
                           </div>
                           <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                              <motion.div 
                                className="bg-cyber-cyan h-full shadow-[0_0_10px_#00f2ff]"
                                initial={{ width: 0 }}
                                animate={{ width: `${confidence * 100}%` }}
                              />
                           </div>
                        </div>
                      </div>

                      <div className="glass-card rounded-2xl p-6">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
                          <RotateCcw size={12} />
                          Reconstructed State
                        </p>
                        <div className="grid grid-cols-12 gap-[1px] bg-white/5 rounded-lg overflow-hidden border border-white/10 p-1">
                          {reconstructed?.map((row, ri) => 
                            row.map((cell, ci) => (
                              <div
                                key={`${ri}-${ci}`}
                                className={`w-full aspect-square ${cell === 1 ? 'bg-cyber-cyan/80' : 'bg-transparent'}`}
                              />
                            ))
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="glass-card rounded-2xl p-10 text-center border-dashed border-white/10">
                      <div className="w-16 h-16 mx-auto bg-white/5 text-slate-600 rounded-full flex items-center justify-center mb-6 border border-white/5">
                        <Cpu size={24} />
                      </div>
                      <h4 className="text-xs font-black uppercase tracking-widest text-white mb-3">Awaiting Signal</h4>
                      <p className="text-[10px] text-slate-500 leading-relaxed font-medium">
                        Input neural data into the grid to begin pattern associative recovery.
                      </p>
                    </div>
                  )}
                </AnimatePresence>
             </div>
          </div>
       </div>
    </div>
  );
};

export default HopfieldLab;
