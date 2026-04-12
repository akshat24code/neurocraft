import React from 'react';
import StatCard from './StatCard';
import LearningTimeline from './LearningTimeline';
import ModuleCard from './ModuleCard';
import { Layout } from 'lucide-react';

const NeuralWorkspace = () => {
  return (
    <div className="flex-1 min-h-screen pl-64 bg-cyber-bg transition-all duration-500">
      <div className="p-10 max-w-7xl mx-auto space-y-12">
        
        {/* Header Section */}
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-5xl font-black tracking-tight mb-4">Neural Workspace</h1>
            <p className="text-slate-400 max-w-2xl font-medium leading-relaxed">
              Visualizing high-dimensional learning vectors across the ether interface. 
              Your synchronization with the local model is at <span className="text-cyber-cyan font-bold italic">84%</span>.
            </p>
          </div>
          <div className="flex items-center gap-3 bg-cyber-lime/10 border border-cyber-lime/30 px-6 py-3 rounded-xl">
            <div className="w-2 h-2 bg-cyber-lime rounded-full animate-pulse shadow-[0_0_10px_#c1ff00]"></div>
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-cyber-lime">Neural Stream: Active</span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-5 gap-6">
          <StatCard label="Total Modules" value="128" progress={45} color="cyan" />
          <StatCard label="Topics Covered" value="14" progress={75} color="lime" />
          <StatCard label="Math Concepts" value="42" subtext="Advanced Calculus Unlocked" color="purple" />
          <StatCard label="Experiments" value="856" subtext="Last active: 2m ago" color="cyan" />
          <StatCard label="3D Visualizations" value="31" subtext="Rendering in Cloud" color="cyan" />
        </div>

        {/* Learning Path */}
        <LearningTimeline />

        {/* Module Section */}
        <div className="space-y-8">
          <div className="flex items-center gap-4">
            <Layout className="w-6 h-6 text-cyber-purple" />
            <h3 className="text-sm font-black uppercase tracking-[0.3em] text-white">Active Learning Modules</h3>
          </div>

          <div className="grid grid-cols-3 gap-8">
            <ModuleCard 
              title="Perceptron"
              description="Master the fundamental unit of artificial intelligence: the single-layer neural node."
              progress={100}
              level="Novice"
              image="https://images.unsplash.com/photo-1620712943543-bcc4628c6bb3?q=80&w=1000&auto=format&fit=crop"
            />
            <ModuleCard 
              title="Multi-Layer Perceptron"
              description="Moving beyond linear separability with hidden layers and backpropagation."
              progress={64}
              status="in-progress"
              image="https://images.unsplash.com/photo-1675271591211-126ad94e495d?q=80&w=1000&auto=format&fit=crop"
            />
            <ModuleCard 
              title="CNN Image Recognition"
              description="Advanced computer vision using convolutional layers for spatial feature extraction."
              progress={0}
              status="locked"
              image="https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1000&auto=format&fit=crop"
            />
          </div>
        </div>
      </div>

      {/* Floating Action Button */}
      <button className="fixed bottom-12 right-12 w-16 h-16 bg-cyber-lime rounded-2xl flex items-center justify-center text-black shadow-[0_0_30px_rgba(193,255,0,0.4)] hover:scale-110 active:scale-95 transition-all duration-300 z-50">
        <span className="text-3xl font-black">+</span>
      </button>
    </div>
  );
};

export default NeuralWorkspace;
