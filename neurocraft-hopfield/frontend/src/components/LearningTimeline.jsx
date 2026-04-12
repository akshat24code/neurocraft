import React from 'react';
import { Check, Zap, Hash, Layers, Network, Boxes } from 'lucide-react';

const TimelineStep = ({ icon: Icon, label, status, subtext }) => {
  const isCompleted = status === 'completed';
  const isActive = status === 'active';
  const isLocked = status === 'locked';

  return (
    <div className="flex flex-col items-center gap-3 relative z-10 w-32">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-500 ${
        isCompleted ? 'bg-cyber-lime/10 border border-cyber-lime text-cyber-lime shadow-[0_0_15px_rgba(193,255,0,0.3)]' :
        isActive ? 'bg-cyber-cyan/10 border-2 border-cyber-cyan text-cyber-cyan shadow-[0_0_20px_rgba(0,242,255,0.4)] scale-110' :
        'bg-white/5 border border-white/10 text-slate-500'
      }`}>
        <Icon className="w-5 h-5" />
        {isActive && (
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-cyber-cyan text-black px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-tighter shadow-lg">
            Current
          </div>
        )}
      </div>
      <div className="text-center">
        <p className={`text-[10px] font-black uppercase tracking-widest ${isActive ? 'text-white' : 'text-slate-500'}`}>{label}</p>
        <p className="text-[8px] font-bold text-slate-600 uppercase tracking-tighter mt-0.5">{subtext}</p>
      </div>
    </div>
  );
};

const LearningTimeline = () => {
  return (
    <div className="glass-card p-10 rounded-2xl relative overflow-hidden bg-gradient-to-r from-white/[0.02] to-transparent">
      <div className="flex items-center gap-4 mb-8">
        <Network className="w-6 h-6 text-cyber-cyan" />
        <h3 className="text-sm font-black uppercase tracking-[0.3em] text-white">Learning Path Timeline</h3>
      </div>

      <div className="relative flex justify-between items-start pt-4">
        {/* Connection Line */}
        <div className="absolute top-10 left-10 right-10 h-[2px] timeline-line opacity-50"></div>
        
        <TimelineStep icon={Check} label="Basics" status="completed" subtext="Completed" />
        <TimelineStep icon={Zap} label="Activation" status="completed" subtext="Completed" />
        <TimelineStep icon={Hash} label="Loss" status="active" subtext="In Progress" />
        <TimelineStep icon={Layers} label="Gradient" status="locked" subtext="Locked" />
        <TimelineStep icon={Network} label="Neural Network" status="locked" subtext="Locked" />
        <TimelineStep icon={Boxes} label="Advanced" status="locked" subtext="Locked" />
      </div>

      {/* Grid Pattern Background */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle, white 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
    </div>
  );
};

export default LearningTimeline;
