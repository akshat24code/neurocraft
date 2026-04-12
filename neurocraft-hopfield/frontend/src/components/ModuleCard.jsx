import React from 'react';

const ModuleCard = ({ title, description, progress, status, level, image }) => {
  const isLocked = status === 'locked';
  const isInProgress = status === 'in-progress';
  const isCompleted = progress === 100;

  return (
    <div className={`glass-card rounded-2xl overflow-hidden group transition-all duration-500 hover:-translate-y-2 ${
      isInProgress ? 'neural-border border-cyber-cyan/30' : ''
    } ${isLocked ? 'opacity-60 grayscale-[0.5]' : ''}`}>
      <div className="h-48 relative overflow-hidden">
        <img 
          src={image} 
          alt={title} 
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-cyber-bg to-transparent opacity-80"></div>
        
        {level && (
          <div className="absolute top-4 left-4 bg-cyber-cyan/20 backdrop-blur-md border border-cyber-cyan/30 px-3 py-1 rounded text-[8px] font-black uppercase tracking-widest text-cyber-cyan">
            Level: {level}
          </div>
        )}

        {isInProgress && (
          <div className="absolute top-4 left-4 bg-cyber-lime/20 backdrop-blur-md border border-cyber-lime/30 px-3 py-1 rounded text-[8px] font-black uppercase tracking-widest text-cyber-lime">
            In Progress
          </div>
        )}

        {isLocked && (
          <div className="absolute top-4 left-4 bg-white/5 backdrop-blur-md border border-white/10 px-3 py-1 rounded text-[8px] font-black uppercase tracking-widest text-slate-400">
            Locked
          </div>
        )}
      </div>

      <div className="p-6">
        <h4 className="text-xl font-black tracking-tight mb-2 group-hover:text-cyber-cyan transition-colors">{title}</h4>
        <p className="text-[11px] text-slate-400 font-medium leading-relaxed mb-6 h-8 line-clamp-2">
          {description}
        </p>

        <div className="flex justify-between items-center mb-2">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Progress</span>
          <span className={`text-[10px] font-black italic ${isCompleted ? 'text-cyber-lime' : isInProgress ? 'text-cyber-cyan' : 'text-slate-500'}`}>
            {progress}%
          </span>
        </div>
        <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden mb-6">
          <div 
            className={`h-full transition-all duration-1000 ${isCompleted ? 'bg-cyber-lime shadow-[0_0_8px_#c1ff00]' : 'bg-cyber-cyan shadow-[0_0_8px_#00f2ff]'}`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        <button className={`w-full py-3 rounded-lg text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-300 relative overflow-hidden ${
          isLocked ? 'bg-white/5 text-slate-600 cursor-not-allowed' : 
          isInProgress ? 'bg-cyber-cyan text-black hover:shadow-[0_0_20px_rgba(0,242,255,0.4)]' : 
          'bg-white/5 text-white hover:bg-white/10'
        }`}>
          {isLocked ? 'Access Denied' : isInProgress ? 'Continue Lab' : 'Review Concept'}
        </button>
      </div>
      
      {/* Glow highlight */}
      <div className={`absolute top-0 right-0 w-32 h-32 blur-[60px] opacity-0 group-hover:opacity-20 transition-opacity ${
        isInProgress ? 'bg-cyber-cyan' : isCompleted ? 'bg-cyber-lime' : ''
      }`}></div>
    </div>
  );
};

export default ModuleCard;
