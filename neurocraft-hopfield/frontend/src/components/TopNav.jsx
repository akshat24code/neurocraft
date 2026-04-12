import React from 'react';
import { Bell, Settings } from 'lucide-react';

const TopNav = () => {
  return (
    <div className="h-20 flex items-center justify-between px-10 border-b border-white/5 bg-cyber-bg/50 backdrop-blur-md sticky top-0 z-40">
      <div className="flex gap-8 h-full items-center">
        {['Beginner', 'Explorer', 'Research'].map((tab, idx) => (
          <button 
            key={tab}
            className={`h-full px-2 text-[10px] font-black uppercase tracking-[0.2em] transition-all relative ${
              idx === 0 ? 'text-cyber-cyan nav-active' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-8">
        <div className="flex gap-4">
          <button className="text-slate-500 hover:text-white transition-colors relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-cyber-lime rounded-full border-2 border-cyber-bg"></span>
          </button>
          <button className="text-slate-500 hover:text-white transition-colors">
            <Settings className="w-5 h-5" />
          </button>
        </div>

        <div className="flex items-center gap-4 border-l border-white/10 pl-8">
          <div className="text-right">
            <p className="text-xs font-bold text-white leading-none">Researcher #402</p>
            <p className="text-[9px] font-bold text-cyber-cyan uppercase tracking-wider mt-1">Level 4 Node</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-cyber-cyan/20 border border-cyber-cyan/30 flex items-center justify-center overflow-hidden">
            <img 
              src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" 
              alt="avatar" 
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopNav;
