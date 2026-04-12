import React from 'react';

const StatCard = ({ label, value, subtext, progress, color = "cyan" }) => {
  const accentColor = color === "cyan" ? "bg-cyber-cyan" : color === "lime" ? "bg-cyber-lime" : "bg-cyber-purple";
  const shadowColor = color === "cyan" ? "shadow-cyber-cyan/20" : color === "lime" ? "shadow-cyber-lime/20" : "shadow-cyber-purple/20";

  return (
    <div className="glass-card p-6 rounded-xl flex flex-col justify-between min-h-[160px] relative overflow-hidden group">
      <div className="relative z-10">
        <p className="stat-label">{label}</p>
        <h2 className="stat-value">{value}</h2>
        {subtext && (
          <p className="text-[10px] text-slate-500 font-medium italic mt-1 flex items-center gap-1">
            {color === 'purple' && <span className="text-cyber-purple">↗</span>}
            {subtext}
          </p>
        )}
      </div>
      
      {progress !== undefined && (
        <div className="w-full h-[6px] bg-white/5 rounded-full mt-4 overflow-hidden">
          <div 
            className={`h-full ${accentColor} ${shadowColor} shadow-[0_0_10px] transition-all duration-1000`}
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}

      {/* Background flare */}
      <div className={`absolute -right-4 -bottom-4 w-24 h-24 rounded-full blur-[40px] opacity-10 ${accentColor} group-hover:opacity-20 transition-opacity`}></div>
    </div>
  );
};

export default StatCard;
