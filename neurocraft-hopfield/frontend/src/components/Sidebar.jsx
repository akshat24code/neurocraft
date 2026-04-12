import React from 'react';
import { 
  FlaskConical, 
  Eye, 
  Share2, 
  Sigma, 
  Gamepad2, 
  UserCircle2, 
  Hexagon 
} from 'lucide-react';

const SidebarItem = ({ icon: Icon, label, active, onClick }) => (
  <div 
    onClick={onClick}
    className={`flex items-center gap-4 px-6 py-4 cursor-pointer transition-all duration-300 group ${
      active ? 'text-cyber-cyan bg-cyber-cyan/5 border-r-2 border-cyber-cyan' : 'text-slate-400 hover:text-white hover:bg-white/5'
    }`}
  >
    <Icon className={`w-5 h-5 ${active ? 'drop-shadow-[0_0_8px_rgba(0,242,255,0.8)]' : 'group-hover:scale-110 transition-transform'}`} />
    <span className="text-xs font-bold uppercase tracking-widest leading-none">{label}</span>
  </div>
);

const Sidebar = ({ activeTab, setActiveTab }) => {
  return (
    <div className="w-64 h-screen bg-black border-r border-white/5 flex flex-col fixed left-0 top-0 z-50">
      <div className="p-8 mb-4">
        <div className="flex items-center gap-3">
          <Hexagon className="w-8 h-8 text-cyber-cyan fill-cyber-cyan/20" />
          <div>
            <h1 className="text-xl font-black tracking-tighter leading-none italic">NEUROLAB</h1>
            <p className="text-[8px] font-bold text-slate-500 tracking-[.3em] mt-1 uppercase">Neural Ether Interface</p>
          </div>
        </div>
      </div>

      <nav className="flex-1">
        <SidebarItem 
          icon={FlaskConical} 
          label="Learning Lab" 
          active={activeTab === 'Learning Lab'} 
          onClick={() => setActiveTab('Learning Lab')}
        />
        <SidebarItem 
          icon={Eye} 
          label="Vision Lab" 
          active={activeTab === 'Vision Lab'} 
          onClick={() => setActiveTab('Vision Lab')}
        />
        <SidebarItem 
          icon={Share2} 
          label="Sequence Models" 
          active={activeTab === 'Sequence Models'} 
          onClick={() => setActiveTab('Sequence Models')}
        />
        <SidebarItem 
          icon={Sigma} 
          label="Math Engine" 
          active={activeTab === 'Math Engine'} 
          onClick={() => setActiveTab('Math Engine')}
        />
        <SidebarItem 
          icon={Gamepad2} 
          label="AI Playground" 
          active={activeTab === 'AI Playground'} 
          onClick={() => setActiveTab('AI Playground')}
        />
        <SidebarItem 
          icon={UserCircle2} 
          label="AI Mentor" 
          active={activeTab === 'AI Mentor'} 
          onClick={() => setActiveTab('AI Mentor')}
        />
      </nav>

      <div className="p-6 mt-auto">
        <button className="w-full py-4 rounded-sm border border-cyber-cyan/30 text-cyber-cyan text-[10px] font-black uppercase tracking-[0.2em] hover:bg-cyber-cyan/10 transition-all duration-300 relative group overflow-hidden">
          <span className="relative z-10">Upgrade to Researcher</span>
          <div className="absolute inset-0 bg-cyber-cyan/5 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
