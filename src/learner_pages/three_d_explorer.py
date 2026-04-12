import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

def plot_3d_architecture(in_dim, hidden_sizes, out_dim, animate=False):
    layers = [in_dim] + hidden_sizes + [out_dim]
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    
    edge_x = []
    edge_y = []
    edge_z = []
    
    layer_dist = 2.0
    
    # Layer colors
    colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    
    for l_idx, n_nodes in enumerate(layers):
        z = l_idx * layer_dist
        # Arrange nodes in a grid or circle for the layer
        grid_size = int(np.ceil(np.sqrt(n_nodes)))
        for i in range(n_nodes):
            row = i // grid_size
            col = i % grid_size
            # Center the layer nodes
            x = col - (grid_size - 1) / 2
            y = row - (grid_size - 1) / 2
            
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(f"Layer {l_idx} Node {i}")
            node_color.append(colors[l_idx % len(colors)])
            
    # Edges
    node_offset = 0
    for l_idx in range(len(layers) - 1):
        n_current = layers[l_idx]
        n_next = layers[l_idx + 1]
        
        current_offset = node_offset
        next_offset = node_offset + n_current
        
        # Grid sizes for layout calc
        curr_g = int(np.ceil(np.sqrt(n_current)))
        next_g = int(np.ceil(np.sqrt(n_next)))
        
        for i in range(n_current):
            for j in range(n_next):
                ei = current_offset + i
                ej = next_offset + j
                
                edge_x.extend([node_x[ei], node_x[ej], None])
                edge_y.extend([node_y[ei], node_y[ej], None])
                edge_z.extend([node_z[ei], node_z[ej], None])
                
        node_offset += n_current

    # Nodes trace
    nodes_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=6, color=node_color, opacity=0.9, line=dict(color='white', width=1)),
        text=node_text,
        hoverinfo='text',
        name='Neurons'
    )
    
    # Edges trace
    edges_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(151, 192, 255, 0.15)', width=1),
        hoverinfo='none',
        name='Synapses'
    )
    
    fig = go.Figure(data=[edges_trace, nodes_trace])
    
    if animate:
        # Add "pulses"
        pulse_x = []
        pulse_y = []
        pulse_z = []
        
        # Simple pulse simulation: pick random edges
        for _ in range(20):
            l = np.random.randint(0, len(layers) - 1)
            # random node in l, random node in l+1
            start_off = sum(layers[:l])
            end_off = sum(layers[:l+1])
            i = np.random.randint(0, layers[l])
            j = np.random.randint(0, layers[l+1])
            
            p_start = np.array([node_x[start_off + i], node_y[start_off + i], node_z[start_off + i]])
            p_end = np.array([node_x[end_off + j], node_y[end_off + j], node_z[end_off + j]])
            
            # Position along the edge
            t = np.random.rand()
            p = p_start + t * (p_end - p_start)
            pulse_x.append(p[0])
            pulse_y.append(p[1])
            pulse_z.append(p[2])
            
        fig.add_trace(go.Scatter3d(
            x=pulse_x, y=pulse_y, z=pulse_z,
            mode='markers',
            marker=dict(size=4, color='#67e8f9', symbol='diamond'),
            name='Signals'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Layers"),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700
    )
    
    return fig

def three_d_explorer_page():
    st.markdown('<div class="nc-section-label">Immersive Lab</div>', unsafe_allow_html=True)
    st.title("3D Neural Architecture Explorer")
    st.caption("Step inside the network. Visualize complex multi-layer topologies and signal flows in interactive 3D space.")

    with st.sidebar.expander("Control Center", expanded=True):
        st.subheader("Architecture")
        preset = st.selectbox("Load Preset", ["Custom", "Simple Perceptron", "Deep MLP", "Wide Hidden Layer"])
        
        if preset == "Simple Perceptron":
            p_in, p_hid, p_hsz, p_out = 2, 1, [1], 1
        elif preset == "Deep MLP":
            p_in, p_hid, p_hsz, p_out = 8, 4, [12, 12, 8, 8], 4
        elif preset == "Wide Hidden Layer":
            p_in, p_hid, p_hsz, p_out = 4, 1, [16], 2
        else:
            p_in, p_hid, p_hsz, p_out = 4, 2, [8, 8], 3

        in_dim = st.slider("Inputs", 1, 16, p_in)
        n_hidden = st.slider("Hidden Layers", 1, 5, p_hid)
        hidden_sizes = []
        st.caption("Neurons per layer:")
        for i in range(n_hidden):
            default_h = p_hsz[i] if i < len(p_hsz) else 8
            h = st.slider(f"Layer {i+1}", 1, 16, default_h, key=f"td_h_{i}")
            hidden_sizes.append(h)
        out_dim = st.slider("Outputs", 1, 10, p_out)
        
        st.divider()
        st.subheader("Visuals")
        show_pulses = st.toggle("Simulate Signal Pulses", value=True)
        auto_rotate = st.toggle("Enable Camera Rotation", value=False)

    # Info Banner
    st.markdown("""
        <div class="nc-panel" style="margin-bottom: 20px;">
            <p><strong>💡 Interaction Tips:</strong> Click and drag to rotate. Scroll to zoom. Use the right-click menu to pan.
            Nodes are arranged in layer planes along the depth (Z) axis.</p>
        </div>
    """, unsafe_allow_html=True)

    fig = plot_3d_architecture(in_dim, hidden_sizes, out_dim, animate=show_pulses)
    
    if auto_rotate:
        # Simple rotation animation check
        # Streamlit doesn't support smooth continuous animation well without refresh, 
        # but we can set the camera orbit.
        fig.update_layout(scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5*np.cos(time.time()*0.2), y=1.5*np.sin(time.time()*0.2), z=0.5)
        ))

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Stats
    total_neurons = in_dim + sum(hidden_sizes) + out_dim
    total_synapses = in_dim * hidden_sizes[0]
    for i in range(len(hidden_sizes)-1):
        total_synapses += hidden_sizes[i] * hidden_sizes[i+1]
    total_synapses += hidden_sizes[-1] * out_dim

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Neurons", total_neurons)
    with c2:
        st.metric("Total Synapses", total_synapses)
    with c3:
        st.metric("Layer Depth", 2 + n_hidden)

    st.divider()
    st.markdown("### 🔍 Exploring 3D Spaces in Deep Learning")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**1. Loss Landscapes**")
            st.write("In training, we optimize weights by descending a high-dimensional loss surface. "
                     "Visualizing this as a 3D mountain range helps explain local minima and saddle points.")
    with col2:
        with st.container(border=False):
            st.markdown("**2. Feature Projections (Latent Space)**")
            st.write("Hidden layers compress information into 'latent vectors'. 3D clusters in these spaces "
                     "often reveal how a model learns to group similar concepts together.")

if __name__ == "__main__":
    three_d_explorer_page()
