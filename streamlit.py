import streamlit as st
import path_planning as pp
import matplotlib.pyplot as plt
from pso import PSO
import numpy as np

# Set smaller font sizes for plots
plt.rcParams.update({
    'font.size': 5,
    'axes.labelsize': 5,
    'axes.titlesize': 5,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
    'legend.fontsize': 4,
    'figure.titlesize': 5
})

st.set_page_config(page_title="Path Planning with PSO", layout="wide")

st.title("Path Planning with Particle Swarm Optimization")

# Initialize session state variables
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = [
        {'center': [0, 40], 'radius': 5},
        {'center': [30, 30], 'radius': 9},
        {'center': [30, 70], 'radius': 10},
        {'center': [50, 10], 'radius': 8},
        {'center': [60, 80], 'radius': 15},
        {'center': [70, 40], 'radius': 12},
        {'center': [80, 20], 'radius': 7},
        {'center': [20, 30], 'radius': 7},
    ]
if 'should_run_pso' not in st.session_state:
    st.session_state.should_run_pso = False

# Create two columns for the main layout
col_main, col_sidebar = st.columns([1.2, 1])  # Changed ratio to 1.2:1

with col_sidebar:
    # Sidebar for parameters
    st.header("Environment Parameters")
    width = st.slider("Environment Width", 50, 200, 100)
    height = st.slider("Environment Height", 50, 200, 100)
    robot_radius = st.slider("Robot Radius", 0.5, 5.0, 1.0)

    # Start and goal positions
    st.subheader("Start and Goal")
    col1, col2 = st.columns(2)
    with col1:
        start_x = st.number_input("Start X", 0, width-1, 5)
        start_y = st.number_input("Start Y", 0, height-1, 5)
    with col2:
        goal_x = st.number_input("Goal X", 0, width-1, width-5)
        goal_y = st.number_input("Goal Y", 0, height-1, height-5)

    # PSO Parameters
    st.header("PSO Parameters")
    max_iter = st.slider("Maximum Iterations", 50, 500, 200)
    pop_size = st.slider("Population Size", 20, 200, 100)
    c1 = st.slider("C1 (Cognitive Parameter)", 0.1, 3.0, 2.0)
    c2 = st.slider("C2 (Social Parameter)", 0.1, 3.0, 1.0)
    w = st.slider("Inertia Weight", 0.1, 1.0, 0.8)

    # Obstacle placement
    st.header("Obstacle Placement")
    st.info("Enter obstacle position and radius, then click 'Add Obstacle'")

    # Create columns for obstacle input
    col1, col2, col3 = st.columns(3)
    with col1:
        new_x = st.number_input("X Position", 0, width-1, width//2)
    with col2:
        new_y = st.number_input("Y Position", 0, height-1, height//2)
    with col3:
        new_radius = st.number_input("Radius", 1, min(20, min(width, height)//2), 5)

    # Add obstacle button
    if st.button("➕ Add Obstacle"):
        st.session_state.obstacles.append({'center': [new_x, new_y], 'radius': new_radius})
        st.rerun()

    # Display current obstacles
    if st.session_state.obstacles:
        st.subheader("Current Obstacles")
        for i, obs in enumerate(st.session_state.obstacles):
            st.write(f"Obstacle {i+1}: Position ({obs['center'][0]}, {obs['center'][1]}), Radius: {obs['radius']}")
            if st.button(f"❌ Remove Obstacle {i+1}", key=f"remove_{i}"):
                st.session_state.obstacles.pop(i)
                st.rerun()

    # Clear all obstacles button
    if st.button("🗑️ Clear All Obstacles", key="clear_all"):
        st.session_state.obstacles = []
        st.rerun()

    # Run button
    if st.button("▶️ Run PSO", key="run_pso"):
        st.session_state.should_run_pso = True
        st.rerun()

with col_main:
    # Create environment with current obstacles
    env_params = {
        'width': width,
        'height': height,
        'robot_radius': robot_radius,
        'start': [start_x, start_y],
        'goal': [goal_x, goal_y],
    }
    env = pp.Environment(**env_params)

    # Add obstacles
    for obs in st.session_state.obstacles:
        env.add_obstacle(pp.Obstacle(**obs))

    # Create a placeholder for the plot
    plot_placeholder = st.empty()

    # Create figure for plotting with smaller size
    fig = plt.figure(figsize=[2.5, 2.5])  # Reduced figure size to 2.5x2.5
    pp.plot_environment(env)
    plt.grid(True, linewidth=0.5)  # Make grid lines thinner
    plt.tight_layout()  # Add tight layout to prevent label cutoff

    # Display the plot
    plot_placeholder.pyplot(fig, use_container_width=True)  # Use container width

    # Create cost function
    cost_function = pp.EnvCostFunction(env, num_control_points=3, resolution=50)  # Default values

    # Optimization Problem
    problem = {
        'num_var': 2*3,
        'var_min': 0,
        'var_max': 1,
        'cost_function': cost_function,
    }

    # PSO parameters
    pso_params = {
        'max_iter': max_iter,
        'pop_size': pop_size,
        'c1': c1,
        'c2': c2,
        'w': w,
        'wdamp': 1.0,  # Default value
        'resetting': 25,  # Default value
    }

    # Run PSO if button was clicked
    if st.session_state.should_run_pso:
        # Create a placeholder for the plot
        plot_placeholder = st.empty()
        
        # Create figure for plotting with smaller size
        fig = plt.figure(figsize=[2.5, 2.5])  # Reduced figure size to 2.5x2.5
        pp.plot_environment(env)
        plt.grid(True, linewidth=0.5)  # Make grid lines thinner
        plt.tight_layout()  # Add tight layout to prevent label cutoff
        
        class PathVisualizer:
            def __init__(self):
                self.path_line = None
                
            def update(self, data):
                it = data['it']
                sol = data['gbest']['details']['sol']
                
                # Update the path
                if self.path_line is None:
                    self.path_line = pp.plot_path(sol, color='b')
                else:
                    pp.update_path(sol, self.path_line)
                    
                length = data['gbest']['details']['length']
                plt.title(f"Iteration: {it}, Length: {length:.2f}")
                
                # Update the plot in the placeholder
                plot_placeholder.pyplot(fig, use_container_width=True)  # Use container width
        
        # Create visualizer instance
        visualizer = PathVisualizer()
        
        # Run PSO
        bestsol, pop = PSO(problem, callback=visualizer.update, **pso_params)
        
        # Display final results
        st.success("Optimization completed!")
        st.write(f"Final path length: {bestsol['details']['length']:.2f}")
        
        # Plot final path in red
        final_path = pp.plot_path(bestsol['details']['sol'], color='r')
        plt.title(f"Final Path - Length: {bestsol['details']['length']:.2f}")
        plot_placeholder.pyplot(fig, use_container_width=True)  # Use container width
        
        # Reset the run_pso flag
        st.session_state.should_run_pso = False 