import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.graph_objects as go

def load_neural_network(h5_file):
    # Load and return a Keras model from an H5 file
    return keras.models.load_model(h5_file)

def create_3d_representation(model):
    layers = []
    connections = []
    
    # Analyze the model structure
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            neurons = layer.units
            layers.append((i, neurons))
            
            # Create connections between this layer and the previous one
            if i > 0 and hasattr(model.layers[i-1], 'units'):
                prev_neurons = model.layers[i-1].units
                weights = layer.get_weights()
                if len(weights) > 0:
                    weights = weights[0]
                    for prev_n in range(prev_neurons):
                        for curr_n in range(neurons):
                            connections.append((i-1, prev_n, i, curr_n, weights[prev_n][curr_n]))

    # Check if there are layers to visualize
    if not layers:
        print("No suitable layers found for visualization.")
        return

    # Find the maximum number of neurons in any layer for scaling
    max_neurons = max(layer[1] for layer in layers)
    
    # Initialize lists to store neuron and edge coordinates
    neuron_x, neuron_y, neuron_z = [], [], []
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []

    # Calculate neuron positions
    for layer_idx, neuron_count in layers:
        x = layer_idx
        for n in range(neuron_count):
            y = n - neuron_count / 2
            z = 0
            neuron_x.append(x)
            neuron_y.append(y)
            neuron_z.append(z)

    # Calculate edge positions and colors
    for prev_layer, prev_n, curr_layer, curr_n, weight in connections:
        start_x = prev_layer
        start_y = prev_n - layers[prev_layer][1] / 2
        end_x = curr_layer
        end_y = curr_n - layers[curr_layer][1] / 2
        
        edge_x.extend([start_x, end_x, None])
        edge_y.extend([start_y, end_y, None])
        edge_z.extend([0, 0, None])
        
        color = 'red' if weight > 0 else 'green'
        edge_colors.extend([color, color, color])

    # Create scatter plot for neurons
    neurons_trace = go.Scatter3d(
        x=neuron_x, y=neuron_y, z=neuron_z,
        mode='markers',
        marker=dict(size=10, color='blue'),
        hoverinfo='none'
    )

    # Create line plot for edges
    edges_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=edge_colors, width=1),
        hoverinfo='none'
    )

    # Combine neuron and edge traces
    fig = go.Figure(data=[neurons_trace, edges_trace])

    # Add arrows to the edges
    for i in range(0, len(edge_x) - 1, 3):
        if edge_x[i] is not None and edge_x[i+1] is not None:
            fig.add_trace(go.Cone(
                x=[edge_x[i+1]],
                y=[edge_y[i+1]],
                z=[edge_z[i+1]],
                u=[edge_x[i+1] - edge_x[i]],
                v=[edge_y[i+1] - edge_y[i]],
                w=[0],
                sizemode="absolute",
                sizeref=0.001,
                showscale=False,
                colorscale=[[0, edge_colors[i]], [1, edge_colors[i]]],
            ))

    # Set up the layout for the 3D plot
    fig.update_layout(
        title='3D Neural Network Visualization',
        scene=dict(
            xaxis_title='Layers',
            yaxis_title='Neurons',
            zaxis_title='Depth',
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5),
            xaxis=dict(tickmode='linear', dtick=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        showlegend=False
    )

    # Save the figure as an interactive HTML file
    fig.write_html('neural_network_3d.html')

def main():
    # Specify the path to your Keras model file
    h5_file = 'modeltest.keras'
    # Load the model
    model = load_neural_network(h5_file)
    # Create and save the 3D representation
    create_3d_representation(model)
    print("3D representation saved as 'neural_network_3d.html'")

if __name__ == "__main__":
    main()