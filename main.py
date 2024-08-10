import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.graph_objects as go
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog

def load_neural_network(h5_file):
    return keras.models.load_model(h5_file)

def create_3d_representation(model, output_path):
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
    fig.write_html(output_path)

class NeuralNetworkVisualizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model_path = None

    def initUI(self):
        self.setWindowTitle('Neural Network Visualizer')
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Select a Keras model file (.h5 or .keras)')
        layout.addWidget(self.label)

        self.browse_button = QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        self.visualize_button = QPushButton('Visualize')
        self.visualize_button.clicked.connect(self.visualize)
        self.visualize_button.setEnabled(False)
        layout.addWidget(self.visualize_button)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Keras Model", "", 
                                                   "Keras Model (*.h5 *.keras)", options=options)
        if file_name:
            self.model_path = file_name
            self.visualize_button.setEnabled(True)
            self.status_label.setText(f"Selected: {os.path.basename(self.model_path)}")

    def visualize(self):
        if not self.model_path:
            self.status_label.setText("Please select a model file first.")
            return

        try:
            model = load_neural_network(self.model_path)
            
            # Create 'downloads' folder if it doesn't exist
            downloads_folder = os.path.join(os.path.dirname(self.model_path), "downloads")
            os.makedirs(downloads_folder, exist_ok=True)
            
            # Generate output file path
            output_filename = f"neural_network_3d_{os.path.splitext(os.path.basename(self.model_path))[0]}.html"
            output_path = os.path.join(downloads_folder, output_filename)
            
            create_3d_representation(model, output_path)
            self.status_label.setText(f"Visualization saved: {output_filename}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

def main():
    app = QApplication(sys.argv)
    ex = NeuralNetworkVisualizerApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()