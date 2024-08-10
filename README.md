# Neural Network Visualizer

A Python application for visualizing the structure of Keras neural network models in 3D using Plotly and PyQt5.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features
- Load Keras neural network models from `.h5` or `.keras` files.
- Visualize the network structure in 3D.
- Display neurons and connections between layers with different colors based on weights.
- Save the visualization as an HTML file.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/neural-network-visualizer.git
    cd neural-network-visualizer
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the application:
    ```sh
    python main.py
    ```
2. Use the GUI to browse and select a Keras model file (`.h5` or `.keras`).
3. Click on the `Visualize` button to generate a 3D visualization of the model. The output will be saved as an HTML file in a `downloads` folder in the same directory as the selected model file.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Plotly
- PyQt5

To install the dependencies, you can use the following command:
```sh
pip install tensorflow keras numpy plotly pyqt5


