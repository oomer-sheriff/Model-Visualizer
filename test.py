import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.graph_objects as go
import keras
from keras import layers
from keras import ops
model = keras.Sequential(
    [   
        keras.layers.Dense(2, activation="relu", name="layer1"),
        keras.layers.Dense(3, activation="relu", name="layer2"),
        keras.layers.Dense(4, name="layer3"),
    ]
)
x = ops.ones((1, 4))
y = model(x)
model.save("modeltest.keras")