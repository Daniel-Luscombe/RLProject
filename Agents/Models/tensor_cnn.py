import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape, n_actions, lr):
    """
    Build a CNN model for reinforcement learning.

    Args:
        input_shape (tuple): Shape of the input (height, width, channels).
        n_actions (int): Number of possible actions.
        lr (float): Learning rate for the optimizer.

    Returns:
        model: A compiled Keras model.
    """
    # Define the CNN model
    model = tf.keras.Sequential([
            layers.Input(shape=input_shape),                 # e.g. (48, 48, 3)
            layers.Conv2D(32, 8, strides=4, activation='relu'),  # large receptive field
            layers.Conv2D(64, 4, strides=2, activation='relu'),
            layers.Conv2D(64, 3, strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),  # learns high-level "features"
            layers.Dense(n_actions)  # output: one Q-value per action
        ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error')
    return model

def save_model(model, path):
    print(f"Saving Model to {path}")
    model.save(path)

def load_model(path, lr):
    print(f"Model loaded from {path}")
    model = tf.keras.models.load_model(path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error'
    )
    return model