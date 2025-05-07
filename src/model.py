import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Enable GPU memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except Exception as e:
        print(f"Error setting memory growth: {e}")

# Enable mixed precision for faster training on Ampere GPUs
mixed_precision = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(mixed_precision)
print(f"Mixed precision policy set to: {mixed_precision.name}")


class TrainModel:
    def __init__(
        self, num_layers, width, batch_size, learning_rate, input_dim, output_dim
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network with proper GPU optimization
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation="relu", dtype="float16")(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation="relu", dtype="float16")(x)

        # Final layer uses float32 for numerical stability
        outputs = layers.Dense(self._output_dim, activation="linear", dtype="float32")(
            x
        )

        model = keras.Model(inputs=inputs, outputs=outputs, name="my_model")

        # Use AMP-friendly optimizer with epsilon parameter
        opt = Adam(learning_rate=self._learning_rate, epsilon=1e-7)

        model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=opt,
            # Enable JIT compilation for faster execution
            jit_compile=True,
        )
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values with GPU optimization
        """
        self._model.fit(
            states,
            q_sa,
            epochs=1,
            verbose=0,
            batch_size=min(self._batch_size, len(states)),
            use_multiprocessing=True,
            workers=4,
        )

    def save_model(self, path):
        """
        Save the current model in the folder as keras file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, "trained_model.keras"))
        try:
            plot_model(
                self._model,
                to_file=os.path.join(path, "model_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            print(f"Warning: Could not generate model visualization: {e}")

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        # First try to load .keras format
        model_file_path_keras = os.path.join(model_folder_path, "trained_model.keras")
        model_file_path_h5 = os.path.join(model_folder_path, "trained_model.h5")

        if os.path.isfile(model_file_path_keras):
            loaded_model = load_model(model_file_path_keras)
            return loaded_model
        elif os.path.isfile(model_file_path_h5):
            loaded_model = load_model(model_file_path_h5)
            return loaded_model
        else:
            sys.exit("Model not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    @property
    def input_dim(self):
        return self._input_dim
