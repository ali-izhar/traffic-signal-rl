"""Neural Network Models for Deep Q-Learning"""

import os
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TrainModel:
    """Neural network model for training a Deep Q-Network agent.

    - Q(s,a) = E[R_t + γ·max_a' Q(s',a') | s_t=s, a_t=a]
    - Loss function: L(θ) = E[(r + γ·max_a' Q(s',a';θ-) - Q(s,a;θ))²]
    where θ are the network parameters and θ- are the target network parameters

    Attributes:
        _input_dim: Dimension of the state space
        _output_dim: Dimension of the action space
        _batch_size: Number of samples per gradient update
        _learning_rate: Learning rate for the optimizer
        _model: Keras model instance
    """

    def __init__(
        self,
        num_layers: int,
        width: int,
        batch_size: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the training model for DQN.

        Args:
            num_layers: Number of hidden layers in the network
            width: Width of each hidden layer (number of neurons)
            batch_size: Number of samples per gradient update
            learning_rate: Learning rate for the Adam optimizer
            input_dim: Dimension of the state space (input)
            output_dim: Dimension of the action space (output)
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

    def _build_model(self, num_layers: int, width: int) -> keras.Model:
        """Build and compile a fully connected deep neural network.

        The network architecture consists of:
        - Input layer: input_dim neurons
        - Hidden layers: num_layers layers with width neurons each
        - Output layer: output_dim neurons (one for each action)

        Uses ReLU activation for hidden layers and linear activation for output.

        Args:
            num_layers: Number of hidden layers
            width: Number of neurons in each hidden layer

        Returns:
            Compiled Keras model
        """
        # Use Keras Functional API for more flexibility
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation="relu", kernel_initializer="he_uniform")(
            inputs
        )

        for _ in range(num_layers):
            x = layers.Dense(width, activation="relu", kernel_initializer="he_uniform")(
                x
            )

        outputs = layers.Dense(
            self._output_dim, activation="linear", kernel_initializer="glorot_uniform"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_model")

        model.compile(
            loss=losses.mean_squared_error,
            optimizer=Adam(learning_rate=self._learning_rate),
        )

        return model

    def predict_one(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a single state.

        Args:
            state: State vector of shape (input_dim,)

        Returns:
            Array of Q-values for each action
        """
        # Reshape to add batch dimension and ensure numpy array
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict Q-values for a batch of states.

        Args:
            states: Batch of state vectors of shape (batch_size, input_dim)

        Returns:
            Array of Q-values for each state-action pair
        """
        return self._model.predict(states, verbose=0)

    def train_batch(self, states: np.ndarray, q_values: np.ndarray) -> None:
        """Train the network using a batch of states and target Q-values.

        Implements the DQN training step with the loss function:
        L(θ) = E[(target_q - Q(s,a;θ))²]

        Args:
            states: Batch of state vectors of shape (batch_size, input_dim)
            q_values: Target Q-values of shape (batch_size, output_dim)
        """
        self._model.fit(
            states, q_values, batch_size=self._batch_size, epochs=1, verbose=0
        )

    def save_model(self, path: str) -> None:
        """Save the current model as an h5 file and generate a structure diagram.

        Args:
            path: Directory path where the model should be saved
        """
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        # Save model
        model_path = os.path.join(path, "trained_model.h5")
        self._model.save(model_path)

        # Generate and save model structure visualization
        try:
            plot_model(
                self._model,
                to_file=os.path.join(path, "model_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            print(f"Warning: Could not generate model plot: {e}")

    @property
    def input_dim(self) -> int:
        """Get input dimension of the model."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Get output dimension of the model."""
        return self._output_dim

    @property
    def batch_size(self) -> int:
        """Get batch size used for training."""
        return self._batch_size


class TestModel:
    """Model for testing/evaluating a trained DQN agent.

    This class loads a pre-trained model and provides prediction functionality
    for inference without further training.

    Attributes:
        _input_dim: Dimension of the state space
        _model: Loaded Keras model instance
    """

    def __init__(self, input_dim: int, model_path: str) -> None:
        """Initialize the test model.

        Args:
            input_dim: Dimension of the state space (input)
            model_path: Path to the directory containing the trained model
        """
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path: str) -> keras.Model:
        """Load a pre-trained model from the specified path.

        Args:
            model_folder_path: Directory path containing the model file

        Returns:
            Loaded Keras model

        Raises:
            SystemExit: If the model file does not exist
        """
        model_file_path = os.path.join(model_folder_path, "trained_model.h5")

        if os.path.isfile(model_file_path):
            try:
                return load_model(model_file_path)
            except Exception as e:
                import sys

                sys.exit(f"Error loading model: {e}")
        else:
            import sys

            sys.exit(f"Model file not found at {model_file_path}")

    def predict_one(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for a single state.

        Args:
            state: State vector of shape (input_dim,)

        Returns:
            Array of Q-values for each action
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    @property
    def input_dim(self) -> int:
        """Get input dimension of the model."""
        return self._input_dim
