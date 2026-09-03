import jax
import jax.numpy as jnp
from typing import Callable, Iterable, Tuple

def construct_feature_matrix(t: jnp.array, omega: float) -> jnp.array:
    raise NotImplementedError("construct_feature_matrix has yet to be implemented")


class NNModel():
  def __init__(self, F_mars: jax.Array, num_hidden:int=8):
    """
    Parameters:
      F_mars (jax.Array): the feature matrix of dimension [num_time, num_features]

    Attributes:
      Phi_mars (jax.Array): the scaled feature matrix for improved
        numerical conditioning of size [num_time, num_features]
      num_input (int): the input size to the neural network (num_features + 1 for a bias term)
      num_hidden (int): the number of hidden features in the neural network
      num_parameters (int): total number of trainable parameters for the NN

    """
    self.num_hidden = num_hidden
    self.num_inputs = F_mars.shape[1] + 1
    self.num_parameters = self.num_inputs * self.num_hidden + self.num_hidden

    self.Phi_mars = jnp.concatenate([F_mars, jnp.ones((F_mars.shape[0], 1))], axis=1)

    return

  def unpack_nn_params(self, theta) -> Tuple[jax.Array, jax.Array]:
    """
    Parameters:
      theta (jax.Array): estimate of NN parameters used for forward pass

    Outputs:
      W1 (jax.Array): weights of first layer
      W2 (jax.Array): weights of second layer 

    """
    num_w1_params = self.num_inputs * self.num_hidden
    W1 = theta[:num_w1_params].reshape(self.num_inputs, self.num_hidden)
    W2 = theta[num_w1_params:].reshape(self.num_hidden, 1)
    return W1, W2

  def forward(self, theta: jax.Array) -> jax.Array:
    """
    Parameters:
      theta (jax.Array): estimate of NN parameters used for forward pass

    Outputs:
      nn_out (jax.Array): neural network output 

    """
    W1, W2 = self.unpack_nn_params(theta)

    # Compute 'forward pass' of NN
    hidden = jax.nn.relu(self.Phi_mars @ W1)
    nn_out = (hidden @ W2)[:, 0]
    return nn_out