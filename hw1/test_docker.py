try:
  import jax
  import casadi
  import numpy as np
  import scipy as sp
  import marimo
  print('Docker is successfully installed!')
except:
  print('Hmm, the Docker does not seem to have successfully been setup, try again')
