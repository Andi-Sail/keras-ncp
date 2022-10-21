import tensorflow as tf

class ScaledRandomUniform(tf.keras.initializers.RandomUniform):

  def __init__(self, scale_weights, minval=-0.05, maxval=0.05, seed=None):
    super().__init__(minval, maxval, seed)
    self.scale_weights = scale_weights

  def __call__(self, shape, dtype=None, **kwargs):
    orig: tf.Tensor = super().__call__(shape, dtype, **kwargs)
    # if not self.scale_weights.shape == shape: # TODO: for now deactivated - assuming correct broadcast below 
    #   raise ValueError('scaled weights are not the correct shape') # TODO: add excpected and actual
    
    return orig * self.scale_weights