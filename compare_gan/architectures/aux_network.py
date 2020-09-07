"""Implementation of auxiliary network."""

from compare_gan.architectures.abstract_arch import _Module
from compare_gan.architectures.arch_ops import linear

import tensorflow as tf

# Non-Linear 2 layer MLP Network
class Aux_Network_v1(_Module):
  """Aux Network architecture"""

  def __init__(self, name="aux_network_v1", aux_ip_size=4, aux_ip_channels=512):
    super(Aux_Network_v1, self).__init__(name=name)
    self._aux_ip_size = aux_ip_size
    self._aux_ip_channels = aux_ip_channels
    
  def apply(self, x):
    # x will be of shape [batch_size, 2 * aux_ip_size * aux_ip_size * aux_ip_channels]
    net = linear(x, self._aux_ip_channels, scope="aux_fc1")
    net = tf.nn.relu(net)
    net = linear(net, 1, scope="aux_fc2")
    return net
    
  def __call__(self, x, reuse=tf.AUTO_REUSE):
    # tf.AUTO_REUSE: https://stackoverflow.com/a/47641978 .
    with tf.variable_scope(self.name, values=[x], reuse=reuse):
      output = self.apply(x)
    return output

# Linear Network
class Aux_Network_v2(_Module):
  """Aux Network architecture"""

  def __init__(self, name="aux_network_v2", aux_ip_size=4, aux_ip_channels=512):
    super(Aux_Network_v2, self).__init__(name=name)
    self._aux_ip_size = aux_ip_size
    self._aux_ip_channels = aux_ip_channels
    
  def apply(self, x):
    # x will be of shape [batch_size, 2 * aux_ip_size * aux_ip_size * aux_ip_channels]
    net = linear(x, 1, scope="aux_fc")
    return net
    
  def __call__(self, x, reuse=tf.AUTO_REUSE):
    # tf.AUTO_REUSE: https://stackoverflow.com/a/47641978 .
    with tf.variable_scope(self.name, values=[x], reuse=reuse):
      output = self.apply(x)
    return output

# Non-Linear 2 layer MLP Network
class Aux_Network_AET_v1(_Module):
  """Aux Network architecture"""

  def __init__(self, name="aux_network_aet_v1", aux_ip_size=4, aux_ip_channels=512, z_dim=128):
    super(Aux_Network_AET_v1, self).__init__(name=name)
    self._aux_ip_size = aux_ip_size
    self._aux_ip_channels = aux_ip_channels
    self._z_dim = z_dim
    
  def apply(self, x):
    # x will be of shape [batch_size, 2 * aux_ip_size * aux_ip_size * aux_ip_channels]
    net = linear(x, self._aux_ip_channels, scope="aux_fc1")
    net = tf.nn.relu(net)
    net = linear(net, self._z_dim, scope="aux_fc2")
    return net
    
  def __call__(self, x, reuse=tf.AUTO_REUSE):
    # tf.AUTO_REUSE: https://stackoverflow.com/a/47641978 .
    with tf.variable_scope(self.name, values=[x], reuse=reuse):
      output = self.apply(x)
    return output

# Non-Linear 2 layer MLP Network
class Aux_Network_AET_v2(_Module):
  """Aux Network architecture"""

  def __init__(self, name="aux_network_aet_v2", aux_ip_size=4, aux_ip_channels=512, num_groups=16):
    super(Aux_Network_AET_v2, self).__init__(name=name)
    self._aux_ip_size = aux_ip_size
    self._aux_ip_channels = aux_ip_channels
    self._num_groups = num_groups
    
  def apply(self, x):
    # x will be of shape [batch_size, 2 * aux_ip_size * aux_ip_size * aux_ip_channels]
    net = linear(x, self._aux_ip_channels, scope="aux_fc1")
    net = tf.nn.relu(net)
    net = linear(net, self._num_groups, scope="aux_fc2")
    return net
    
  def __call__(self, x, reuse=tf.AUTO_REUSE):
    # tf.AUTO_REUSE: https://stackoverflow.com/a/47641978 .
    with tf.variable_scope(self.name, values=[x], reuse=reuse):
      output = self.apply(x)
    return output
