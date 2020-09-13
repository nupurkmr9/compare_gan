# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides ModularGAN for GAN models with penalty loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl import flags
from absl import logging
from compare_gan import test_utils
from compare_gan import utils
from compare_gan.architectures import dcgan
from compare_gan.architectures import infogan
from compare_gan.architectures import resnet30
from compare_gan.architectures import resnet5
from compare_gan.architectures import resnet_biggan
from compare_gan.architectures import resnet_biggan_deep
from compare_gan.architectures import resnet_cifar
from compare_gan.architectures import resnet_stl
from compare_gan.architectures import sndcgan
from compare_gan.architectures import aux_network
import random
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans import penalty_lib
from compare_gan.gans.abstract_gan import AbstractGAN
from compare_gan.tpu import tpu_random
from compare_gan.tpu import tpu_summaries
import gin
import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub


FLAGS = flags.FLAGS


# pylint: disable=not-callable
@gin.configurable(blacklist=["dataset", "parameters", "model_dir"])
class ModularGAN_Aux_Task_AET_v2(AbstractGAN):
  """Base class for GANs models that support the Estimator API."""

  def __init__(self,
               dataset,
               parameters,
               model_dir,
               deprecated_split_disc_calls=False,
               experimental_joint_gen_for_disc=False,
               experimental_force_graph_unroll=False,
               g_use_ema=False,
               ema_decay=0.9999,
               ema_start_step=40000,
               g_optimizer_fn=tf.train.AdamOptimizer,
               d_optimizer_fn=None,
               g_lr=0.0002,
               d_lr=None,
               conditional=False,
               fit_label_distribution=False,
               warm_start_ckpt='',
               d_bs=64,
               g_bs=64,
               aux_ip_size=4,
               aux_ip_channels=512,
               aux_avg_pool_sz=4,
               num_eps=2,
               eps_max=1.0,
               aux_lr=0.001,
               lambda_bce_loss=1.0,
               aet_loss='L2',
               choice_of_f='concat', 
               use_tanh=False,
               which_eps_distr='trunc_normal',
               num_groups=16):
    """ModularGAN  is a Gin configurable implementation of AbstractGAN.

    Graph Unrolling:
    For better performance TPUs perform multiple training steps in a single
    session run call. To utilize this we perform both D and G training in a
    single training step. The inputs to model_fn are split into multiple
    sub-steps:
    One sub-step for each discriminator training step (disc_iters) and a
    separate sub-step (with new inputs) for the generator training step.
    The configured batch size is the batch size used in a sub-step.

    Warning: Graph unrolling can increase the memory requirement and load to
    memory issues on GPUs. Therefore it is turned off when running on GPUs, but
    can be forced to be on with experimental_force_graph_unroll.

    Args:
      dataset: `ImageDataset` object. If `conditional` the dataset must provide
        labels and the number of classes bust known.
      parameters: Legacy Python dictionary with additional parameters. This must
        have the keys 'architecture', 'z_dim' and 'lambda'.
      model_dir: Directory path for storing summary files.
      deprecated_split_disc_calls: If True pass fake and real images separately
        through the discriminator network.
      experimental_joint_gen_for_disc: If True generate fake images for all D
        iterations jointly. This increase the batch size in G when generating
        fake images for D. The G step is stays the same.
      experimental_force_graph_unroll: Force unrolling of the graph as described
        above. When running on TPU the graph is always unrolled.
      g_use_ema: If True keep moving averages for weights in G and use them in
        the TF-Hub module.
      ema_decay: Decay rate for moving averages for G's weights.
      ema_start_step: Start step for keeping moving averages. Before this the
        decay rate is 0.
      g_optimizer_fn: Function (or constructor) to return an optimizer for G.
      d_optimizer_fn: Function (or constructor) to return an optimizer for D.
        If None will call `g_optimizer_fn`.
      g_lr: Learning rate for G.
      d_lr: Learning rate for D. Defaults to `g_lr`.
      conditional: Whether the GAN is conditional. If True both G and Y will
        get passed labels.
      fit_label_distribution: Whether to fit the label distribution.
    """
    super(ModularGAN_Aux_Task_AET_v2, self).__init__(
        dataset=dataset, parameters=parameters, model_dir=model_dir)
    self._deprecated_split_disc_calls = deprecated_split_disc_calls
    self._experimental_joint_gen_for_disc = experimental_joint_gen_for_disc
    self._experimental_force_graph_unroll = experimental_force_graph_unroll
    self._g_use_ema = g_use_ema
    self._ema_decay = ema_decay
    self._ema_start_step = ema_start_step
    self._g_optimizer_fn = g_optimizer_fn
    self._d_optimizer_fn = d_optimizer_fn
    if self._d_optimizer_fn is None:
      self._d_optimizer_fn = g_optimizer_fn
    self._g_lr = g_lr
    self._d_lr = g_lr if d_lr is None else d_lr
    
    if conditional and not self._dataset.num_classes:
      raise ValueError(
          "Option 'conditional' selected but dataset {} does not have "
          "labels".format(self._dataset.name))
    self._conditional = conditional
    self._fit_label_distribution = fit_label_distribution

    self._tpu_summary = tpu_summaries.TpuSummaries(model_dir)

    # Parameters that have not been ported to Gin.
    self._architecture = parameters["architecture"]
    self._z_dim = parameters["z_dim"]
    self._lambda = parameters["lambda"]

    # Number of discriminator iterations per one iteration of the generator.
    self._disc_iters = parameters.get("disc_iters", 1)
    self._force_graph_unroll = parameters.get("force_graph_unroll")

    # Will be set by create_loss().
    self.d_loss = None
    self.g_loss = None
    self.penalty_loss = None
    self.aux_loss = None
    
    # Cache for discriminator and generator objects.
    self._discriminator = None
    self._generator = None
    
    self._warm_start_ckpt = warm_start_ckpt
    
    self._d_bs = d_bs
    self._g_bs = g_bs
    self._aux_ip_size = aux_ip_size
    self._aux_ip_channels = aux_ip_channels
    self._aux_avg_pool_sz = aux_avg_pool_sz
    self._aux_network = None
    self._num_eps = min(g_bs, num_eps)
    self._eps_min = -1 * eps_max
    self._eps_max = eps_max
    self._aux_lr = aux_lr
    self._lambda_bce_loss = lambda_bce_loss
    self._aux_net = parameters["aux_net"]
    self._aet_loss = aet_loss
    self._choice_of_f = choice_of_f
    self._use_tanh = use_tanh
    self._which_eps_distr = which_eps_distr
    self._num_groups = num_groups  

  def _get_num_sub_steps(self, unroll_graph):
    if unroll_graph:
      return self._disc_iters + 1
    return 1

  @property
  def conditional(self):
    return self._conditional

  @property
  def generator(self):
    if self._generator is None:
      architecture_fns = {
          c.DCGAN_ARCH: dcgan.Generator,
          c.DUMMY_ARCH: test_utils.Generator,
          c.INFOGAN_ARCH: infogan.Generator,
          c.RESNET5_ARCH: resnet5.Generator,
          c.RESNET30_ARCH: resnet30.Generator,
          c.RESNET_BIGGAN_ARCH: resnet_biggan.Generator,
          c.RESNET_BIGGAN_DEEP_ARCH: resnet_biggan_deep.Generator,
          c.RESNET_CIFAR_ARCH: resnet_cifar.Generator,
          c.RESNET_STL_ARCH: resnet_stl.Generator,
          c.SNDCGAN_ARCH: sndcgan.Generator,
      }
      if self._architecture not in architecture_fns:
        raise NotImplementedError(
            "Generator architecture {} not implemented.".format(
                self._architecture))
      self._generator = architecture_fns[self._architecture](
          image_shape=self._dataset.image_shape)
    return self._generator

  @property
  def discriminator(self):
    """Returns an instantiation of `AbstractDiscriminator`."""
    if self._discriminator is None:
      architecture_fns = {
          c.DCGAN_ARCH: dcgan.Discriminator,
          c.DUMMY_ARCH: test_utils.Discriminator,
          c.INFOGAN_ARCH: infogan.Discriminator,
          c.RESNET5_ARCH: resnet5.Discriminator,
          c.RESNET30_ARCH: resnet30.Discriminator,
          c.RESNET_BIGGAN_ARCH: resnet_biggan.Discriminator,
          c.RESNET_BIGGAN_DEEP_ARCH: resnet_biggan_deep.Discriminator,
          c.RESNET_CIFAR_ARCH: resnet_cifar.Discriminator,
          c.RESNET_STL_ARCH: resnet_stl.Discriminator,
          c.SNDCGAN_ARCH: sndcgan.Discriminator,
      }
      if self._architecture not in architecture_fns:
        raise NotImplementedError(
            "Discriminator architecture {} not implemented.".format(
                self._architecture))
      self._discriminator = architecture_fns[self._architecture]()
    return self._discriminator

  @property
  def aux_network(self):
    if self._aux_network is None:
      if self._aux_net == 'aux_network_aet_v2':
        self._aux_network = aux_network.Aux_Network_AET_v2(aux_ip_size=self._aux_ip_size, 
                                                     aux_ip_channels=self._aux_ip_channels, num_groups=self._num_groups)
        return self._aux_network
      architecture_fns = {
          # "aux_network_v1": aux_network.Aux_Network_v1,
          # "aux_network_v2": aux_network.Aux_Network_v2,
          "aux_network_aet_v1": aux_network.Aux_Network_AET_v1,
      }
      self._aux_network = architecture_fns[self._aux_net](aux_ip_size=self._aux_ip_size, 
                                                     aux_ip_channels=self._aux_ip_channels)
    return self._aux_network


  def as_estimator(self, run_config, batch_size, use_tpu):
    """Returns a TPUEstimator for this GAN."""
    unroll_graph = self._experimental_force_graph_unroll or use_tpu
    num_sub_steps = self._get_num_sub_steps(unroll_graph=unroll_graph)
    return tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=use_tpu,
        model_fn=self.model_fn,
        train_batch_size=batch_size * num_sub_steps)

  def _module_fn(self, model, batch_size):
    """Module Function to create a TF Hub module spec.

    Args:
      model: `tf.estimator.ModeKeys` value.
      batch_size: batch size.
    """
    if model not in {"gen", "disc"}:
      raise ValueError("Model {} not support in module_fn()".format(model))
    placeholder_fn = tf.placeholder if batch_size is None else tf.zeros
    is_training = False
    inputs = {}
    y = None
    if model == "gen":
      inputs["z"] = placeholder_fn(
          shape=(batch_size, self._z_dim),
          dtype=tf.float32,
          name="z_for_eval")
    elif model == "disc":
      inputs["images"] = placeholder_fn(
          shape=[batch_size] + list(self._dataset.image_shape),
          dtype=tf.float32,
          name="images_for_eval")
    if self.conditional:
      inputs["labels"] = placeholder_fn(
          shape=(batch_size,),
          dtype=tf.int32,
          name="labels_for_eval")
      y = self._get_one_hot_labels(inputs["labels"])
    else:
      y = None

    logging.info("Creating module for model %s with inputs %s and y=%s",
                 model, inputs, y)
    outputs = {}
    if model == "disc":
      outputs["prediction"], _, _ = self.discriminator(
          inputs["images"], y=y, is_training=is_training)
    else:
      z = inputs["z"]
      generated = self.generator(z=z, y=y, is_training=is_training)
      if self._g_use_ema and not is_training:
        g_vars = [var for var in tf.trainable_variables()
                  if "generator" in var.name]
        ema = tf.train.ExponentialMovingAverage(decay=self._ema_decay)
        # Create the variables that will be loaded from the checkpoint.
        ema.apply(g_vars)
        def ema_getter(getter, name, *args, **kwargs):
          var = getter(name, *args, **kwargs)
          ema_var = ema.average(var)
          if ema_var is None:
            var_names_without_ema = {"u_var", "accu_mean", "accu_variance",
                                     "accu_counter", "update_accus"}
            if name.split("/")[-1] not in var_names_without_ema:
              logging.warning("Could not find EMA variable for %s.", name)
            return var
          return ema_var
        with tf.variable_scope("", values=[z, y], reuse=True,
                               custom_getter=ema_getter):
          generated = self.generator(z, y=y, is_training=is_training)
      outputs["generated"] = generated

    hub.add_signature(inputs=inputs, outputs=outputs)

  def as_module_spec(self):
    """Returns the generator network as TFHub module spec."""
    models = ["gen", "disc"]
    default_batch_size = 64
    batch_sizes = [8, 16, 32, 64]
    if "resnet" in self._architecture:
      # Only ResNet architectures support dynamic batch size.
      batch_sizes.append(None)
      default_batch_size = None
    tags_and_args = [
        (set(), {"model": "gen", "batch_size": default_batch_size})]
    for model, bs in itertools.product(models, batch_sizes):
      tags = {model, "bs{}".format(bs)}
      args = {"model": model, "batch_size": bs}
      tags_and_args.append((tags, args))
    return hub.create_module_spec(
        self._module_fn, tags_and_args=tags_and_args,
        drop_collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES])

  def _grid_shape(self, num_summary_images):
    """Returns the shape for a rectangle grid with `num_summarry_images`."""
    if num_summary_images & (num_summary_images - 1) != 0:
      raise ValueError(
          "Number of summary images must be a power of 2 to create a grid of "
          "images but was {}.".format(num_summary_images))
    # Since b = 2^c we can use x = 2^(floor(c/2)) and y = 2^(ceil(c/2)).
    x = 2 ** int(np.log2(num_summary_images) / 2)
    y = num_summary_images // x
    return x, y

  def _add_images_to_summary(self, images, summary_name, params):
    """Called from model_fn() to add a grid of images as summary."""
    # All summary tensors are synced to host 0 on every step. To avoid sending
    # more images then needed we transfer at most `sampler_per_replica` to
    # create a 8x8 image grid.
    batch_size_per_replica = images.shape[0].value
    num_replicas = params["context"].num_replicas if "context" in params else 1
    total_num_images = batch_size_per_replica * num_replicas
    if total_num_images >= 64:
      grid_shape = (8, 8)
      # This can be more than 64. We slice all_images below.
      samples_per_replica = int(np.ceil(64 / num_replicas))
    else:
      grid_shape = self._grid_shape(total_num_images)
      samples_per_replica = batch_size_per_replica
    def _merge_images_to_grid(all_images):
      logging.info("Creating images summary for fake images: %s", all_images)
      return tfgan.eval.image_grid(
          all_images[:np.prod(grid_shape)],
          grid_shape=grid_shape,
          image_shape=self._dataset.image_shape[:2],
          num_channels=self._dataset.image_shape[2])
    self._tpu_summary.image(summary_name,
                            images[:samples_per_replica],
                            reduce_fn=_merge_images_to_grid)

  def _check_variables(self):
    """Check that every variable belongs to either G or D."""
    t_vars = tf.trainable_variables()
    g_vars = self.generator.trainable_variables
    d_vars = self.discriminator.trainable_variables
    aux_vars = self.aux_network.trainable_variables
    shared_vars = (set(d_vars) & set(g_vars)) | (set(d_vars) & set(aux_vars)) | (set(g_vars) & set(aux_vars))
    if shared_vars:
      logging.info("g_vars: %s", g_vars)
      logging.info("d_vars: %s", d_vars)
      logging.info("aux_vars: %s", aux_vars)
      raise ValueError("Shared trainable variables: %s" % shared_vars)
    unused_vars = set(t_vars) - set(d_vars) - set(g_vars) - set(aux_vars)
    if unused_vars:
      raise ValueError("Unused trainable variables: %s" % unused_vars)

  def _get_one_hot_labels(self, labels):
    if not self.conditional:
      raise ValueError(
          "_get_one_hot_labels() called but GAN is not conditional.")
    return tf.one_hot(labels, self._dataset.num_classes)

  @gin.configurable("z", blacklist=["shape", "name"])
  def z_generator(self, shape, distribution_fn=tf.random.uniform,
                  minval=-1.0, maxval=1.0, stddev=1.0, name=None):
    """Random noise distributions as TF op.

    Args:
      shape: A 1-D integer Tensor or Python array.
      distribution_fn: Function that create a Tensor. If the function has any
        of the arguments 'minval', 'maxval' or 'stddev' these are passed to it.
      minval: The lower bound on the range of random values to generate.
      maxval: The upper bound on the range of random values to generate.
      stddev: The standard deviation of a normal distribution.
      name: A name for the operation.

    Returns:
      Tensor with the given shape and dtype tf.float32.
    """
    return utils.call_with_accepted_args(
        distribution_fn, shape=shape, minval=minval, maxval=maxval,
        stddev=stddev, name=name)

  def label_generator(self, shape, name=None):
    if not self.conditional:
      raise ValueError("label_generator() called but GAN is not conditional.")
    # Assume uniform label distribution.
    return tf.random.uniform(shape, minval=0, maxval=self._dataset.num_classes,
                             dtype=tf.int32, name=name)

  def _preprocess_fn(self, images, labels, seed=None):
    """Creates the feature dictionary with images and z."""
    logging.info("_preprocess_fn(): images=%s, labels=%s, seed=%s",
                 images, labels, seed)
    tf.set_random_seed(seed)
    features = {
        "images": images,
        "z": self.z_generator([self._z_dim], name="z"),
    }
    if self.conditional:
      if self._fit_label_distribution:
        features["sampled_labels"] = labels
      else:
        features["sampled_labels"] = self.label_generator(
            shape=[], name="sampled_labels")
    return features, labels

  def input_fn(self, params, mode):
    """Input function that retuns a `tf.data.Dataset` object.

    This function will be called once for each host machine.

    Args:
      params: Python dictionary with parameters given to TPUEstimator.
          Additional TPUEstimator will set the key `batch_size` with the batch
          size for this host machine and `tpu_contextu` with a TPUContext
          object.
      mode: `tf.estimator.MoedeKeys` value.

    Returns:
      A `tf.data.Dataset` object with batched features and labels.
    """
    return self._dataset.input_fn(mode=mode, params=params,
                                  preprocess_fn=self._preprocess_fn)

  def _split_inputs_and_generate_samples(self, features, labels, num_sub_steps):
    # Encode labels.
    if self.conditional:
      assert "sampled_labels" in features
      features["sampled_y"] = self._get_one_hot_labels(
          features["sampled_labels"])

    eps_bs = features["z"].get_shape().as_list()[0]
    half_bs = eps_bs // (2*num_sub_steps)

    logging.info("$$$$$$$$$$$$ sample inputs are x=%s", features["sampled_y"].shape)
    logging.info("$$$$$$$$$$$$ sample inputs are x=%s", features["z"].shape)
    logging.info("$$$$$$$$$$$$ sample inputs are x=%s", features["images"].shape)

    
    features["random"] = tf.random.uniform(shape=[eps_bs, self._num_groups], minval=0.0, maxval=1.0, name="random")
    features["random_mask"] = tf.less(features["random"], 0.5, name="random_mask")
    features["eps_original"] = tf.where(features["random_mask"], tf.constant(self._eps_max, shape=[eps_bs, self._num_groups]), tf.constant(self._eps_min, shape=[eps_bs, self._num_groups]), name="eps_original")
    features["eps_original"] = tf.reshape(features["eps_original"], [eps_bs, -1, 1], name="reshape_eps_1")
    features["eps_original"] = tf.tile(features["eps_original"], multiples=[1, 1, self._z_dim // self._num_groups], name="eps_group_tile")
    features["eps_original"] = tf.reshape(features["eps_original"], [eps_bs, -1], name="reshape_eps_2")

    features["eps"] = tf.tile(features["eps_original"], multiples=[self._g_bs // self._num_eps, 1], name="eps")

    logging.info("$$$$$$$$$$$$ sample inputs are x=%s", features["eps"].shape)
    logging.info("$$$$$$$$$$$$ sample inputs are x=%s", features["random_mask"].shape)
    # Split i

    # Split inputs for sub-steps.
    fs = [(k, tf.split(features[k], num_sub_steps)) for k in features]
    fs = [{k: v[i] for k, v in fs} for i in range(num_sub_steps)]
    ls = tf.split(labels, num_sub_steps)

    total_batch_size = features["z"].shape[0].value
    assert total_batch_size % num_sub_steps == 0
    batch_size = total_batch_size // num_sub_steps

    if self._experimental_joint_gen_for_disc:
      # Generate samples from G for D steps.
      with tf.name_scope("gen_for_disc"):
        # Only the last sub-step changes the generator weights. Thus we can
        # combine all forward passes through G to achieve better efficiency.
        # The forward pass for G's step needs to be separated since compute
        # gradients for it.
        z = features["z"][:batch_size * self._disc_iters]
        sampled_y = None
        if self.conditional:
          sampled_y = features["sampled_y"][:batch_size * self._disc_iters]
        generated = self.generator(z, y=sampled_y, is_training=True)
        generated = tf.split(generated, self._disc_iters)
        for i in range(self._disc_iters):
          fs[i]["generated"] = generated[i]
      # Generate samples from G for G step.
      with tf.name_scope("gen_for_gen"):
        sampled_y = fs[-1].get("sampled_y", None)
        fs[-1]["generated"] = self.generator(fs[-1]["z"], y=sampled_y, is_training=True)
        fs[-1]["eps_generated"] = self.generator(tf.add(fs[-1]["z"], fs[-1]["eps"]), y=sampled_y, is_training=True)
    else:
      for f in fs:
        sampled_y = f.get("sampled_y", None)
        f["generated"] = self.generator(f["z"], y=sampled_y, is_training=True)
        f["eps_generated"] = self.generator(tf.add(f["z"], f["eps"]), y=sampled_y, is_training=True)

    return fs, ls

  def _train_discriminator(self, features, labels, step, optimizer, params):
    features = features.copy()
    features["generated"] = tf.stop_gradient(features["generated"])
    features["eps_generated"] = tf.stop_gradient(features["eps_generated"])

    # Set the random offset tensor for operations in tpu_random.py.
    tpu_random.set_random_offset_from_features(features)
    # create_loss will set self.d_loss.
    self.create_loss(features, labels, params=params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          self.d_loss,
          var_list=self.discriminator.trainable_variables,
          global_step=step)
      with tf.control_dependencies([train_op]):
        return tf.identity(self.d_loss)

  def _train_generator(self, features, labels, step, optimizer, aux_optimizer, params):
    # Set the random offset tensor for operations in tpu_random.py.
    tpu_random.set_random_offset_from_features(features)
    # create_loss will set self.g_loss.
    self.create_loss(features, labels, params=params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      aux_op = aux_optimizer.minimize(
          self.aux_loss,
          var_list=self.aux_network.trainable_variables)
      with tf.control_dependencies([aux_op]):
        train_op = optimizer.minimize(
            self.g_loss,
            var_list=self.generator.trainable_variables,
            global_step=step)
        if self._g_use_ema:
          g_vars = self.generator.trainable_variables
          with tf.name_scope("generator_ema"):
            logging.info("Creating moving averages of weights: %s", g_vars)
            # The decay value is set to 0 if we're before the moving-average start
            # point, so that the EMA vars will be the normal vars.
            decay = self._ema_decay * tf.cast(
                tf.greater_equal(step, self._ema_start_step), tf.float32)
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            with tf.control_dependencies([train_op]):
              train_op = ema.apply(g_vars)
        with tf.control_dependencies([train_op]):
          return tf.identity(self.g_loss)

  def model_fn(self, features, labels, params, mode):
    """Constructs the model for the given features and mode.

    Args:
      features: A dictionary with the feature tensors.
      labels: Tensor will labels. Will be None if mode is PREDICT.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      mode: `tf.estimator.ModeKeys` value (TRAIN, EVAL, PREDICT). The mode
          should be passed to the TPUEstimatorSpec and your model should be
          build this mode.

    Returns:
      A `tf.contrib.tpu.TPUEstimatorSpec`.
    """
    logging.info("model_fn(): features=%s, labels=%s,mode=%s, params=%s",
                 features, labels, mode, params)
    if mode != tf.estimator.ModeKeys.TRAIN:
      raise ValueError("Only training mode is supported.")

    use_tpu = params["use_tpu"]
    unroll_graph = self._experimental_force_graph_unroll or use_tpu
    num_sub_steps = self._get_num_sub_steps(unroll_graph=unroll_graph)
    if unroll_graph:
      logging.warning("Graph will be unrolled.")
    if self._experimental_joint_gen_for_disc and not unroll_graph:
      raise ValueError("Joining G forward passes is only supported for ",
                       "unrolled graphs.")

    # Clean old summaries from previous calls to model_fn().
    self._tpu_summary = tpu_summaries.TpuSummaries(self._model_dir)

    # Get features for each sub-step.
    fs, ls = self._split_inputs_and_generate_samples(
        features, labels, num_sub_steps=num_sub_steps)

    disc_optimizer = self.get_disc_optimizer(params["use_tpu"])
    disc_step = tf.get_variable(
        "global_step_disc", [], dtype=tf.int32, trainable=False)
    train_disc_fn = functools.partial(
        self._train_discriminator,
        step=disc_step,
        optimizer=disc_optimizer,
        params=params)

    gen_optimizer = self.get_gen_optimizer(params["use_tpu"])
    gen_step = tf.train.get_or_create_global_step()
    train_gen_fn = functools.partial(
        self._train_generator,
        features=fs[-1],
        labels=ls[-1],
        step=gen_step,
        optimizer=gen_optimizer,
        aux_optimizer=aux_optimizer,
        params=params)

    if not unroll_graph and self._disc_iters != 1:
      train_fn = train_gen_fn
      train_gen_fn = lambda: tf.cond(
          tf.equal(disc_step % self._disc_iters, 0), train_fn, lambda: 0.0)

    # Train D.
    d_losses = []
    d_steps = self._disc_iters if unroll_graph else 1
    for i in range(d_steps):
      with tf.name_scope("disc_step_{}".format(i + 1)):
        with tf.control_dependencies(d_losses):
          d_losses.append(train_disc_fn(features=fs[i], labels=ls[i]))

    # Train G.
    with tf.control_dependencies(d_losses):
      with tf.name_scope("gen_step"):
        g_loss = train_gen_fn()

    for i, d_loss in enumerate(d_losses):
      self._tpu_summary.scalar("loss/d_{}".format(i), d_loss)
    self._tpu_summary.scalar("loss/g", g_loss)
    # self._add_images_to_summary(fs[0]["generated"], "fake_images", params)
    # self._add_images_to_summary(fs[0]["images"], "real_images", params)

    self._check_variables()
    utils.log_parameter_overview(self.generator.trainable_variables,
                                 msg="Generator variables:")
    utils.log_parameter_overview(self.discriminator.trainable_variables,
                                 msg="Discriminator variables:")
    utils.log_parameter_overview(self.aux_network.trainable_variables,
                                 msg="Aux Network variables:")

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        host_call=self._tpu_summary.get_host_call(),
        # Estimator requires a loss which gets displayed on TensorBoard.
        # The given Tensor is evaluated but not used to create gradients.
        loss=d_losses[0],
        train_op=g_loss.op)

  def get_disc_optimizer(self, use_tpu=True):
    opt = self._d_optimizer_fn(self._d_lr, name="d_opt")
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt

  def get_gen_optimizer(self, use_tpu=True):
    opt = self._g_optimizer_fn(self._g_lr, name="g_opt")
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt

  def get_aux_optimizer(self, use_tpu=True):
    opt = self._g_optimizer_fn(self._aux_lr, name="aux_opt")
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt


  def create_loss(self, features, labels, params, is_training=True):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Real images.
    generated = features["generated"]  # Fake images.
    eps_generated = features["eps_generated"] #fake eps 

    eps_bs = generated.get_shape().as_list()[0]
    half_bs = eps_bs // 2

    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
      all_y = tf.concat([y, sampled_y,sampled_y], axis=0)
    else:
      y = None
      sampled_y = None
      all_y = None

    if self._deprecated_split_disc_calls:
      with tf.name_scope("disc_for_real"):
        d_real, d_real_logits, _ = self.discriminator(
            images, y=y, is_training=is_training)
      with tf.name_scope("disc_for_fake"):
        d_fake, d_fake_logits, D_in_op_g = self.discriminator(
            generated, y=sampled_y, is_training=is_training)
      with tf.name_scope("disc_eps_for_fake"):
        d_fake_eps, d_fake_logits_eps, D_in_op_eg  = self.discriminator(
            eps_generated, y=sampled_y, is_training=is_training)
    else:
      # Compute discriminator output for real and fake images in one batch.
      all_images = tf.concat([images, generated , eps_generated], axis=0)
      d_all, d_all_logits, _ = self.discriminator(
          all_images, y=all_y, is_training=is_training)
      d_real, d_fake, d_fake_eps= tf.split(d_all, 3)
      d_real_logits, d_fake_logits, d_fake_logits_eps = tf.split(d_all_logits, 3)

     
    self.d_loss, _, _, _ = loss_lib.get_losses(
        d_real=d_real, d_fake=tf.concat([d_fake[:half_bs],d_fake_eps[:half_bs]] , axis = 0), d_real_logits=d_real_logits,
        d_fake_logits=tf.concat([d_fake_logits[:half_bs] , d_fake_logits_eps[:half_bs]] , axis=0) )

    _, _, _, g_loss = loss_lib.get_losses(
        d_real=None, d_fake=d_fake, d_real_logits=None,
        d_fake_logits=d_fake_logits)
    
    _, _, _, g_loss_eps = loss_lib.get_losses(
        d_real=None, d_fake=d_fake_eps, d_real_logits=None,
        d_fake_logits=d_fake_logits_eps)

    self.g_loss = g_loss + g_loss_eps

    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, y=y, is_training=is_training,
        discriminator=self.discriminator)
    self.d_loss += self._lambda * penalty_loss



    D_in_op_g = tf.reshape(D_in_op_g, [eps_bs , -1], name="reshape_g")
    D_in_op_eg = tf.reshape(D_in_op_eg, [eps_bs , -1], name="reshape_eg")
    
    if self._choice_of_f == 'subtract':
        f = tf.subtract(D_in_op_g, D_in_op_eg)
    elif self._choice_of_f == 'concat':
        f = tf.concat([D_in_op_g, D_in_op_eg], axis=1)
    
    pred_eps = self.aux_network(f)

    features["labels_aux"] = tf.cast(features["random_mask"], tf.float32, name="labels")
    self.aux_loss = tf.losses.sigmoid_cross_entropy(features["labels_aux"],pred_eps) # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_eps, labels=features["labels"], name="aux_cross_entropy"))
    self.aux_acc = tf.reduce_mean(tf.cast(tf.equal(features["labels_aux"], tf.round(tf.sigmoid(pred_eps))), dtype=tf.float32))
    self.g_loss += self._lambda_bce_loss * self.aux_loss
