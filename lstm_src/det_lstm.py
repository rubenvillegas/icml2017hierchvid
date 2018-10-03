import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from ops import *


class DET_LSTM(object):
  def __init__(self,
               batch_size,
               input_size,
               layers,
               seen_step,
               fut_step,
               keep_prob,
               logs_dir,
               learning_rate,
               mode='train'):

    self.input_size = input_size
    self.point_size = input_size / 2
    self.batch_size = batch_size
    self.seen_step = seen_step
    self.fut_step = fut_step
    if mode == 'train':
      self.seq_len = seen_step + fut_step
    else:
      self.seq_len = seen_step
    self.enc_units = layers[0]
    self.keep_prob = keep_prob
    self.learning_rate = learning_rate

    self.seq_ = tf.placeholder(
        tf.float32, shape=[batch_size, self.seq_len, input_size], name='seq')
    self.mask_ = tf.placeholder(
        tf.float32,
        shape=[batch_size, self.seq_len, self.point_size],
        name='mask')

    stacked_lstm = self.lstm_model(layers)

    mask = tf.concat([self.mask_, self.mask_], 2)
    masked_seq = mask * self.seq_

    act_emb = None
    input_list = []
    input_list_enc = []
    for t in range(self.seen_step):
      input_list.append(masked_seq[:, t, :])
      input_list_enc.append(relu(linear(
          masked_seq[:, t, :], 32, name='lm_enc', reuse=tf.AUTO_REUSE)))

    with tf.variable_scope('GEN'):
      with tf.variable_scope('G_LSTM'):
        enc_out, states = tf.contrib.rnn.static_rnn(
            stacked_lstm, input_list_enc, dtype=dtypes.float32)

    reuse_lstm = True
    reuse_output = False
    output_list = input_list
    empty_input = tf.zeros_like(input_list_enc[-1])

    with tf.variable_scope('GEN'):
      for t in range(fut_step):
        with tf.variable_scope('G_LSTM', reuse=reuse_lstm):
          enc_out, states = tf.contrib.rnn.static_rnn(
              stacked_lstm, [empty_input],
              initial_state=states,
              dtype=dtypes.float32)
        with tf.variable_scope('G_LSTM', reuse=reuse_output):
          output = self.decoder(enc_out[-1])
        output_list.append(output)
        reuse_output = True
      self.output = tf.stack(output_list, 1)

    if mode == 'train':
      self.recons_loss = tf.reduce_mean(
          mask[:, seen_step:, :] *
          (self.output[:, seen_step:, :] - self.seq_[:, seen_step:, :])**2)

      loss_sum = tf.summary.scalar("loss", self.recons_loss)
      self.g_sum = tf.summary.merge([loss_sum])
      self.writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

      self.g_vars = tf.trainable_variables()

      self.global_step = tf.Variable(0, trainable=False)
      optimizer = tf.train.RMSPropOptimizer(
          self.learning_rate, name='optimizer')
      gradients, g = zip(
          *optimizer.compute_gradients(self.recons_loss, var_list=self.g_vars))

      gradients, _ = tf.clip_by_global_norm(gradients, 25)

      self.optimizer = optimizer.apply_gradients(
          zip(gradients, g), global_step=self.global_step)

      num_param = 0
      for var in self.g_vars:
        num_param += int(np.prod(var.get_shape()))
      print('NUMBER OF PARAMETERS: ' + str(num_param))
    self.saver = tf.train.Saver()

  def decoder(self, input_, reuse=False, name='decoder'):
    out = linear(input_, self.point_size * 2, name='dec_fc2')
    return tanh(out)

  def lstm_model(self, layers):
    lstm_cells = [
        tf.nn.rnn_cell.BasicLSTMCell(units, state_is_tuple=True)
        for units in layers
    ]
    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        for cell in lstm_cells
    ]
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
    return stacked_lstm

  def train(self, sess, batches, mask, step, save_logs=False):
    feed_dict = dict()
    feed_dict[self.seq_] = batches
    feed_dict[self.mask_] = mask
    if save_logs:
      _, summary = sess.run([self.optimizer, self.g_sum], feed_dict=feed_dict)
      self.writer.add_summary(summary, step)
    else:
      _ = sess.run([self.optimizer, self.recons_loss], feed_dict=feed_dict)

    errG = self.recons_loss.eval(feed_dict=feed_dict)

    self.global_step = self.global_step + 1
    return errG

  def predict(self, sess, seq_, mask_):
    feed_dict = dict()
    feed_dict[self.seq_] = seq_
    feed_dict[self.mask_] = mask_
    output = self.output.eval(feed_dict=feed_dict)
    return output

  def save(self, sess, checkpoint_dir, step):
    model_name = "DET_LSTM.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(
        sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, sess, checkpoint_dir, model_name=None):
    print("[*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if model_name is None: model_name = ckpt_name
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: "+str(model_name))
      return True, model_name
    else:
      return False, None

