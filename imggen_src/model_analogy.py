import os
import time
from glob import glob
import tensorflow as tf

from alexnet import alexnet
from hg_stacked import hg_forward
from ops import *
from utils import *


class IMGGEN(object):
  def __init__(self,
               image_size=128,
               batch_size=32,
               c_dim=3,
               layer=0,
               alpha=1.,
               beta=1.,
               n_joints=48,
               is_train=True,
               checkpoint_dir=None):

    self.batch_size = batch_size
    self.image_size = image_size
    self.c_dim = c_dim
    self.n_joints = n_joints

    self.layer = layer
    self.alpha = alpha
    self.beta = beta
    self.gf_dim = 64
    self.df_dim = 64
    self.is_train = is_train

    self.img_shape = [image_size, image_size, self.c_dim]
    self.pose_shape = [image_size, image_size, n_joints]
    self.checkpoint_dir = checkpoint_dir

    self.build_model()

  def build_model(self):
    self.xt_ = tf.placeholder(
        tf.float32, [self.batch_size] + self.img_shape, name='xt')
    self.xtpn_ = tf.placeholder(
        tf.float32, [self.batch_size] + self.img_shape, name='xtpn')
    self.pt_ = tf.placeholder(
        tf.float32, [self.batch_size] + self.pose_shape, name='pt')
    self.ptpn_ = tf.placeholder(
        tf.float32, [self.batch_size] + self.pose_shape, name='ptpn')

    with tf.variable_scope('GEN'):
      xtpn = self.generator(self.xt_, self.pt_, self.ptpn_)
    self.G = tf.reshape(
        xtpn,
        [self.batch_size, self.image_size, self.image_size, 1, self.c_dim])

    if self.is_train:
      true_sim = inverse_transform(self.xtpn_)
      gen_sim = inverse_transform(xtpn)
      h_sim1_t = alexnet(true_sim * 255. - tf.reduce_mean(true_sim * 255.))
      h_sim1_g = alexnet(gen_sim * 255. - tf.reduce_mean(gen_sim * 255.))
      h_sim2_t = hg_forward(true_sim)
      h_sim2_g = hg_forward(gen_sim)

      good_data = tf.concat(axis=3, values=[self.xtpn_, self.ptpn_])
      bad_data = tf.concat(axis=3, values=[self.xt_, self.ptpn_])
      gen_data = tf.concat(axis=3, values=[xtpn, self.ptpn_])

      with tf.variable_scope('DIS'):
        self.D, self.D_logits = self.discriminator(good_data)
      with tf.variable_scope('DIS', reuse=True):
        self.DB, self.DB_logits = self.discriminator(bad_data)
        self.D_, self.D_logits_ = self.discriminator(gen_data)

      self.feat_loss1 = 0.5 * tf.reduce_mean(tf.square(h_sim1_t - h_sim1_g))
      self.feat_loss2 = 0.5 * tf.reduce_mean(
          tf.square(h_sim2_t[self.layer] - h_sim2_g[self.layer]))
      self.L_feat = self.alpha * self.feat_loss1 + self.beta * self.feat_loss2
      self.L_img = 0.5 * tf.reduce_mean(tf.square(xtpn - self.xtpn_))
      self.L = self.L_img + self.L_feat

      self.d_loss_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits, labels=tf.ones_like(self.D)))

      self.d_loss_fake1 = 0.5 * tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

      self.d_loss_fake2 = 0.5 * tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.DB_logits, labels=tf.zeros_like(self.DB)))
      self.d_loss_fake = self.d_loss_fake1 + self.d_loss_fake2
      self.d_loss = self.d_loss_real + self.d_loss_fake
      self.g_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits_, labels=tf.ones_like(self.D_)))

      self.L_sum = tf.summary.scalar('L', self.L)
      self.L_img_sum = tf.summary.scalar('L_img', self.L_img)
      self.L_feat_sum = tf.summary.scalar('L_feat', self.L_feat)
      self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
      self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
      self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
      self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)

      self.t_vars = tf.trainable_variables()
      self.g_vars = [var for var in self.t_vars if 'GEN' in var.name]
      self.dis_vars = [var for var in self.t_vars if 'DIS' in var.name]
      num_param = 0.0
      for var in self.g_vars:
        num_param += int(np.prod(var.get_shape()))
      print('NUMBER OF PARAMETERS: ' + str(num_param))
    self.saver = tf.train.Saver()

  def generator(self, xt, pt, ptpn, reuse=False):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    h_img = self.img_encoder(xt)
    h_pose = self.pose_encoder(tf.concat(axis=0, values=[pt, ptpn]))
    h_pose_t = h_pose[:self.batch_size, :, :, :]
    h_pose_tpn = h_pose[self.batch_size:, :, :, :]
    h_diff = h_pose_tpn - h_pose_t
    xtp1 = self.decoder(h_img + h_diff)

    return xtp1

  def img_encoder(self, xt):
    conv1_1 = relu(
        conv2d(
            xt,
            output_dim=self.gf_dim,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv1_1'))
    conv1_2 = relu(
        conv2d(
            conv1_1,
            output_dim=self.gf_dim,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv1_2'))
    pool1 = MaxPooling(conv1_2, [2, 2])

    conv2_1 = relu(
        conv2d(
            pool1,
            output_dim=self.gf_dim * 2,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv2_1'))
    conv2_2 = relu(
        conv2d(
            conv2_1,
            output_dim=self.gf_dim * 2,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv2_2'))
    pool2 = MaxPooling(conv2_2, [2, 2])

    conv3_1 = relu(
        conv2d(
            pool2,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv3_1'))
    conv3_2 = relu(
        conv2d(
            conv3_1,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv3_2'))
    conv3_3 = relu(
        conv2d(
            conv3_2,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_iconv3_3'))
    pool3 = MaxPooling(conv3_3, [2, 2])

    return pool3

  def pose_encoder(self, pt):
    conv1_1 = relu(
        conv2d(
            pt,
            output_dim=self.gf_dim,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv1_1'))
    conv1_2 = relu(
        conv2d(
            conv1_1,
            output_dim=self.gf_dim,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv1_2'))
    pool1 = MaxPooling(conv1_2, [2, 2])

    conv2_1 = relu(
        conv2d(
            pool1,
            output_dim=self.gf_dim * 2,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv2_1'))
    conv2_2 = relu(
        conv2d(
            conv2_1,
            output_dim=self.gf_dim * 2,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv2_2'))
    pool2 = MaxPooling(conv2_2, [2, 2])

    conv3_1 = relu(
        conv2d(
            pool2,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv3_1'))
    conv3_2 = relu(
        conv2d(
            conv3_1,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv3_2'))
    conv3_3 = relu(
        conv2d(
            conv3_2,
            output_dim=self.gf_dim * 4,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='enc_pconv3_3'))
    pool3 = MaxPooling(conv3_3, [2, 2])

    return pool3

  def decoder(self, h_comb):
    shapel3 = [
        self.batch_size, self.image_size / 4, self.image_size / 4,
        self.gf_dim * 4
    ]
    shapeout3 = [
        self.batch_size, self.image_size / 4, self.image_size / 4,
        self.gf_dim * 2
    ]
    depool3 = FixedUnPooling(h_comb, [2, 2])
    deconv3_3 = relu(
        deconv2d(
            depool3,
            output_shape=shapel3,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv3_3'))
    deconv3_2 = relu(
        deconv2d(
            deconv3_3,
            output_shape=shapel3,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv3_2'))
    deconv3_1 = relu(
        deconv2d(
            deconv3_2,
            output_shape=shapeout3,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv3_1'))

    shapel2 = [
        self.batch_size, self.image_size / 2, self.image_size / 2,
        self.gf_dim * 2
    ]
    shapeout3 = [
        self.batch_size, self.image_size / 2, self.image_size / 2, self.gf_dim
    ]
    depool2 = FixedUnPooling(deconv3_1, [2, 2])
    deconv2_2 = relu(
        deconv2d(
            depool2,
            output_shape=shapel2,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv2_2'))
    deconv2_1 = relu(
        deconv2d(
            deconv2_2,
            output_shape=shapeout3,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv2_1'))

    shapel1 = [self.batch_size, self.image_size, self.image_size, self.gf_dim]
    shapeout1 = [self.batch_size, self.image_size, self.image_size, self.c_dim]
    depool1 = FixedUnPooling(deconv2_1, [2, 2])
    deconv1_2 = relu(
        deconv2d(
            depool1,
            output_shape=shapel1,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv1_2'))
    xtp1 = tanh(
        deconv2d(
            deconv1_2,
            output_shape=shapeout1,
            k_h=3,
            k_w=3,
            d_h=1,
            d_w=1,
            name='dec_deconv1_1'))
    return xtp1

  def discriminator(self, image):
    h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
    h1 = lrelu(
        batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'), 'bn1'))
    h2 = lrelu(
        batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'), 'bn2'))
    h3 = lrelu(
        batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'), 'bn3'))
    h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

    return tf.nn.sigmoid(h), h

  def save(self, sess, checkpoint_dir, step):
    model_name = 'IMGGEN.model'

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
