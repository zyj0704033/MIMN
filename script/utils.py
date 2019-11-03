import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
#from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K
import numpy as np

def din_attention(query, facts, attention_size, mask=None, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("query_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all

    if mask is not None:
        mask = tf.equal(mask, tf.ones_like(mask))
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    
    if return_alphas:
        return output, scores
        
    return output


class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

def calc_gauc(raw_arr, nick_index):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    last_index = 0
    gauc = 0.
    pv_sum = 0
    for idx in xrange(len(nick_index)):
        if nick_index[idx] != nick_index[last_index]:
            input_arr = raw_arr[last_index:idx]
            auc_val=calc_auc(input_arr)
            if auc_val >= 0.0:
                gauc += auc_val * len(input_arr)
                pv_sum += len(input_arr)
            else:
                pv_sum += len(input_arr) 
            last_index = idx
    return gauc / pv_sum
                



def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    input_size = query.get_shape().as_list()[-1]

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        tmp1 = tf.tensordot(facts, w1, axes=1)
        tmp2 = tf.tensordot(query, w2, axes=1)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    if mask is not None:
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch[:, 0:i+1, :],
                                               ATTENTION_SIZE, mask[:, 0:i+1], softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch,
                                               ATTENTION_SIZE, mask, softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1_trans_shine' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, facts_size, activation=tf.nn.sigmoid, name='f1_shine_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, facts_size, activation=tf.nn.sigmoid, name='f2_shine_att' + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all, tf.shape(facts))
    output = d_layer_2_all
    return output


def mask_to_length(data):
    assert len(data.shape) > 2, "shape should be (batch_size, steps, ...)"
    mask = tf.to_int32(tf.not_equal(tf.reduce_sum(data, axis=range(2, len(data.shape)), keepdims=True), 0))
    length = tf.reduce_sum(mask, [1, 2])
    return tf.to_float(mask), length

def dist_matrix(inputs, mask, dist_type='Euclid'):
    # inputs : b * seq_len * embed_dim
    shape = inputs.shape.as_list()
    mask = tf.reshape(mask, [shape[0], shape[1], 1])
    if mask is not None:
        inputs = tf.multiply(inputs, mask)
    if dist_type == 'Euclid':
        square_input = tf.reduce_sum(tf.square(inputs), axis=2, keep_dims=True) # b * seq_len * 1
        tf_ones = tf.constant(np.ones((shape[0], shape[1], 1)), dtype=tf.float32) # b * seq_len * 1
        sqx = tf.concat([square_input, tf_ones], axis=2)
        sqy = tf.concat([tf_ones, square_input], axis=2)
        # sqy = tf.reshape(sqy, [shape[0], 2, shape[1]])
        square_matrix = tf.matmul(sqx, sqy, transpose_b=True) # b * seq_len * seq_len()
        xy = tf.matmul(inputs, inputs, transpose_b=True) 
        dis_matrix = square_matrix - 2*xy + tf.constant(1e-6, dtype=tf.float32)
        dis_matrix = tf.sqrt(dis_matrix)

        return dis_matrix


def get_eigenvec(inputs, mask, use_svd=False):
    '''
    calculate eigenvec

    inputs: b * seq_len * embed_dim
    mask: 
    return: b * embed
    '''
    shape = inputs.shape.as_list()
    mask = tf.reshape(mask, [shape[0], shape[1], 1])
    inputs = tf.multiply(inputs, mask)
    inputs =  tf.matmul(inputs, inputs, transpose_a=True) # b * embed * embed
    eigvec = power_iteration(inputs)[1]
    eigvec = tf.reshape(eigvec, [shape[0], shape[-1]])
    print("\33[35m eigenvec shape: \33[0m")
    print(eigvec.shape.as_list)
    return eigvec



def cal_cluster_loss(inputs, mask, dist_type='Euclid', use_svd=False):
    '''
    calculate the sum of Lapalace Matrix eigen value as cluster loss
    
    inputs: b * seq_len * embed_dim
    return: scalar cluster_loss
    '''
    shape = inputs.shape.as_list()
    W = dist_matrix(inputs, mask ,dist_type=dist_type) # b * seq_len * seq_len
    degree = tf.reduce_sum(W, axis=2) # b * seq_len
    degree = tf.reshape(degree, [shape[0], shape[1], 1]) # b * seq_len * 1
    tf_eye = tf.eye(shape[1], batch_shape=[shape[0]]) # b * seq_len * seq_len
    D = tf.multiply(degree, tf_eye)
    D_norm =tf.multiply(1 / tf.sqrt(degree), tf_eye) # b * seq_len * seq_len
    L = D - W
    L = tf.matmul(D_norm, L)
    L = tf.matmul(L, D_norm)
    I = tf.eye(shape[1], batch_shape=[shape[0]])
    L = L - 0.5 * I # friendly to power iteration
    if use_svd:
        s = tf.svd(L, compute_uv=False)
        s = s[:,0]
    else:
        s = power_iteration(L)[0]
        s = tf.reshape(s, [shape[0]])
    return -s # b

def power_iteration(L, iteration=10):
    '''
    power iteration to calculate eigenvalue and eigenvector
    ref: https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    '''
    shape = L.shape.as_list() # b * seq_len * seq_len
    u = tf.get_variable("u", [shape[0], 1, shape[-1]], initializer=tf.random_normal_initializer(), trainable=False) # b * 1 * seq_len
    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, L, transpose_b=True)
        v_hat = tf.nn.l2_normalize(v_, dim=0)

        u_ = tf.matmul(v_hat, L)
        u_hat = tf.nn.l2_normalize(u_, dim=0)
    
    # u_hat = tf.stop_gradient(u_hat)
    # v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, L), u_hat, transpose_b=True)

    return sigma, u_hat

'''
def matrix_symmetric(x):
    return (x + tf.transpose(x, [0,2,1])) / 2

def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])

    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res

@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    """
    Define the gradient for SVD
    References
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers
        
    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------
    """
    s, u, v = op.outputs
    v_t = tf.transpose(v, [0,2,1])

    with tf.name_scope('K'):
        K = get_eigen_K(s, True)
    inner = matrix_symmetric(K * tf.matmul(v_t, grad_v))

    # Create the shape accordingly.
    u_shape = u.get_shape()[1].value
    v_shape = v.get_shape()[1].value

    # Recover the complete S matrices and its gradient
    eye_mat = tf.eye(v_shape, u_shape)
    realS = tf.matmul(tf.reshape(tf.matrix_diag(s), [-1, v_shape]), eye_mat)
    realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])

    real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(grad_s), [-1, v_shape]), eye_mat)
    real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])

    dxdz = tf.matmul(u, tf.matmul(2 * tf.matmul(realS, inner) + real_grad_S, v_t))
    return dxdz
'''
class kmeans(object):
    def __init__(self, cluster_num, max_iter=20, use_plus=False, distance_type='Euclid'):
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.use_plus = use_plus # wether use kmeans++
        self.distance_type = distance_type

    def __call__(self, input):
        '''
        input: b * seq * e
        return: center of the seq b * seq * e
        '''
        # initialization the centers
        input_stop = tf.stop_gradient(input, name='kmeans_stop')
        mask, length = mask_to_length(input_stop)
        shape = input.shape.as_lit() # b * seq * e
        centroids_idx = tf.to_int32(tf.floor(tf.random_uniform((1, self.cluster_num), minval=0, maxval=1) * shape[1]))
        centroids = tf.gather(input_stop, centroids_idx[0,:], axis=1) # b * cn * e
        # floop
        for i in range(self.max_iter):
            pdistance = self.point_distance(input_stop, centroids) # b * seq * cn
            passignment = tf.argmin(pdistance, axis=2) # b * seq
            center_list = []
            for k in range(self.cluster_num):
                mask_k = tf.reshape(tf.to_float(tf.equal(passignment, k)), (shape[0], shape[1], 1)) # b * seq_len * 1
                center_k = tf.reduce_sum(input_stop * mask_k, axis=1, keep_dims=True) # b * 1 * e
                pkn = tf.ones(shape=(shape[0], shape[1], 1)) * mask * mask_k # b * seq_len * 1
                pkn = tf.reduce_sum(pkn, axis=1, keep_dims=True) # b * 1 * 1
                center_k = center_k / (pkn + 0.01)
                center_list.append(center_k)
            centroids = tf.concat(center_list, axis=1)
        return centroids

    def point_distance(self, inputs, centroids):
        shape = inputs.shape.as_list()
        if self.distance_type == 'cosian':
            pass
        if self.distance_type == 'Euclid':
            mask, length  = mask_to_length(inputs)
            inputs = inputs * mask
            square_input = tf.reduce_sum(tf.square(inputs), axis=2, keep_dims=True) # b * seq_len * 1
            sqx = tf.matmul(square_input, tf.ones(shape=(shape[0], 1, self.cluster_num))) # b * seq_len * cn
            square_center = tf.reduce_sum(tf.square(centroids), axis=2, keep_dims=True) # b * cn * 1
            sqy = tf.matmul(tf.ones(shape[0], shape[1], 1), square_center, transpose_b=True) # b * seq_len * cn
            xy = tf.matmul(inputs, centroids, transpose_b=True) # b * seq_len * cn
            dis_matrix = sqx + sqy - 2*xy + tf.constant(1e-6, dtype=tf.float32)
            dis_matrix = tf.sqrt(dis_matrix)

            return dis_matrix # b * seq_len * cn
