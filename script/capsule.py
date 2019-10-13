import numpy as np
import tensorflow as tf

class Capsule_cell(object):
    def __init__(self, num_outputs, emb_dim, mask=None, bIJ=None, routing_iter=3):
        self.num_outputs = num_outputs
        self.emb_dim = emb_dim
        self.mask = mask # (b, seq_len, 1)
        self.bIJ = bIJ
        self.routing_iter = 3
    
    def __call__(self, input):
        # input (b, seq_len, emb_dim)
        shape = input.shape.as_list()
        with tf.variable_scope('routing'):
            if self.bIJ is None:
                self.bIJ = tf.constant(np.random.rand(shape[0], shape[1], self.num_outputs, 1,1), dtype=np.float32) # (b, seq_len, num_output , 1, 1)
            cap = self.routing(input, self.bIJ)
            cap = tf.reshape(cap, [shape[0], self.num_outputs, self.emb_dim])
            return cap
    
    def routing(self, input, bIJ):
        shape = input.shape.as_list()
        W = tf.get_variable('SWeight', shape=[1, 1, 1, shape[2], self.emb_dim],  
                        dtype=tf.float32) # (b, seq_len, num_outputs, iemd_dim, oemb_dim)
        input = tf.reshape(input, [shape[0], shape[1], 1, 1, shape[2]])
        W = tf.tile(W, [shape[0], shape[1], self.num_outputs, 1, 1])
        input = tf.tile(input, [1, 1, self.num_outputs, 1, 1])
        u_hat = tf.matmul(input, W) # (b, seq_len, num_outputs, 1, oemb_dim)
        u_hat = tf.reshape(u_hat, [shape[0], shape[1], self.num_outputs, self.emb_dim ,1])# (b, seq_len, num_outputs, oemb_dim, 1)
        if self.mask is not None:
            mask = tf.reshape(self.mask, [shape[0], shape[1], 1, 1, 1])
            u_hat = tf.multiply(u_hat, mask) #mask
        u_hat_stop = tf.stop_gradient(u_hat, name='stop_gradient') # (b, seq_len, num_outputs, oemb_dim, 1)

        for r_iter in range(self.routing_iter):
            with tf.variable_scope('iter_' + str(r_iter)):
                cIJ = tf.nn.softmax(bIJ, dim=2) # (b, seq_len, num_output , 1, 1)
                if r_iter == self.routing_iter - 1:
                    sJ = tf.multiply(cIJ, u_hat) # (b, seq_len, num_outputs, oemb_dim, 1) TODO: mask for SJ
                    sJ = tf.reduce_sum(sJ, axis=1) # (b, num_outputs, oemb_dim, 1)
                    vJ = self.squash(sJ)
                
                if r_iter < self.routing_iter - 1:
                    sJ = tf.multiply(cIJ, u_hat_stop) # (b, seq_len, num_outputs, oemb_dim, 1) TODO: mask for SJ and add bias
                    sJ = tf.reduce_sum(sJ, axis=1, keep_dims=True) # (b, 1, num_outputs, oemb_dim, 1)
                    vJ = self.squash(sJ) # (b, 1, num_outputs, oemb_dim, 1)

                    vJ_tiled = tf.tile(vJ, [1, shape[1], 1, 1, 1]) # (b, seq_len, num_outputs, oemb_dim, 1)
                    u_dot_v = tf.reduce_sum(u_hat_stop*vJ_tiled, axis=3, keep_dims=True) # (b, seq_len, num_outputs, 1)
                    # print(bIJ.shape)
                    # print(u_dot_v.shape)
                    bIJ += u_dot_v
        return vJ

    def squash(self, vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 0.000000001)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)
