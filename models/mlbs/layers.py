import tensorflow as tf
import tensorflow_probability as tfp


class Weights(tf.keras.layers.Layer):

    def __init__(self, shape, initializer="random_normal"):
        super().__init__()
        self.w = self.add_weight(
            shape=shape,
            initializer=initializer,
            trainable=True,
        )

    def build(self, input_shape):
      super(Weights, self).build(input_shape)

    def call(self, inputs):
        return self.w


class Sigmoid(tf.keras.layers.Layer):

    def __init__(self, slope=0.5):
        super().__init__()
        self.slope = slope
      
    def build(self, input_shape):
      super(Sigmoid, self).build(input_shape)

    def call(self, inputs):
        return tf.math.sigmoid(self.slope * inputs)


class Normalization(tf.keras.layers.Layer):

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def build(self, input_shape):
      super(Normalization, self).build(input_shape)
        
    def call(self, inputs):
        s = tf.reduce_mean(inputs, axis=0)
        cond = tf.cast(tf.greater_equal(s, self.alpha), dtype=tf.float32)
        ge = (self.alpha / s) * inputs
        lt = 1.0 - (((1.0 - self.alpha) / (1.0 - s)) * (1.0 - inputs))
        return cond * ge + (1 - cond) * lt


class ContinuousBernoulliSampling(tf.keras.layers.Layer):

    def __init__(self, slope=10000):
        super().__init__()
        self.slope = slope

    def build(self, input_shape):
        unif_sampler = tf.random_uniform_initializer(minval=0, maxval=1)
        samples = tf.constant(unif_sampler((1, input_shape[0])))
        self.thres = samples
        super(ContinuousBernoulliSampling, self).build(input_shape)


    def call(self, inputs, training):
        if training == True: # @Training
          return tf.math.sigmoid(self.slope * (inputs - self.thres))
        else: # @Inference
          return float(inputs > self.thres)