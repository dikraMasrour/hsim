import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


''' Credits for resnet implementation: https://github.com/Sakib1263/TF-1D-2D-ResNetV1-2-SEResNet-ResNeXt-SEResNeXt'''
class Conv_1D_Block(tf.keras.layers.Layer): 
    def __init__(self, num_filters, kernel_size, strides):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.conv1d = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, self.strides, 
                                             padding="same", kernel_initializer="he_normal")
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        super(Conv_1D_Block, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


# **********  RESNET normal layers  *************
    
class Stem_Block(tf.keras.layers.Layer):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = 7
        self.strides = 2
    
    def build(self, input_shape):
        self.conv1d_block = Conv_1D_Block(self.num_filters, self.kernel_size, self.strides)
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")
        super(Stem_Block, self).build(input_shape)

    def call(self, inputs):
        conv = self.conv1d_block(inputs)
        if conv.shape[1] <= 2:
            pool = self.maxpool1(conv)
        else:
            pool = self.maxpool2(conv)
        
        return pool

class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, strides1, strides2):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides1 = strides1 
        self.strides2 = strides2 
    
    def build(self, input_shape):
        self.conv1d_block1 = Conv_1D_Block(self.num_filters, self.kernel_size, self.strides1)
        self.conv1d_block2 = Conv_1D_Block(self.num_filters, self.kernel_size, self.strides2)
        super(Conv_Block, self).build(input_shape)

    def call(self, inputs):
        # Construct Block of Convolutions without Pooling
        # x        : input into the block
        # n_filters: number of filters
        conv = self.conv1d_block1(inputs)
        conv = self.conv1d_block2(conv)
        return conv
       

class Residual_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, strides1, strides2):
            super().__init__()
            self.num_filters = num_filters
            self.kernel_size = kernel_size 
            self.strides1 = strides1 
            self.strides2 = strides2 

    def build(self, input_shape):
        self.conv_block = Conv_Block(self.num_filters, self.kernel_size, self.strides1, self.strides2)
        self.add_residual = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu')
        super(Residual_Block, self).build(input_shape)

    def call(self, inputs):
        # Construct a Residual Block of Convolutions
        # x        : input into the block
        # n_filters: number of filters
        shortcut = inputs
        conv = self.conv_block(inputs)
        res = self.add_residual([conv, shortcut])
        output = self.relu(res)

        return (output)

class Residual_Group(tf.keras.layers.Layer):
    def __init__(self, num_filters, n_blocks):
            super().__init__()
            self.num_filters = num_filters
            self.n_blocks = n_blocks
            self.kernel_size = 3
            self.conv_strides1, self.conv_strides2 = 2, 1
            self.res_strides1, self.res_strides2 = 1, 1
    
    def build(self, input_shape):
        self.residual_block = Residual_Block(self.num_filters, self.kernel_size, self.res_strides1, self.res_strides2)
        self.conv_block = Conv_Block(self.num_filters*2, self.kernel_size, self.conv_strides1, self.conv_strides2)
        super(Residual_Group, self).build(input_shape)

    def call(self, inputs, conv=True):
        # x        : input to the group
        # n_filters: number of filters
        # n_blocks : number of blocks in the group
        # conv     : flag to include the convolution block connector
        output = inputs
        for i in range(self.n_blocks):
            output = self.residual_block(output)
        
        # Connecting block
        if conv:
            output = self.conv_block(output)
        return output
    

# **********  RESNET bottleneck layers  *************
class Stem_Bottleneck_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters 
        self.kernel_size = 7
        self.strides = 2
    
    def build(self, input_shape):
        self.conv1d_block = Conv_1D_Block(self.num_filters, self.kernel_size, self.strides)
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")
        super(Stem_Bottleneck_Block, self).build(input_shape)

    def call(self, inputs):
        conv = self.conv1d_block(inputs)
        if conv.shape[1] <= 2:
            pool = self.maxpool1(conv)
        else:
            pool = self.maxpool2(conv)
        
        return pool

class Conv_Bottleneck_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size1, kernel_size2, strides1):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.strides1 = strides1
    
    def build(self, input_shape):
        self.conv1d_block1 = Conv_1D_Block(self.num_filters, self.kernel_size1, self.strides1)
        self.conv1d_block2 = Conv_1D_Block(self.num_filters, self.kernel_size2, self.strides1)
        self.conv1d_block3 = Conv_1D_Block(self.num_filters*4, self.kernel_size1, self.strides1)
        super(Conv_Bottleneck_Block, self).build(input_shape)

    def call(self, inputs):
        # Construct Block of Convolutions without Pooling
        # x        : input into the block
        # n_filters: number of filters
        conv = self.conv1d_block1(inputs)
        conv = self.conv1d_block2(conv)
        conv = self.conv1d_block3(conv)
        return conv

class Residual_Bottleneck_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size1, kernel_size2, strides1):
            super().__init__()
            self.num_filters = num_filters
            self.kernel_size1 = kernel_size1
            self.kernel_size2 = kernel_size2
            self.strides1 = strides1

    def build(self, input_shape):
        self.shortcut_conv = Conv_1D_Block(self.num_filters*4, self.kernel_size1, self.strides1)
        self.conv_bottleneck_block = Conv_Bottleneck_Block(self.num_filters, self.kernel_size1, self.kernel_size2, self.strides1)
        self.add_residual = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu')
        super(Residual_Bottleneck_Block, self).build(input_shape)

    def call(self, inputs):
        # Construct a Residual Block of Convolutions
        # x        : input into the block
        # n_filters: number of filters
        shortcut = self.shortcut_conv(inputs)
        conv = self.conv_bottleneck_block(inputs)
        res = self.add_residual([conv, shortcut])
        output = self.relu(res)

        return (output)
    
class Residual_Bottleneck_Group(tf.keras.layers.Layer):
    def __init__(self, num_filters, n_blocks):
        super().__init__()
        self.num_filters = num_filters
        self.n_blocks = n_blocks
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.res_strides1 = 1
        self.conv_strides1, self.conv_strides2 = 2, 1
    
    def build(self, input_shape):
        self.residual_bottleneck_block = Residual_Bottleneck_Block(self.num_filters, self.kernel_size1, self.kernel_size2, self.res_strides1)
        self.conv_block = Conv_Block(self.num_filters*2, self.kernel_size2, self.conv_strides1, self.conv_strides2)
        super(Residual_Bottleneck_Group, self).build(input_shape)

    def call(self, inputs, conv=True):
        # x        : input to the group
        # n_filters: number of filters
        # n_blocks : number of blocks in the group
        # conv     : flag to include the convolution block connector
        output = inputs
        for i in range(self.n_blocks):
            output = self.residual_bottleneck_block(output)
        
        # Connecting block
        if conv:
            output = self.conv_block(output)
        return output
    

# **********  RESNET Learners (18, 34, 50, 101, 152)  *************

class Learner18(tf.keras.layers.Layer):
    def __init__(self, num_filters):
            self.num_filters = num_filters
            super().__init__()
            
    def build(self, input_shape):
        self.residual_group = Residual_Group(self.num_filters, n_blocks=2)
        self.residual_group1 = Residual_Group(self.num_filters*2, n_blocks=1)
        self.residual_group2 = Residual_Group(self.num_filters*4, n_blocks=1)
        self.residual_group3 = Residual_Group(self.num_filters*8, n_blocks=1)
        super(Learner18, self).build(input_shape)

    def call(self, inputs):
        x = self.residual_group(inputs)
        x = self.residual_group1(x)
        x = self.residual_group2(x)
        x = self.residual_group3(x, conv=False)
        return x
    

class Learner34(tf.keras.layers.Layer):
    def __init__(self, num_filters):
            self.num_filters = num_filters
            super().__init__()
            
    def build(self, input_shape):
        self.residual_group = Residual_Group(self.num_filters, n_blocks=3)
        self.residual_group1 = Residual_Group(self.num_filters*2, n_blocks=3)
        self.residual_group2 = Residual_Group(self.num_filters*4, n_blocks=5)
        self.residual_group3 = Residual_Group(self.num_filters*8, n_blocks=2)
        super(Learner34, self).build(input_shape)

    def call(self, inputs):
        x = self.residual_group(inputs)
        x = self.residual_group1(x)
        x = self.residual_group2(x)
        x = self.residual_group3(x, conv=False)
        return x
    
class Learner50(tf.keras.layers.Layer):
    def __init__(self, num_filters):
            self.num_filters = num_filters
            super().__init__()
            
    def build(self, input_shape):
        self.residual_bottleneck_group = Residual_Bottleneck_Group(self.num_filters, n_blocks=3)
        self.residual_bottleneck_group1 = Residual_Bottleneck_Group(self.num_filters*2, n_blocks=3)
        self.residual_bottleneck_group2 = Residual_Bottleneck_Group(self.num_filters*4, n_blocks=5)
        self.residual_bottleneck_group3 = Residual_Bottleneck_Group(self.num_filters*8, n_blocks=2)
        super(Learner50, self).build(input_shape)

    def call(self, inputs):
        x = self.residual_bottleneck_group(inputs)
        x = self.residual_bottleneck_group1(x)
        x = self.residual_bottleneck_group2(x)
        x = self.residual_bottleneck_group3(x, conv=False)
        return x


class Learner101(tf.keras.layers.Layer):
    def __init__(self, num_filters):
            self.num_filters = num_filters
            super().__init__()
            
    def build(self, input_shape):
        self.residual_bottleneck_group = Residual_Bottleneck_Group(self.num_filters, n_blocks=3)
        self.residual_bottleneck_group1 = Residual_Bottleneck_Group(self.num_filters*2, n_blocks=3)
        self.residual_bottleneck_group2 = Residual_Bottleneck_Group(self.num_filters*4, n_blocks=2)
        self.residual_bottleneck_group3 = Residual_Bottleneck_Group(self.num_filters*8, n_blocks=2)
        super(Learner101, self).build(input_shape)

    def call(self, inputs):
        x = self.residual_bottleneck_group(inputs)
        x = self.residual_bottleneck_group1(x)
        x = self.residual_bottleneck_group2(x)
        x = self.residual_bottleneck_group3(x, conv=False)
        return x
    
class Learner152(tf.keras.layers.Layer):
    def __init__(self, num_filters):
            self.num_filters = num_filters
            super().__init__()
            
    def build(self, input_shape):
        self.residual_bottleneck_group = Residual_Bottleneck_Group(self.num_filters, n_blocks=3)
        self.residual_bottleneck_group1 = Residual_Bottleneck_Group(self.num_filters*2, n_blocks=7)
        self.residual_bottleneck_group2 = Residual_Bottleneck_Group(self.num_filters*4, n_blocks=3)
        self.residual_bottleneck_group3 = Residual_Bottleneck_Group(self.num_filters*8, n_blocks=2)
        super(Learner152, self).build(input_shape)

    def call(self, inputs):
        x = self.residual_bottleneck_group(inputs)
        x = self.residual_bottleneck_group1(x)
        x = self.residual_bottleneck_group2(x)
        x = self.residual_bottleneck_group3(x, conv=False)
        return x
    

# **********  MLBS Custom layers   *************
    
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
        samples = tf.constant(unif_sampler((1, input_shape[1])))
        self.thres = samples
        super(ContinuousBernoulliSampling, self).build(input_shape)


    def call(self, inputs, training):
        if training == True: # @Training
          return tf.math.sigmoid(self.slope * (inputs - self.thres))
        else: # @Inference
          return float(inputs > self.thres)
        