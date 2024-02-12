import tensorflow as tf
from .layers import (
    Weights,
    Sigmoid,
    Normalization,
    ContinuousBernoulliSampling,
    Stem_Block, Stem_Bottleneck_Block,
    Learner18, Learner34, Learner50, Learner101, Learner152
)
class RESNET_BINARY(tf.keras.models.Model):

    def __init__(self, n_bands, n_classes, res_num_filters=4, dropout=0.3, learner=18):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.num_filters = res_num_filters
        self.dropout = dropout
        self.learner = learner

        self._build_feature_extractor()
        self._build_head_classifier()
    
    def _build_feature_extractor(self):
        self.reshape = tf.keras.layers.Reshape((self.n_bands, 1))
        if self.learner == 18: 
            self.stem = Stem_Block(self.num_filters)
            self.learner = Learner18(self.num_filters)
        elif self.learner == 34: 
            self.stem = Stem_Block(self.num_filters)
            self.learner = Learner34(self.num_filters)
        elif self.learner == 50: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner50(self.num_filters)
        elif self.learner == 101: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner101(self.num_filters)
        elif self.learner == 152: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner152(self.num_filters)

    def _build_head_classifier(self):
        self.maxpool = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.hc_dense2 = tf.keras.layers.Dense(1)
        self.hc_sigmoid = tf.keras.layers.Activation('sigmoid')

    def _call_feature_extractor(self, inputs):
        x = self.reshape(inputs)
        stem_ = self.stem(x)  
        features = self.learner(stem_)
        return features

    def _call_head_classifier(self, inputs):
        x = self.maxpool(inputs)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.hc_dense2(x)
        probabilities = self.hc_sigmoid(x)
        return probabilities

    def call(self, inputs, training):
        features = self._call_feature_extractor(inputs)
        probability = self._call_head_classifier(features)
        return probability

    def model(self, training=True):
        x = tf.keras.Input(shape=(self.n_bands, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training))
    

class RESNET_BINARY_MLBS(tf.keras.models.Model):

    """
        This is an alternative implementation of MLBS
        that performs binary classification with a ResNet18
        as classification network
        
    """

    def __init__(self, n_bands, n_classes, res_num_filters, band_selection_ratio=0.3, filtering_slope=5, sampler_slope=200, dropout=0.3, learner=18):

        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.band_selection_ratio = band_selection_ratio
        self.filtering_slope = filtering_slope
        self.sampler_slope = sampler_slope
        self.num_filters = res_num_filters
        self.dropout = dropout
        self.learner = learner

        self._build_mask_sampler()
        self._build_feature_extractor()
        self._build_head_classifier()

    def _build_mask_sampler(self):
        self.ms_weights = Weights(shape=(1, self.n_bands), initializer="random_uniform")
        self.ms_sigmoid = Sigmoid(slope=self.filtering_slope)
        self.ms_normalization = Normalization(alpha=self.band_selection_ratio)
        self.ms_continuous_bernoulli_sampling = ContinuousBernoulliSampling(slope=self.sampler_slope)
    
    def _build_feature_extractor(self):
        self.reshape = tf.keras.layers.Reshape((self.n_bands, 1))
        if self.learner == 18: 
            self.stem = Stem_Block(self.num_filters)
            self.learner = Learner18(self.num_filters)
        elif self.learner == 34: 
            self.stem = Stem_Block(self.num_filters)
            self.learner = Learner34(self.num_filters)
        elif self.learner == 50: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner50(self.num_filters)
        elif self.learner == 101: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner101(self.num_filters)
        elif self.learner == 152: 
            self.stem = Stem_Bottleneck_Block(self.num_filters)
            self.learner = Learner152(self.num_filters)

    def _build_head_classifier(self):
        self.maxpool = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.hc_dense2 = tf.keras.layers.Dense(1)
        self.hc_sigmoid = tf.keras.layers.Activation('sigmoid')

    def _call_mask_sampler(self, inputs, training):
        x = self.ms_weights(inputs)
        x = self.ms_sigmoid(x)
        x = self.ms_normalization(x)
        masks = self.ms_continuous_bernoulli_sampling(x, training)
        return masks

    def _call_feature_extractor(self, inputs):
        x = self.reshape(inputs)
        stem_ = self.stem(x)  
        features = self.learner(stem_)
        return features

    def _call_head_classifier(self, inputs):
        x = self.maxpool(inputs)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.hc_dense2(x)
        probabilities = self.hc_sigmoid(x)
        return probabilities

    def call(self, inputs, training):
        masks = self._call_mask_sampler(inputs, training)
        masked_pixels = tf.multiply(inputs, masks)
        features = self._call_feature_extractor(masked_pixels)
        probability = self._call_head_classifier(features)
        return probability

    def model(self, training=True):
        x = tf.keras.Input(shape=(1, self.n_bands))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training))
