import tensorflow as tf
from .layers import (
    Weights,
    Sigmoid,
    Normalization,
    ContinuousBernoulliSampling
)


class MLBS(tf.keras.models.Model):

    """
        This is an implementation of MLBS
        paper: https://doi.org/10.3390/rs15184460
    """

    def __init__(self,
                 n_bands, n_classes,
                 band_selection_ratio=0.3,
                 filtering_slope=5,
                 sampler_slope=200):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.band_selection_ratio = band_selection_ratio
        self.filtering_slope = filtering_slope
        self.sampler_slope = sampler_slope
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
        # block 1
        self.fe_conv1d_1 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        self.fe_conv1d_2 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        self.fe_conv1d_3 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        self.fe_conv1d_4 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_1 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')
        # block 2
        self.fe_conv1d_5 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        self.fe_conv1d_6 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        self.fe_conv1d_7 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        self.fe_conv1d_8 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_2 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

    def _build_head_classifier(self):
        self.flatten = tf.keras.layers.Flatten()
        self.hc_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.hc_dense2 = tf.keras.layers.Dense(self.n_classes)
        self.hc_softmax = tf.keras.layers.Activation('softmax')

    def _call_mask_sampler(self, inputs, training):
        x = self.ms_weights(inputs)
        x = self.ms_sigmoid(x)
        x = self.ms_normalization(x)
        masks = self.ms_continuous_bernoulli_sampling(x, training)
        return masks

    def _call_feature_extractor(self, inputs):
        x = self.reshape(inputs)
        x = self.fe_conv1d_1(x)
        x = self.fe_conv1d_2(x)
        x = self.fe_conv1d_3(x)
        # x = self.fe_conv1d_4(x)
        x = self.fe_maxpooling1d_1(x)
        x = self.fe_conv1d_5(x)
        x = self.fe_conv1d_6(x)
        x = self.fe_conv1d_7(x)
        # x = self.fe_conv1d_8(x)
        features = self.fe_maxpooling1d_2(x)
        return features

    def _call_head_classifier(self, inputs):
        x = self.flatten(inputs)
        # x = self.hc_dense(x)
        x = self.hc_dense1(x)
        # x = self.dropout(x)
        x = self.hc_dense2(x)
        probabilities = self.hc_softmax(x)
        return probabilities

    def call(self, inputs, training):
        masks = self._call_mask_sampler(inputs, training)
        masked_pixels = tf.multiply(inputs, masks)
        features = self._call_feature_extractor(masked_pixels)
        probabilities = self._call_head_classifier(features)
        return probabilities

    def model(self, training=True):
        x = tf.keras.Input(shape=(1, self.n_bands))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training))
    

class BINARY_MLBS32(tf.keras.models.Model):

    """
        This is an alternative implementation of MLBS
        that performs binary classification
        
    """

    def __init__(self,
                 n_bands, n_classes,
                 band_selection_ratio=0.3,
                 filtering_slope=5,
                 sampler_slope=200):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.band_selection_ratio = band_selection_ratio
        self.filtering_slope = filtering_slope
        self.sampler_slope = sampler_slope
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
        # block 1
        self.fe_conv1d_1 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        self.fe_conv1d_2 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_1 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')
        # block 2
        self.fe_conv1d_5 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        self.fe_conv1d_6 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_2 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')
        # block 3
        self.fe_conv1d_7 = tf.keras.layers.Conv1D(16, 3, 1, padding='same', activation='relu')
        self.fe_conv1d_8 = tf.keras.layers.Conv1D(16, 3, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_3 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

    def _build_head_classifier(self):
        self.flatten = tf.keras.layers.Flatten()
        self.hc_dense1 = tf.keras.layers.Dense(32, activation='relu')
        # self.hc_dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
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
        x = self.fe_conv1d_1(x)
        x = self.fe_conv1d_2(x)
        x = self.fe_maxpooling1d_1(x)
        x = self.fe_conv1d_5(x)
        x = self.fe_conv1d_6(x)
        x = self.fe_maxpooling1d_2(x)
        x = self.fe_conv1d_7(x)
        x = self.fe_conv1d_8(x)
        features = self.fe_maxpooling1d_3(x)
        return features

    def _call_head_classifier(self, inputs):
        x = self.flatten(inputs)
        x = self.hc_dense1(x)
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


class BINARY_MLBS16(tf.keras.models.Model):

    """
        This is an alternative implementation of MLBS
        that performs binary classification
        
    """

    def __init__(self,
                 n_bands, n_classes,
                 band_selection_ratio=0.3,
                 filtering_slope=5,
                 sampler_slope=200):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.band_selection_ratio = band_selection_ratio
        self.filtering_slope = filtering_slope
        self.sampler_slope = sampler_slope
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
        # block 1
        self.fe_conv1d_1 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        self.fe_conv1d_2 = tf.keras.layers.Conv1D(64, 15, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_1 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')
        # block 2
        self.fe_conv1d_5 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        self.fe_conv1d_6 = tf.keras.layers.Conv1D(32, 9, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_2 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')
        # block 3
        self.fe_conv1d_7 = tf.keras.layers.Conv1D(16, 3, 1, padding='same', activation='relu')
        self.fe_conv1d_8 = tf.keras.layers.Conv1D(16, 3, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_3 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

    def _build_head_classifier(self):
        self.flatten = tf.keras.layers.Flatten()
        # self.hc_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.hc_dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
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
        x = self.fe_conv1d_1(x)
        x = self.fe_conv1d_2(x)
        x = self.fe_maxpooling1d_1(x)
        x = self.fe_conv1d_5(x)
        x = self.fe_conv1d_6(x)
        x = self.fe_maxpooling1d_2(x)
        x = self.fe_conv1d_7(x)
        x = self.fe_conv1d_8(x)
        features = self.fe_maxpooling1d_3(x)
        return features

    def _call_head_classifier(self, inputs):
        x = self.flatten(inputs)
        x = self.hc_dense1(x)
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