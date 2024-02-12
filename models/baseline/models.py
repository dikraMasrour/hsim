import tensorflow as tf

class BASELINE(tf.keras.models.Model):

    '''
    BASELINE model does the classification of pixels using all the bands
    It serves as a baseline reference for performance improvement of the MLBS and its variants
    
    '''

    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self._build_feature_extractor()
        self._build_head_classifier()

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
        # # block 3
        # self.fe_conv1d_9 = tf.keras.layers.Conv1D(128, 9, 1, padding='same', activation='relu')
        # self.fe_conv1d_10 = tf.keras.layers.Conv1D(128, 9, 1, padding='same', activation='relu')
        # self.fe_conv1d_11 = tf.keras.layers.Conv1D(128, 9, 1, padding='same', activation='relu')
        # self.fe_conv1d_12 = tf.keras.layers.Conv1D(128, 9, 1, padding='same', activation='relu')
        # # pooling
        # self.fe_maxpooling1d_3 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

    def _build_head_classifier(self):
        self.flatten = tf.keras.layers.Flatten()
        self.hc_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.hc_dense2 = tf.keras.layers.Dense(self.n_classes)
        self.hc_softmax = tf.keras.layers.Activation('softmax')

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
        x = self.hc_dense1(x)
        # x = self.dropout(x)
        x = self.hc_dense2(x)
        probabilities = self.hc_softmax(x)
        return probabilities

    def call(self, inputs):
        features = self._call_feature_extractor(inputs)
        probabilities = self._call_head_classifier(features)
        return probabilities

    def model(self):
        x = tf.keras.Input(shape=(self.n_bands, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    

    
class BINARY_BASELINE(tf.keras.models.Model):

    '''
    BINARY_BASELINE model does the binary classification of pixels using all the bands
    
    '''

    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self._build_feature_extractor()
        self._build_head_classifier()

    def build(self, input_shape):
        print(input_shape)
        super(BINARY_BASELINE, self).build(input_shape)

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
        self.fe_conv1d_9 = tf.keras.layers.Conv1D(16, 3, 1, padding='same', activation='relu')
        self.fe_conv1d_10 = tf.keras.layers.Conv1D(16, 9, 1, padding='same', activation='relu')
        # pooling
        self.fe_maxpooling1d_3 = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

    def _build_head_classifier(self):
        self.flatten = tf.keras.layers.Flatten()
        self.hc_dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.hc_dense2 = tf.keras.layers.Dense(1)
        self.hc_softmax = tf.keras.layers.Activation('sigmoid')

    def _call_feature_extractor(self, inputs):
        x = self.reshape(inputs)
        x = self.fe_conv1d_1(x)
        x = self.fe_conv1d_2(x)
        x = self.fe_maxpooling1d_1(x)
        x = self.fe_conv1d_5(x)
        x = self.fe_conv1d_6(x)
        x = self.fe_maxpooling1d_2(x)
        x = self.fe_conv1d_9(x)
        x = self.fe_conv1d_10(x)
        features = self.fe_maxpooling1d_3(x)
        return features

    def _call_head_classifier(self, inputs):
        x = self.flatten(inputs)
        x = self.hc_dense1(x)
        x = self.dropout(x)
        x = self.hc_dense2(x)
        probabilities = self.hc_softmax(x)
        return probabilities

    def call(self, inputs):
        features = self._call_feature_extractor(inputs)
        probabilities = self._call_head_classifier(features)
        return probabilities

    def model(self):
        x = tf.keras.Input(shape=(self.n_bands, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))