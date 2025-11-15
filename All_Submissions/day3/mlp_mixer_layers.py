# mlp_mixer_layers.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="MLPMixer")
class PatchEmbedding(layers.Layer):
    """Extract and embed image patches with proper serialization"""

    def __init__(self, num_patches, patch_dim, hidden_dim, image_height,
                 image_width, image_channels, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.patch_size = patch_size

    def build(self, input_shape):
        # Define weights in build method
        self.projection = layers.Dense(self.hidden_dim, name='projection')
        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        images = tf.reshape(images, [-1, self.image_height, self.image_width, self.image_channels])

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])
        embedded = self.projection(patches)
        return embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'patch_dim': self.patch_dim,
            'hidden_dim': self.hidden_dim,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'image_channels': self.image_channels,
            'patch_size': self.patch_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="MLPMixer")
class MixerBlock(layers.Layer):
    """MLP-Mixer block with proper serialization"""

    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim,
                 channels_mlp_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Define all sublayers in build method
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='norm2')

        self.token_dense1 = layers.Dense(self.tokens_mlp_dim, activation='gelu', name='token_dense1')
        self.token_dropout1 = layers.Dropout(self.dropout_rate, name='token_dropout1')
        self.token_dense2 = layers.Dense(self.num_patches, name='token_dense2')
        self.token_dropout2 = layers.Dropout(self.dropout_rate, name='token_dropout2')

        self.channel_dense1 = layers.Dense(self.channels_mlp_dim, activation='gelu', name='channel_dense1')
        self.channel_dropout1 = layers.Dropout(self.dropout_rate, name='channel_dropout1')
        self.channel_dense2 = layers.Dense(self.hidden_dim, name='channel_dense2')
        self.channel_dropout2 = layers.Dropout(self.dropout_rate, name='channel_dropout2')

        super().build(input_shape)

    def call(self, x, training=False):
        # Token-mixing
        y = self.norm1(x)
        y = tf.transpose(y, perm=[0, 2, 1])
        y = self.token_dense1(y)
        y = self.token_dropout1(y, training=training)
        y = self.token_dense2(y)
        y = self.token_dropout2(y, training=training)
        y = tf.transpose(y, perm=[0, 2, 1])
        x = x + y

        # Channel-mixing
        y = self.norm2(x)
        y = self.channel_dense1(y)
        y = self.channel_dropout1(y, training=training)
        y = self.channel_dense2(y)
        y = self.channel_dropout2(y, training=training)
        x = x + y

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'hidden_dim': self.hidden_dim,
            'tokens_mlp_dim': self.tokens_mlp_dim,
            'channels_mlp_dim': self.channels_mlp_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

