import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, Model
from decoder import Decoder

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, output_shape):
        super(Generator, self).__init__()
        self.decoder = Decoder(output_shape)

    def call(self, latent_code, noise_vector):
        combined = tf.concat([latent_code, noise_vector], axis=-1)
        return self.decoder(combined)