import tensorflow as tf
from tensorflow.python.keras import layers


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        # Inicialização das camadas Conv2D com distribuição normal
        self.conv1 = layers.Conv2D(64, 4, strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn1 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act1 = layers.LeakyReLU(alpha=0.2)
        
        self.conv2 = layers.Conv2D(128, 4, strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn2 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act2 = layers.LeakyReLU(alpha=0.2)
        
        self.conv3 = layers.Conv2D(256, 4, strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn3 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act3 = layers.LeakyReLU(alpha=0.2)
        
        self.flatten = layers.Flatten()
        
        # Inicialização das camadas Dense com distribuição normal
        self.mu = layers.Dense(latent_dim,
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.sigma = layers.Dense(latent_dim,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.flatten(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma