import tensorflow as tf
from tensorflow.python.keras import layers


class Decoder(tf.keras.Model):
    def __init__(self, output_shape):
        super(Decoder, self).__init__()
        
        # Inicialização do Dense com RandomNormal
        self.dense = layers.Dense(512 * 4 * 4, use_bias=False,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn_dense = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act_dense = layers.ReLU()  # ReLU não tem o parâmetro alpha como LeakyReLU
        
        self.reshape = layers.Reshape((4, 4, 512))
        
        # Inicialização das camadas Conv2DTranspose com RandomNormal
        self.deconv1 = layers.Conv2DTranspose(256, 4, strides=2, padding='same',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn1 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act1 = layers.ReLU()
        
        self.deconv2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn2 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act2 = layers.ReLU()
        
        self.deconv3 = layers.Conv2DTranspose(64, 4, strides=2, padding='same',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.bn3 = layers.BatchNormalization(
            gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            beta_initializer=tf.keras.initializers.Zeros()
        )
        self.act3 = layers.ReLU()
        
        # Saída final com Tanh
        self.output_layer = layers.Conv2DTranspose(output_shape[-1], 4, strides=2, padding='same', 
                                                   activation='tanh',
                                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn_dense(x)
        x = self.act_dense(x)
        x = self.reshape(x)
        
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        return self.output_layer(x)