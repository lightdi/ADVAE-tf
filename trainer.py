import tensorflow as tf
import numpy as np
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from .models.decoder import Decoder
from .models.encoder import Encoder
from .models.generator import Generator
from .models.discriminator import Discriminator
from tensorboardX import SummaryWriter

class Trainer:

    def __init__(self, latent_dim, id_dim, dataset, lr, b1, b2):
        self.E = Encoder(latent_dim=latent_dim)
        self.D = Decoder(output_shape=(64, 64, 3)) 
        self.G = Generator(output_shape=(64, 64, 3))
        self.C = Discriminator(Nd=id_dim) 

        #loss functions
        self.loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_criterion_gan = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        #log
        # TensorBoard setup
        self.log_dir = "logs/train"
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)


        # Otimizadores
        self.optimizer_E = optimizers.Adam(learning_rate=lr, 
                                            beta_1=b1, beta_2=b2)
        self.optimizer_G = optimizers.Adam(learning_rate=lr, 
                                            beta_1=b1, beta_2=b2)
        self.optimizer_D = optimizers.Adam(learning_rate=lr, 
                                            beta_1=b1, beta_2=b2)
        self.optimizer_C = optimizers.Adam(learning_rate=lr, 
                                            beta_1=b1, beta_2=b2)

    def encoder_loss(self, x_real, eps):
        z_real_mu, z_real_log_sigma = self.E(x_real)
        z_real = z_real_mu + tf.exp(z_real_log_sigma) * eps
        x_real_mu = self.D(z_real)

        # Calculando a KL Divergence (loss KL)
        kl_loss = tf.reduce_mean(tf.reduce_sum(
            0.5 * (tf.square(z_real_mu) + tf.exp(2 * z_real_log_sigma) - 1 - 2 * z_real_log_sigma),
            axis=1
        ))
        # Calculando a Log-Likelihood Loss (perda de reconstrução)
        ll_loss = tf.reduce_mean(tf.reduce_sum(
            0.5 * tf.square((x_real - x_real_mu) / 1), axis=(1, 2, 3)
        ))

        encoder_loss = kl_loss + ll_loss

        return encoder_loss

    def decoder_loss(self, x_real, eps):
        z_real_mu, z_real_log_sigma = self.E(x_real)
        z_real = z_real_mu + tf.exp(z_real_log_sigma) * eps
        x_real_mu = self.D(z_real)

        ll_loss = tf.reduce_mean(tf.reduce_sum(
            0.5 * tf.square((x_real - x_real_mu) / 1), axis=(1, 2, 3)
        ))

        return ll_loss, x_real_mu

    def generator_loss(self, z, x_real, x_real_pro,  x_id_label, x_var, eps):
        o1, o2, o3 = 5, 0.5, 0.1

        z_real_mu, z_real_log_sigma = self.E(x_real)
        c = z_real_mu + tf.exp(z_real_log_sigma) * eps
        x_pro = self.G(tf.concat([c, z], axis=1))

        z_real_pro_mu, z_real_pro_log_sigma = self.E(x_real_pro)
        c_pro= z_real_pro_mu + tf.exp(z_real_pro_log_sigma) * eps
        x_pro_real = self.G(tf.concat([c_pro, z], axis=1))



        d_pro = self.C(x_pro)

        gan_loss = self.loss_criterion_gan(d_pro[:, self.Nd], tf.ones_like(d_pro[:, self.Nd]))  # Perda GAN

        id_loss = self.loss_criterion(d_pro[:, :self.Nd], x_id_label)
        
        var_loss = self.loss_criterion(d_pro[:, self.Nd + 1], tf.zeros_like(d_pro[:, self.Nd + 1])) 

        Index = tf.where(x_var == 0)[:, 0]

        rec_loss = tf.reduce_sum(tf.square(x_pro_real[Index] - x_pro[Index])) / tf.size(Index)

        gen_loss = gan_loss + o1 * id_loss + o2 * var_loss + o3 * rec_loss

        w = tf.nn.softmax(-2 * z_real_log_sigma[Index], axis=0)
        lat_loss = 0.5 * tf.reduce_sum(tf.square(c[Index] - z_real_mu[Index]) * w)


        g_loss = gen_loss + lat_loss

        return g_loss, x_pro




    def critic_loss(self, x_real, X_id_label,x_var, x_pro, z, eps):
        
        u1, u2 = 5, 0.5

        z_real_mu, z_real_log_sigma = self.E(x_real)
        c = z_real_mu + tf.exp(z_real_log_sigma) * eps

        x_fake = self.G(tf.concat([c, z], axis=1))

        d_pro = self.C(x_pro)
        d_fake = self.C(x_fake)
        d_real = self.C(x_real)

        critic_loss = (self.loss_criterion_gan(d_pro[:, self.Nd], self.batch_ones_label) +
                       self.loss_criterion_gan(d_fake[:, self.Nd], self.batch_zeros_label))

        id_loss = self.loss_criterion(d_real[:, :self.Nd], X_id_label)

        var_loss = self.loss_criterion(d_real[:, self.Nd + 1], x_var + 0.0)

        critic_loss = critic_loss + u1 * id_loss + u2 * var_loss
        
        return critic_loss



    def train_step(self, batch_image, 
                    batch_id_label, batch_var, batch_pro):
        x_real = batch_image
        x_id_label = batch_id_label
        x_var = batch_var
        x_pro = batch_pro

        # Gerar ruído para o latent code
        eps = tf.random.normal([self.batch_size, self.latent_dim])
        z = tf.random.normal([self.batch_size, self.noise_dim])

        # Treinando o Encoder
        with tf.GradientTape() as tape_E:
            e_loss = self.encoder_loss(x_real, eps)
        gradients_E = tape_E.gradient(e_loss, self.E.trainable_variables)
        self.optimizer_E.apply_gradients(zip(gradients_E, self.E.trainable_variables))

        with tf.GradientTape() as tape_D:
            d_loss, x_decoded = self.decoder_loss(x_real, eps)
        gradients_D = tape_D.gradient(d_loss, self.D.trainable_variables)
        self.optimizer_D.apply_gradients(zip(gradients_D, self.D.trainable_variables))

        with tf.GradientTape() as tape_G:
            g_loss, x_generated = self.generator_loss(z, x_real, x_pro, x_id_label, x_var, eps)
        gradients_G = tape_G.gradient(g_loss, self.G.trainable_variables)
        self.optimizer_G.apply_gradients(zip(gradients_G, self.G.trainable_variables))

        with tf.GradientTape() as tape_C:
            c_loss = self.critic_loss(x_real, x_id_label, x_var, x_pro, z, eps)
        gradients_C = tape_C.gradient(c_loss, self.C.trainable_variables)
        self.optimizer_C.apply_gradients(zip(gradients_C, self.C.trainable_variables))

        return g_loss, d_loss, e_loss, c_loss, x_real, x_decoded, x_pro, x_generated


    def train(self, epochs):
        for epoch in range(epochs):
            for batch in self.dataset:
                batch_image, batch_id_label, batch_var, batch_pro = batch
                # Realiza um passo de treinamento
                g_loss, d_loss, e_loss, c_loss, x_real, x_decoded, x_pro, x_generated = self.train_step(batch_image, batch_id_label, batch_var, batch_pro)

            # Usando TensorBoard para logar as métricas e imagens
            with self.summary_writer.as_default():
                tf.summary.scalar('g_loss', g_loss, step=epoch)
                tf.summary.scalar('d_loss', d_loss, step=epoch)
                tf.summary.scalar('e_loss', e_loss, step=epoch)
                tf.summary.scalar('c_loss', c_loss, step=epoch)
                
                # Logando imagens geradas
                tf.summary.image('Generated Images', x_generated, step=epoch, max_images=5)
                tf.summary.image('Decoded Images', x_decoded, step=epoch, max_images=5)
            print(f"Epoch: {epoch+1}, G_loss: {g_loss}, D_loss: {d_loss}, E_loss: {e_loss}, C_loss: {c_loss}")
