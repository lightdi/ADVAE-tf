import os 
import tensorflow as tf
from utils.dataloader import DataLoader
from keras._tf_keras.keras.preprocessing import image
from trainer import Trainer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#GPU
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Usando dispositivo: {device_name}")

#Inicializando o treino do Modelo
latent_dim = 100
id_dim = 100
batch_size = 16
lr = 0.0002
b1 = 0.5
b2 = 0.999

dataset = DataLoader(
    '/media/lightdi/CRUCIAL/Datasets/AR-Cropped/',
    'dataset_file/Load_AR_training_50_0.txt',
    transform=image.ImageDataGenerator(
                                     rescale=1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True),
                                     is_test=False,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True)

trainer = Trainer(latent_dim=latent_dim, 
                    id_dim=id_dim, 
                    batch_size=batch_size,
                    dataset=dataset, lr=lr, b1=b1, b2=b2)

trainer.train(100)

