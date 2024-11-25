import os 
import tensorflow as tf
from tensorflow import keras 
from keras._tf_keras.keras.preprocessing import image
import numbers as np
from PIL import Image


class DataLoader(tf.data.Dataset):
    def __init__(self, root, fileList, transform=None, is_test=False, batch_size=32, shuffle=True, drop_last=True):
        self.root = root
        self.imgList = self.list_reader(fileList) if is_test else self.default_list_reader(fileList)
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def list_reader(self, fileList):
        imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                imgPath, label, disguise, proPath = line.strip().split(' ')
                if int(disguise) != 0:
                    disguise = 1 
                imgList.append((imgPath, int(label), int(disguise), proPath))
        return imgList
    

    def __new__(cls, root, fileList, transform=None, is_test=False, batch_size=32, shuffle=True, drop_last=True):
        dataset = super(DataLoader, cls).__new__(cls)
        imgList = dataset.imgList

        def generator():
            for imgPath, id_label, disguise_label, proPath in imgList:
                img = Image.open(os.path.join(root, imgPath) + '.bmp')
                img = img.convert('L').convert('RGB')
                if transform:
                    img = transform(img)

                pro = Image.open(os.path.join(root, proPath) + '.bmp')
                pro = pro.convert('L').convert('RGB')
                if transform:
                    pro = transform(pro)

                yield np.array(img, dtype=np.float32), id_label, disguise_label, np.array(pro, dtype=np.float32)

        # Criação do Dataset
        dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32, tf.int32, tf.float32))
        
        # Adicionando o batching e o shuffle
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if drop_last:
            dataset = dataset.take(len(imgList) // batch_size * batch_size)  # Remove o último batch se necessário

        return dataset


# Exemplo de uso:
# dataset = DataLoader('path/to/dataset', 'path/to/filelist.txt', batch_size=32)
# for img, label, disguise, pro in dataset:
#     print(img.shape, label, disguise, pro.shape)

