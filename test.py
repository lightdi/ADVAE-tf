import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Verificar a versão do TensorFlow
print(f"TensorFlow versão: {tf.__version__}")

print("Versão do cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])

print("Versão do CUDA:", tf.sysconfig.get_build_info()["cuda_version"])

# Verificar se há GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        print(f"Detalhes da GPU: {gpu}")
else:
    print("Nenhuma GPU encontrada. O TensorFlow está utilizando a CPU.")

# Testar se as operações podem ser executadas na GPU
try:
    with tf.device('/GPU:0'):  # Força a execução na GPU
        print("Executando operação na GPU...")
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
        print("Operação concluída na GPU.")
except RuntimeError as e:
    print(f"Erro ao executar na GPU: {e}")

