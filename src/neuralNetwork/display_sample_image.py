import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train / 255.0

def display_sample_image():
    randomint = random.randint(1, 100)
    plt.figure()
    plt.imshow(x_train[randomint], cmap=plt.cm.binary)
    plt.title(f'Sample Image - Label: {y_train[randomint]}')
    plt.colorbar()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    display_sample_image()