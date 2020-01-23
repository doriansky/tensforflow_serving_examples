#see  https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile, shutil
import json
import requests
import random


print(tf.__version__)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show(idx, title):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()


def importFashionData():

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('\ntest_images.shape: {}, of {}'.format(test_images.shape, train_images.dtype))
    return (train_images, train_labels), (test_images, test_labels)

def trainModel(train_images, train_labels, test_images, test_labels, epochs):
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
    ])

    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    model.save('/tmp/fashionModel')
    return model

if __name__=="__main__":

    (train_images, train_labels), (test_images, test_labels) = importFashionData()

    """
    model = trainModel(train_images, train_labels, test_images, test_labels, 10)
    version = 1
    MODEL_DIR = os.path.join('/tmp/fashionModel', str(version))
    os.environ["MODEL_DIR"] = MODEL_DIR
    model.save(MODEL_DIR)
    """

    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:5].tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = json.loads(json_response.text)['predictions']

    for i in range(0,5):
        show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
        class_names[np.argmax(predictions[i])], test_labels[i], class_names[np.argmax(predictions[i])], test_labels[i]))


"""
1. Examine the saved model :
            saved_model_cli show --dir /tmp/1 --all

2. Start running TF Serving and load the model:
             nohup tensorflow_model_server   --rest_api_port=8501   --model_name=fashion_model   --model_base_path="/tmp/fashionModel" >server.log 2>&1

"""

