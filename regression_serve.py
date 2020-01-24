# see https://www.tensorflow.org/tutorials/keras/regression

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import json, requests

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print (tf.__version__)


def importAutoMPGdataset():
    # Download the AutoMPG dataset
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    # Import it using Pandas
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()

    # Clean the unknown values
    dataset.isna().sum()
    dataset = dataset.dropna()
    # Convert "origin" column to a one-hot (it's categorical, not numerical)
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    # Split into train and test datasets
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Overall statistics
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    # Separate features from labels
    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")

    #Normalize
    normed_train_data = (train_dataset-train_stats['mean'])/train_stats['std']
    normed_test_data = (test_dataset-train_stats['mean'])/train_stats['std']

    return (normed_train_data, train_labels), (normed_test_data, test_labels)


#Build the model
def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

    return model

if __name__=="__main__":

    (normed_train_data, train_labels), (normed_test_data, test_labels) = importAutoMPGdataset()
    #sns.pairplot(normed_train_data[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    model = build_model(normed_train_data)
    model.summary()

    EPOCHS=1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    early_history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Early Stopping': early_history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE[MPG]')
    plt.show()

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print('Testing set Mean abs error: {:5.2f} MPG'.format(mae))

    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

    #Save model
    version = 1
    MODEL_DIR = os.path.join('/home/dorian/work/src/PycharmProjects/tf_tests/models/AutoMPGmodel', str(version))
    os.environ["MODEL_DIR"] = MODEL_DIR
    model.save(MODEL_DIR)

    #Start TFS server
    # nohup tensorflow_model_server - -rest_api_port = 8501 - -model_name = autoMPG - -model_base_path = "/home/dorian/work/src/PycharmProjects/tf_tests/models/AutoMPGmodel" > server.log 2 > & 1

    #Serve
    data = json.dumps({"signature_name": "serving_default", "instances": normed_test_data[0:5].values.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/autoMPG:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = json.loads(json_response.text)['predictions']