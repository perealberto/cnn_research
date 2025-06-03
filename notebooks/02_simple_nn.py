# %% [markdown]
# # Simple Neural Network

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import polars as pl

# %%
# import mnist dataset
from rcnn.datasets import get_mnist_data

(x_train, y_train), (x_test, y_test) = get_mnist_data(flatten=True)

# %%
# create a simple model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# %%
# training
epochs = 10
batch_size = 128

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)

# %%
# results
from rcnn.visualization import (
    plot_history,
    history_to_polars,
    print_classification_report,
    plot_confusion_matrix,
)

y_pred = model.predict(x_test, batch_size=batch_size)

history_to_polars(history.history).select(pl.last(history.history.keys()))

# %%
plot_history(history.history, separate=True)

# %%
print_classification_report(y_test, y_pred)

# %%
plot_confusion_matrix(y_test, y_pred, normalize=True)
