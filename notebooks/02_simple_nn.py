# %% [markdown]
# # Simple Neural Network

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# %%
# import mnist dataset
from rcnn.data.datasets import get_mnist_data

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
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Results:\n - Accuracy: {acc:.4f}\n - Loss: {loss:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
