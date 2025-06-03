# %% [markdown]
# # Saliency Maps

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rcnn.factory import build_model_from_file
from rcnn.datasets import get_mnist_data

# %%
# MNIST dataset
(x_train, y_train), (x_test, y_test) = get_mnist_data()

# load and train model
config_file = "../src/models/simple_cnn.yaml"
model = build_model_from_file(config_file)

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# %%
# predict a selected image
# error -> 18
idx = 30
image = x_test[idx]
true_label = y_test[idx]

entry = tf.keras.preprocessing.image.img_to_array(image)
entry = entry.reshape((1, *entry.shape))
entry.shape

pred = model(entry, training=False)[0]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"Number: {true_label}")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
classes = np.arange(10)
plt.xticks(classes)
plt.bar(classes, pred, width=0.8)
plt.ylabel("Probability")
plt.title(f"Prediction: {np.argmax(pred)}")

plt.tight_layout()
plt.show()

# %%
# calculate saliency map
entry = tf.Variable(entry, dtype=float)

with tf.GradientTape() as tape:
    pred = model(entry, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

grads = tape.gradient(loss, entry)
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(image, cmap="gray")
i = axes[1].imshow(grad_eval, cmap="jet", alpha=0.8)
fig.colorbar(i)
plt.show()

# %% [markdown]
# # todo
# - recoger todas las fotos que se clasifican mal y ordenarlas por el mayor nivel de confusión
# - probar con diferentes configuraciones del model (overfitted, underfitted, normal) y el mismo número a ver si se reflejan cambios en los píxeles

# %%
