# %% # noqa: D100 [markdown]
# # MNIST Exploration Notebook

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

# %%
# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
# dataset info
labels = np.unique(y_test)

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Labels: {labels}, len = {len(labels)}")

# %%
# flatten data
x_train_flat = x_train.reshape(x_train.shape[0], -1)
flat_mnist = pd.DataFrame(x_train_flat)
flat_mnist["label"] = y_train
flat_mnist.head()

# %%
# visualization of random samples
fig, axes = plt.subplots(1, 5, figsize=(6, 6))
indices = np.random.choice(x_train.shape[0], 5, replace=False)
for ax, idx in zip(axes.flatten(), indices):
    ax.imshow(x_train[idx], cmap="gray")
    ax.set_title(int(y_train[idx]))
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# stats of labels
plt.figure(figsize=(6, 3))
plt.hist(y_train, bins=np.arange(-0.5, 10.5, 1), density=True, rwidth=0.8)
plt.xticks(range(10))
plt.xlabel("Label")
plt.ylabel("Relative Frequency")
plt.title("MNIST Label Distribution (y_train)")
plt.show()
