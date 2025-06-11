# %% [markdown]
# # Saliency Maps

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from rcnn.factory import build_model_from_file
from rcnn.datasets import get_mnist_data
from rcnn.predictions import confusion_scores, compute_saliency, average_saliency

# dataset
(x_train, y_train), (x_test, y_test) = get_mnist_data()

# %% [markdown]
# ## Concept of Saliency Map

# %%
# load and train model
config_file = "../src/models/simple_cnn.yaml"
model, _ = build_model_from_file(config_file)

# %%
# predict a selected image
# error -> 18
idx = 30
image = x_test[idx]
true_label = y_test[idx]

entry = tf.keras.preprocessing.image.img_to_array(image)
entry = entry.reshape((1, *entry.shape))

# calculate saliency map
img_var = tf.Variable(entry)
target_cls = np.argmax(pred)
with tf.GradientTape() as tape:
    logits = model(img_var, training=False)
    score = logits[0, target_cls]
grads = tape.gradient(score, img_var)
sal = tf.reduce_max(tf.abs(grads), axis=-1)[0]
sal = (sal - tf.reduce_min(sal)) / (tf.reduce_max(sal) - tf.reduce_min(sal) + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(image, cmap="gray")
i = axes[1].imshow(sal, cmap="jet", alpha=0.6)
fig.colorbar(i)
axes[0].set_axis_off()
axes[1].set_axis_off()
plt.show()

# %% [markdown]
# ## Display of the N most confusing

# %%
# load and train model
print("TRAINING MODEL")
config_file = "../src/models/simple_cnn_train.yaml"
model, _ = build_model_from_file(config_file)

# %%
# make predictions
print("MAKING PREDICTIONS")
probs = model.predict(x_test, verbose=1)
# probs = tf.nn.softmax(probs, axis=1)

print("RESULTS...")
# confusion ranking
wrong_sorted, scores = confusion_scores(probs, y_test)

# visualization N most confusing
N = 10
plt.figure(figsize=(15, 2.2 * N))
for rank, (idx, sc) in enumerate(zip(wrong_sorted[-N:], scores[-N:])):
    img = x_test[idx]
    pred = np.argmax(probs[idx])
    true = y_test[idx]

    # original image
    plt.subplot(N, 3, 3 * rank + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Rank {rank+1} ({idx})\nTrue {true}  Pred {pred}\nScore {sc:.2f}")

    # dist
    plt.subplot(N, 3, 3 * rank + 2)
    plt.bar(np.arange(10), probs[idx])
    plt.xticks(np.arange(10))
    plt.title("Distribution")

    sal_map = compute_saliency(model, x_test[idx], pred)
    plt.subplot(N, 3, 3 * rank + 3)
    plt.imshow(img, cmap="gray")
    plt.imshow(sal_map, cmap="jet", alpha=0.6)
    plt.colorbar()
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
i = 1033
print(probs[i])
plt.bar(np.arange(10), probs[i])
plt.xticks(np.arange(10))
plt.show()

# %%

# %% [markdown]
# ## Average Saliency Map

# %%
# standard model
std_model, _ = build_model_from_file(
    "../src/models/standard_cnn_train.yaml", x_train, y_train
)
plt.figure(figsize=(15, 6))
for n in tqdm(range(10)):
    avg_sal, n_indices = average_saliency(std_model, x_test, y_test, n)
    ax = plt.subplot(2, 5, n + 1)
    ax.imshow(avg_sal, cmap="jet")
    ax.set_title(f"Digit {n}\n(n={n_indices})", fontsize=10)
    ax.axis("off")
plt.suptitle("Average Saliency Maps Per Digit (Test set)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# complex model
complex_model = tf.keras.models.load_model("../src/models/saved/overfit_cnn.keras")
plt.figure(figsize=(15, 6))
for n in tqdm(range(10)):
    avg_sal, n_indices = average_saliency(complex_model, x_test, y_test, n)
    ax = plt.subplot(2, 5, n + 1)
    ax.imshow(avg_sal, cmap="jet")
    ax.set_title(f"Digit {n}\n(n={n_indices})", fontsize=10)
    ax.axis("off")
plt.suptitle("Average Saliency Maps Per Digit (Test set)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# simple2 model
simple2_model, _ = build_model_from_file(
    "../src/models/underfit_cnn.yaml", x_train, y_train
)
plt.figure(figsize=(15, 6))
for n in tqdm(range(10)):
    avg_sal, n_indices = average_saliency(simple2_model, x_test, y_test, n)
    ax = plt.subplot(2, 5, n + 1)
    ax.imshow(avg_sal, cmap="jet")
    ax.set_title(f"Digit {n}\n(n={n_indices})", fontsize=10)
    ax.axis("off")
plt.suptitle("Average Saliency Maps Per Digit (Test set)", fontsize=14)
plt.tight_layout()
plt.show()
