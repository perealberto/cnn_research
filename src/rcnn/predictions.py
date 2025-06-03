"""predictions.py

Utility functions for computing scores and other metrics to test model performance.

Example
-------
>>> from rcnn.predictions import *
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

__all__ = ["confusion_scores", "compute_saliency"]


def confusion_scores(
    probs: tf.Tensor, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates confusion score over each bad prediction.
    score = p_pred - p_true (the higher, the more misconfidence).

    Returns
    -------
    tuple with sorted_indices (np.ndarray) and sorted_scores (np.ndarray).
    """

    pred_cls = tf.argmax(probs, axis=1, output_type=np.int64)
    wrong_mask = tf.not_equal(pred_cls, y_true)
    wrong_indices = tf.where(wrong_mask)[:, 0]
    y_true = tf.cast(y_true, dtype=np.int64)

    p_pred = tf.gather_nd(
        probs, tf.stack([wrong_indices, tf.gather(pred_cls, wrong_indices)], axis=1)
    )
    p_true = tf.gather_nd(
        probs, tf.stack([wrong_indices, tf.gather(y_true, wrong_indices)], axis=1)
    )
    scores = p_pred - p_true

    sorted_order = tf.argsort(scores, direction="DESCENDING")
    return (
        wrong_indices.numpy()[sorted_order.numpy()],
        scores.numpy()[sorted_order.numpy()],
    )


def compute_saliency(
    model: tf.keras.Model, image_batch: np.ndarray, target_class: int
) -> np.ndarray:
    """Returns a normalized saliency map [0,1] for the target class."""
    entry = tf.keras.preprocessing.image.img_to_array(image_batch)
    entry = entry.reshape((1, *entry.shape))
    img_var = tf.Variable(entry)
    with tf.GradientTape() as tape:
        logits = model(img_var, training=False)
        score = logits[0, target_class]
    grads = tape.gradient(score, img_var)
    sal = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    sal = (sal - tf.reduce_min(sal)) / (tf.reduce_max(sal) - tf.reduce_min(sal) + 1e-9)
    return sal
