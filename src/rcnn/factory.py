"""factory.py

Constructs Keras models from a **YAML** file with the sequential definition of
sequential layer definition. It also allows the compilation parameters to be specified.

Example file `cnn_mnist.yaml`
----------------------------------
```yaml
model:
  type: sequential
  input_shape: [28, 28, 1]
  layers:
    - Conv2D: {filters: 32, kernel_size: 3, activation: relu}
    - MaxPooling2D: {pool_size: 2}
    - Conv2D: {filters: 64, kernel_size: 3, activation: relu}
    - MaxPooling2D: {pool_size: 2}
    - Flatten: {}
    - Dense: {units: 128, activation: relu}
    - Dense: {units: 10, activation: softmax}

compile:
  optimizer: adam
  loss: sparse_categorical_crossentropy
  metrics: [accuracy]

train:
    epochs: 20
    batch_size: 128
    validation_split: 0.1
    callbacks:
        - EarlyStopping: {monitor: val_loss, patience: 3, restore_best_weights: true}
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from tensorflow import keras
from tensorflow.keras import callbacks, layers

__all__ = ["build_model", "build_model_from_file"]

LAYER_MAPPING: Dict[str, type] = {
    "Input": layers.Input,
    "Conv2D": layers.Conv2D,
    "MaxPooling2D": layers.MaxPooling2D,
    "AveragePooling2D": layers.AveragePooling2D,
    "BatchNormalization": layers.BatchNormalization,
    "Flatten": layers.Flatten,
    "Dense": layers.Dense,
    "Dropout": layers.Dropout,
    # Data‑augmentation
    "RandomRotation": layers.RandomRotation,
    "RandomTranslation": layers.RandomTranslation,
    "RandomZoom": layers.RandomZoom,
}

CALLBACK_MAPPING: Dict[str, type] = {
    "EarlyStopping": callbacks.EarlyStopping,
    "ReduceLROnPlateau": callbacks.ReduceLROnPlateau,
    "ModelCheckpoint": callbacks.ModelCheckpoint,
}

# helpers


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _instantiate_layer(layer_spec: Dict[str, Dict[str, Any]]) -> layers.Layer:
    """Generate a Keras layer from yaml block."""

    if len(layer_spec) != 1:
        raise ValueError(f"Each layer must have a unique type. Received: {layer_spec}")

    layer_name, params = next(iter(layer_spec.items()))
    LayerClass = LAYER_MAPPING.get(layer_name)
    if LayerClass is None:
        raise KeyError(f"Layer '{layer_name}' not supported. Add it to LAYER_MAPPING.")
    return LayerClass(**params)


def _instantiate_callback(cb_spec: Dict[str, Dict[str, Any]]) -> callbacks.Callback:
    if len(cb_spec) != 1:
        raise ValueError(f"Each callback must have a unique type: {cb_spec}")
    cb_name, params = next(iter(cb_spec.items()))
    CbClass = CALLBACK_MAPPING.get(cb_name)
    if CbClass is None:
        raise KeyError(f"Callback '{cb_name}' not supported.")
    return CbClass(**params)


# main functions


def build_model(
    config: Dict[str, Any], x_train=None, y_train=None
) -> keras.Model | Tuple[keras.Model, keras.callbacks.History]:
    """Construct and compiles a model *Sequential* from a dict."""

    model_cfg = config.get("model")
    if model_cfg is None:
        raise KeyError("Missing 'model' section.")
    if model_cfg.get("type", "sequential").lower() != "sequential":
        raise NotImplementedError(
            "Build model have support only for 'sequential' models."
        )

    input_shape = tuple(model_cfg["input_shape"])
    layers_cfg: List[Dict[str, Dict[str, Any]]] = model_cfg["layers"]

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for layer_spec in layers_cfg:
        model.add(_instantiate_layer(layer_spec))

    # Compilation
    compile_cfg = config.get("compile", {})
    model.compile(
        optimizer=compile_cfg.get("optimizer", "adam"),
        loss=compile_cfg.get("loss", "sparse_categorical_crossentropy"),
        metrics=compile_cfg.get("metrics", ["accuracy"]),
    )

    # Optional training
    train_cfg = config.get("train")
    if train_cfg is None:
        return model  # builded and compiled

    # dataset validation
    if x_train is None or y_train is None:
        raise ValueError(
            "If 'train' section appears in `config_file` you must give x_train and y_train."
        )

    # fit parameters
    fit_kwargs = dict(
        epochs=train_cfg.get("epochs", 10),
        batch_size=train_cfg.get("batch_size", 32),
        validation_split=train_cfg.get("validation_split", 0.0),
        shuffle=train_cfg.get("shuffle", True),
    )

    # callbacks
    callbacks_cfg: List[Dict[str, Dict[str, Any]]] = train_cfg.get("callbacks", [])
    fit_kwargs["callbacks"] = [_instantiate_callback(c) for c in callbacks_cfg]

    history = model.fit(x_train, y_train, verbose=1, **fit_kwargs)
    return model, history


def build_model_from_file(
    config_path: str | Path, x_train=None, y_train=None
) -> keras.Model:
    """Loads a YAML and contructs and train (optional) the corresponding model."""
    config = _load_yaml(config_path)
    return build_model(config, x_train=x_train, y_train=y_train)
