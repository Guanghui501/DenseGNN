"""Training script for DenseGNN Multimodal V6 (3-way update).

Usage:
    python train_crystal_multimodal_v6_full.py \
        --hyper hyper/hyper_multimodal_v6.py \
        --category DenseGNN_Multimodal_V6 \
        --text_embeddings path/to/text_embeddings.npy \
        --split 8:1:1 \
        --seed 42
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
from datetime import timedelta

import tensorflow as tf
from sklearn.model_selection import KFold

from kgcnn.data.utils import save_pickle_file
from kgcnn.data.transform.scaler.standard import StandardLabelScaler
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError
from kgcnn.training.hyper import HyperParameter
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.utils.plots import plot_predict_true
from kgcnn.utils.devices import set_devices_gpu

from kgcnn.literature.DenseGNN._make_dense_multimodal_v6 import make_model_multimodal_v6


def build_model(hyper, text_dim=768):
    """Build DenseGNN Multimodal V6."""
    model_cfg = hyper["model"]["config"]

    inputs = model_cfg["inputs"].copy()
    inputs["text_embedding"] = {"shape": (text_dim,), "name": "text_embedding", "dtype": "float32", "ragged": False}

    model = make_model_multimodal_v6(
        inputs=inputs,
        name=model_cfg.get("name", "DenseGNN_Multimodal_V6"),
        input_block_cfg=model_cfg.get("input_block_cfg", {}),
        output_block_cfg=model_cfg.get("output_block_cfg", {}),
        input_embedding=model_cfg.get("input_embedding", {}),
        depth=model_cfg.get("depth", 5),
        gin_args=model_cfg.get("gin_args", {}),
        gin_mlp=model_cfg.get("gin_mlp", {}),
        graph_mlp=model_cfg.get("graph_mlp", {}),
        n_units=model_cfg.get("n_units", 128),
        text_embedding_dim=model_cfg.get("text_embedding_dim", text_dim),
        text_projection_dim=model_cfg.get("text_projection_dim", 128),
        graph_projection_dim=model_cfg.get("graph_projection_dim", 128),
        use_middle_fusion=model_cfg.get("use_middle_fusion", True),
        middle_fusion_layers=model_cfg.get("middle_fusion_layers", [2]),
        middle_fusion_cfg=model_cfg.get("middle_fusion_cfg", {}),
        late_fusion_type=model_cfg.get("late_fusion_type", "gated"),
        late_fusion_cfg=model_cfg.get("late_fusion_cfg", {}),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--gpu", default=None, nargs="+", type=int)
    parser.add_argument("--fold", default=None, nargs="+", type=int)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--text_embeddings", default=None)
    parser.add_argument("--random_text_emb", action="store_true")
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("DenseGNN Multimodal V6 (3-way: Node + Edge + Global)")
    print("=" * 60)
    print(f"Args: {args}")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    set_devices_gpu(args.gpu)

    hyper = HyperParameter(hyper_info=args.hyper, hyper_category=args.category)

    print("\nLoading dataset...")
    if "dataset" in hyper._hyper:
        dataset = deserialize_dataset(hyper["dataset"])
    else:
        dataset = deserialize_dataset(hyper["data"]["dataset"])

    original_inputs = hyper["model"]["config"]["inputs"]
    dataset.assert_valid_model_input(original_inputs)
    dataset.clean(original_inputs)
    data_length = len(dataset)
    print(f"Dataset size: {data_length}")

    if args.test:
        print("\n*** TEST MODE: 100 samples, 3 epochs ***")
        test_size = min(100, data_length)
        test_indices = np.random.choice(data_length, test_size, replace=False)
        dataset = dataset[test_indices]
        data_length = len(dataset)

    # Text embeddings
    text_dim = hyper["model"]["config"].get("text_embedding_dim", args.text_dim)
    if args.text_embeddings:
        text_embeddings = np.load(args.text_embeddings).astype(np.float32)
        if args.test:
            text_embeddings = text_embeddings[test_indices]
        text_dim = text_embeddings.shape[1]
    elif args.random_text_emb:
        text_embeddings = np.random.randn(data_length, text_dim).astype(np.float32) * 0.1
    else:
        text_embeddings = np.zeros((data_length, text_dim), dtype=np.float32)

    if len(text_embeddings) > data_length:
        text_embeddings = text_embeddings[:data_length]
    elif len(text_embeddings) < data_length:
        pad = np.zeros((data_length - len(text_embeddings), text_dim), dtype=np.float32)
        text_embeddings = np.concatenate([text_embeddings, pad])
    print(f"Text embeddings: {text_embeddings.shape}")

    labels = np.array(dataset.obtain_property("graph_labels"))
    if len(labels.shape) <= 1:
        labels = np.expand_dims(labels, axis=-1)
    print(f"Labels shape: {labels.shape}")

    # Get charge data (for global state)
    charge_data = dataset.obtain_property("charge")
    if charge_data is None or len(charge_data) == 0:
        print("Warning: No charge data found, using zeros")
        charge_data = np.zeros((data_length, 1), dtype=np.float32)
    else:
        charge_data = np.array(charge_data, dtype=np.float32)
        if len(charge_data.shape) == 1:
            charge_data = np.expand_dims(charge_data, axis=-1)
    print(f"Charge data shape: {charge_data.shape}")

    # Output filepath
    try:
        filepath = hyper.results_file_path()
    except:
        dataset_name = hyper["data"]["dataset"]["class_name"]
        filepath = os.path.join("results", dataset_name)
    postfix = hyper["info"]["postfix_file"]
    filepath = os.path.join(filepath, f"seed_{args.seed}_v6")
    os.makedirs(filepath, exist_ok=True)

    # Save config
    config_to_save = {
        "args": vars(args),
        "model": hyper["model"]["config"],
        "training": hyper["training"],
    }
    with open(os.path.join(filepath, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2, default=str)
    print(f"Config saved to: {filepath}/config.json")

    # Splits
    if args.split:
        ratios = [float(x) for x in args.split.split(':')]
        total = sum(ratios)
        train_ratio = ratios[0] / total
        val_ratio = ratios[1] / total

        indices = np.random.permutation(data_length)
        n_train = int(data_length * train_ratio)
        n_val = int(data_length * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        print(f"Using {args.split} split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        splits = [(train_idx, val_idx, test_idx)]
        execute_folds = [0]
    else:
        kf = KFold(**hyper["training"]["cross_validation"]["config"])
        splits = [(train_idx, None, test_idx) for train_idx, test_idx in kf.split(np.zeros((data_length, 1)), labels)]
        execute_folds = args.fold if args.fold else list(range(len(splits)))

    time_list = []
    for fold_idx, split in enumerate(splits):
        if fold_idx not in execute_folds:
            continue

        train_idx, val_idx, test_idx = split

        print(f"\n{'=' * 60}")
        if args.split:
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        else:
            print(f"FOLD {fold_idx + 1}/{len(splits)}")
            val_idx = test_idx
        print("=" * 60)

        model = build_model(hyper, text_dim=text_dim)

        # Input order: offset, voronoi_ridge_area, atomic_number, AGNIFinger, charge, edge_indices, text
        input_keys = ['offset', 'voronoi_ridge_area', 'atomic_number', 'AGNIFinger', 'edge_indices']
        input_list = [original_inputs[k] for k in input_keys if k in original_inputs]

        x_train_graph = dataset[train_idx].tensor(input_list)
        x_val_graph = dataset[val_idx].tensor(input_list)
        x_test_graph = dataset[test_idx].tensor(input_list)

        # Insert charge before edge_indices, then text at end
        # Graph inputs: [offset, voronoi, atomic_number, AGNIFinger, edge_indices]
        # Final order: [offset, voronoi, atomic_number, AGNIFinger, charge, edge_indices, text]
        x_train = x_train_graph[:-1] + [charge_data[train_idx]] + [x_train_graph[-1]] + [text_embeddings[train_idx]]
        x_val = x_val_graph[:-1] + [charge_data[val_idx]] + [x_val_graph[-1]] + [text_embeddings[val_idx]]
        x_test = x_test_graph[:-1] + [charge_data[test_idx]] + [x_test_graph[-1]] + [text_embeddings[test_idx]]

        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]

        scaler, metrics = None, None
        if "scaler" in hyper["training"]:
            scaler = StandardLabelScaler(**hyper["training"]["scaler"]["config"])
            y_train = scaler.fit_transform(y=y_train)
            y_val = scaler.transform(y=y_val)
            y_test = scaler.transform(y=y_test)
            scale = scaler.get_scaling()
            mae = ScaledMeanAbsoluteError(scale.shape, name="scaled_mae")
            mae.set_scale(scale)
            metrics = [mae]

        model.compile(**hyper.compile(loss="mean_absolute_error", metrics=metrics))
        print(model.summary())

        print("\nTraining...")
        start = time.time()
        fit_kwargs = hyper.fit()
        if args.test:
            fit_kwargs["epochs"] = 3
            fit_kwargs["callbacks"] = []

        best_model_path = os.path.join(filepath, f"best_model{postfix}_fold_{fold_idx}.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_scaled_mae',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        if "callbacks" not in fit_kwargs or fit_kwargs["callbacks"] is None:
            fit_kwargs["callbacks"] = []
        fit_kwargs["callbacks"].append(checkpoint)

        hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), **fit_kwargs)
        elapsed = time.time() - start
        time_list.append(str(timedelta(seconds=elapsed)))
        print(f"Training time: {timedelta(seconds=elapsed)}")

        save_pickle_file(hist.history, os.path.join(filepath, f"history{postfix}_fold_{fold_idx}.pickle"))
        model.save_weights(os.path.join(filepath, f"weights{postfix}_fold_{fold_idx}.h5"))

        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model.load_weights(best_model_path)

        pred = model.predict(x_test)
        true = y_test
        if scaler:
            pred = scaler.inverse_transform(y=pred)
            true = scaler.inverse_transform(y=true)

        mae_val = np.mean(np.abs(pred - true))
        print(f"Test MAE: {mae_val:.4f}")

        model_name = hyper["model"]["config"].get("name", "DenseGNN_Multimodal_V6")
        if "dataset" in hyper._hyper:
            dataset_name = hyper["dataset"].get("class_name", "Dataset")
        else:
            dataset_name = hyper["data"]["dataset"].get("class_name", "Dataset")

        plot_predict_true(pred, true, filepath=filepath,
                         model_name=model_name,
                         dataset_name=dataset_name,
                         file_name=f"predict{postfix}_fold_{fold_idx}.png", show_fig=False)

        # Plot history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist.history['loss'], label='Train Loss')
        if 'val_loss' in hist.history:
            axes[0].plot(hist.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Fold {fold_idx} - Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(hist.history['scaled_mae'], label='Train MAE')
        if 'val_scaled_mae' in hist.history:
            axes[1].plot(hist.history['val_scaled_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'Fold {fold_idx} - MAE (Best: {mae_val:.2f})')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(filepath, f"loss{postfix}_fold_{fold_idx}.png"), dpi=150)
        plt.close()

    print(f"\nDone! Results saved to: {filepath}")


if __name__ == "__main__":
    main()
