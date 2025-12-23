"""Training example for multimodal DenseGNN.

Usage:
    python train_multimodal_example.py --data_path your_data.json --target formation_energy
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from _make_dense_multimodal import make_model_multimodal, GraphNetworkConfigurator
from data_loader_multimodal import MultimodalDataLoader, create_example_data


def create_model(node_dim=64, edge_dim=64, depth=4, output_dim=1):
    """Create multimodal DenseGNN model."""

    # Input configuration
    inputs = {
        'edge_indices': {'shape': (None, 2), 'dtype': 'int32', 'ragged': True, 'name': 'edge_indices'},
        'atomic_number': {'shape': (None,), 'dtype': 'int32', 'ragged': True, 'name': 'atomic_number'},
        'offset': {'shape': (None, 3), 'dtype': 'float32', 'ragged': True, 'name': 'offset'},
    }

    # Input block config
    input_block_cfg = {
        'node_size': node_dim,
        'edge_size': edge_dim,
        'atomic_mass': True,
        'atomic_radius': True,
        'electronegativity': True,
    }

    # GIN layer config
    gin_args = {
        'units': node_dim,
        'pooling_method': 'sum',
    }

    gin_mlp = {
        'units': [node_dim],
        'activation': 'swish'
    }

    # Output MLP
    output_mlp = {
        'units': [64, 32, output_dim],
        'activation': ['swish', 'swish', 'linear']
    }

    # Pooling config
    g_pooling_args = {'pooling_method': 'mean'}

    # Middle fusion config
    middle_fusion_cfg = {
        'hidden_dim': 128,
        'dropout': 0.1,
        'use_gate_norm': False,
        'use_learnable_scale': False
    }

    # Cross-modal attention config
    cross_modal_cfg = {
        'hidden_dim': 256,
        'num_heads': 4,
        'dropout': 0.1
    }

    # Late fusion config
    late_fusion_cfg = {
        'output_dim': 64,
        'dropout': 0.1
    }

    model = make_model_multimodal(
        inputs=inputs,
        name='DenseGNN_Multimodal',
        input_block_cfg=input_block_cfg,
        depth=depth,
        gin_args=gin_args,
        gin_mlp=gin_mlp,
        output_mlp=output_mlp,
        g_pooling_args=g_pooling_args,
        text_embedding_dim=768,
        text_projection_dim=64,
        graph_projection_dim=64,
        use_middle_fusion=True,
        middle_fusion_layers=[2],
        middle_fusion_cfg=middle_fusion_cfg,
        use_cross_modal_attention=True,
        cross_modal_cfg=cross_modal_cfg,
        late_fusion_type='gated',
        late_fusion_cfg=late_fusion_cfg,
    )

    return model


def train_step(model, optimizer, batch, loss_fn):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Prepare inputs
        inputs = [
            batch['offset'],
            batch['atomic_number'],
            batch['edge_indices'],
            batch['text_embedding']
        ]
        predictions = model(inputs, training=True)
        loss = loss_fn(batch['target'], predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def evaluate(model, dataset, loss_fn):
    """Evaluate model on dataset."""
    total_loss = 0
    total_mae = 0
    n_batches = 0

    for batch in dataset:
        inputs = [
            batch['offset'],
            batch['atomic_number'],
            batch['edge_indices'],
            batch['text_embedding']
        ]
        predictions = model(inputs, training=False)
        loss = loss_fn(batch['target'], predictions)
        mae = tf.reduce_mean(tf.abs(batch['target'] - predictions))

        total_loss += loss.numpy()
        total_mae += mae.numpy()
        n_batches += 1

    return total_loss / n_batches, total_mae / n_batches


def main(args):
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading data...")
    if args.data_path:
        loader = MultimodalDataLoader(
            data_path=args.data_path,
            target_col=args.target,
            text_col=args.text_col,
            atoms_col=args.atoms_col,
            batch_size=args.batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=args.seed
        )
    else:
        # Use example data for testing
        print("Using example data...")
        df = create_example_data()
        loader = MultimodalDataLoader(
            data_df=df,
            target_col='target',
            text_col='text',
            atoms_col='atoms',
            batch_size=args.batch_size
        )

    train_ds, val_ds, test_ds = loader.get_datasets()

    # Create model
    print("Creating model...")
    model = create_model(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        depth=args.depth
    )
    model.summary()

    # Optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Training loop
    print("\nStarting training...")
    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_losses = []
        for batch in train_ds:
            loss = train_step(model, optimizer, batch, loss_fn)
            train_losses.append(loss.numpy())

        avg_train_loss = np.mean(train_losses)

        # Validate
        val_loss, val_mae = evaluate(model, val_ds, loss_fn)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            if args.save_path:
                model.save_weights(args.save_path)
                print(f"  Saved best model (MAE: {val_mae:.4f})")

    # Test
    print("\nEvaluating on test set...")
    if args.save_path and Path(args.save_path + '.index').exists():
        model.load_weights(args.save_path)

    test_loss, test_mae = evaluate(model, test_ds, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file (CSV/JSON)')
    parser.add_argument('--target', type=str, default='target',
                        help='Target column name')
    parser.add_argument('--text_col', type=str, default='text',
                        help='Text column name')
    parser.add_argument('--atoms_col', type=str, default='atoms',
                        help='Atoms column name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='best_model')

    args = parser.parse_args()
    main(args)
