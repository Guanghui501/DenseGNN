"""Hyperparameter configuration for DenseGNN Multimodal V6.

V6 uses full 3-way update (Node + Edge + Global) with text fusion.
Requires 'charge' input for global state initialization.
"""

from copy import deepcopy

n_units = 128
depth = 5

input_block_cfg = {
    'node_size': n_units,
    'edge_size': n_units,
    'atomic_mass': True,
    'atomic_radius': True,
    'electronegativity': True,
    'ionization_energy': True,
    'oxidation_states': True,
    'melting_point': True,
    'density': True,
    'edge_embedding_args': {
        'bins_distance': 32,
        'max_distance': 8.0,
        'distance_log_base': 1.0,
        'bins_voronoi_area': 25,
        'max_voronoi_area': 32
    }
}

output_block_cfg = {
    'edge_mlp': None,
    'node_mlp': None,
    'global_mlp': {'units': [n_units, 64, 1], 'activation': ['swish', 'swish', 'linear']},
    'aggregate_edges_local': 'sum',
    'aggregate_edges_global': None,
    'aggregate_nodes': 'mean',
    'return_updated_edges': False,
    'return_updated_nodes': True,
    'return_updated_globals': True,
    'edge_attention_mlp_local': {'units': [32, 1], 'activation': ['swish', 'swish']},
    'node_attention_mlp': {'units': [32, 1], 'activation': ['swish', 'swish']},
}

# DenseGNN conv args (3-way update)
gin_args = {
    'pooling_method': 'sum',
    'g_pooling_method': 'mean',
    'activation': 'swish',
    'edge_mlp_args': {'units': [n_units] * 2, 'activation': ['swish', 'swish']},
    'node_mlp_args': {'units': [n_units] * 2, 'activation': ['swish', 'swish']},
    'graph_mlp_args': {'units': [n_units] * 2, 'activation': ['swish', 'swish']},
    'concat_args': {'axis': -1},
}

gin_mlp = {'units': [n_units], 'activation': ['swish']}
graph_mlp = {'units': [n_units], 'activation': ['swish']}

hyper = {
    "DenseGNN_Multimodal_V6": {
        "model": {
            "class_name": "make_model_multimodal_v6",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN_Multimodal_V6",
                "inputs": {
                    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                    "voronoi_ridge_area": {"shape": (None,), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                    "AGNIFinger": {"shape": (None, 24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    "charge": {"shape": [1], "name": "charge", "dtype": "float32", "ragged": False},
                },
                "input_block_cfg": input_block_cfg,
                "output_block_cfg": output_block_cfg,
                "input_embedding": {"graph": {"input_dim": 95, "output_dim": n_units}},
                "depth": depth,
                "gin_args": gin_args,
                "gin_mlp": gin_mlp,
                "graph_mlp": graph_mlp,
                "n_units": n_units,
                # Multimodal config
                "text_embedding_dim": 768,
                "text_projection_dim": 128,
                "graph_projection_dim": 128,
                "use_middle_fusion": True,
                "middle_fusion_layers": [2],
                "middle_fusion_cfg": {
                    "num_heads": 0,
                    "use_gate_norm": True,
                    "use_learnable_scale": True,
                    "initial_scale": 1.0,
                    "dropout": 0.1,
                },
                "late_fusion_type": "gated",
                "late_fusion_cfg": {"output_dim": 128, "dropout": 0.1},
            },
        },
        "training": {
            "fit": {
                "batch_size": 256,
                "epochs": 100,
                "validation_freq": 1,
                "verbose": 2,
                "callbacks": [],
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "learning_rate": {
                            "class_name": "ExponentialDecay",
                            "config": {
                                "initial_learning_rate": 0.001,
                                "decay_steps": 5800,
                                "decay_rate": 0.5,
                                "staircase": False,
                            },
                        },
                    },
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {
                "class_name": "KFold",
                "config": {"n_splits": 5, "random_state": 42, "shuffle": True},
            },
            "scaler": {
                "class_name": "StandardScaler",
                "config": {"with_std": True, "with_mean": True, "copy": True},
            },
        },
        "data": {
            "dataset": {
                "class_name": "JarvisMaxEfgDataset",
                "module_name": "kgcnn.data.datasets.JarvisMaxEfgDataset",
                "config": {},
                "methods": [
                    {
                        "set_representation": {
                            "pre_processor": {
                                "class_name": "VoronoiUnitCell",
                                "module_name": "kgcnn.crystal.preprocessor",
                                "config": {"min_ridge_area": 0.1},
                            },
                            "reset_graphs": False,
                        }
                    },
                ],
            },
            "data_unit": "",
        },
        "info": {
            "postfix": "",
            "postfix_file": "_DenseGNN_multimodal_v6",
            "kgcnn_version": "4.0.0",
        },
    },
}
