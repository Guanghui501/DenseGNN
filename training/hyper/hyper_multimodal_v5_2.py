"""Hyperparameter configuration for multimodal DenseGNN V5_2.

Graph structure IDENTICAL to original DenseGNN.
Same gin_args, same pooling, same output_mlp.

Usage:
    python train_crystal_multimodal_v6.py --hyper hyper/hyper_multimodal_v5_2.py --category DenseGNN_Multimodal_V5_2 --fold 0
"""

hyper = {
    "DenseGNN_Multimodal_V5_2": {
        "model": {
            "class_name": "make_model_multimodal_v5",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN_Multimodal_V5_2",
                "inputs": {
                    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                    "voronoi_ridge_area": {"shape": (None,), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                    "AGNIFinger": {"shape": (None, 24), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                },
                "input_block_cfg": {
                    "node_size": 128,
                    "edge_size": 128,
                    "atomic_mass": True,
                    "atomic_radius": True,
                    "electronegativity": True,
                    "ionization_energy": True,
                    "oxidation_states": True,
                    "melting_point": True,
                    "density": True,
                    "edge_embedding_args": {
                        "bins_distance": 32,
                        "max_distance": 8.0,
                        "distance_log_base": 1.0,
                        "bins_voronoi_area": 25,
                        "max_voronoi_area": 32,
                    },
                },
                "input_embedding": {
                    "node": {"input_dim": 96, "output_dim": 64},
                    "graph": {"input_dim": 100, "output_dim": 64},
                },
                "depth": 5,
                # Same as original DenseGNN
                "gin_mlp": {"units": [128], "use_bias": True, "activation": ["swish"]},
                "gin_args": {
                    "pooling_method": "mean",
                    "edge_mlp_args": {"units": [128], "use_bias": True, "activation": ["swish"]},
                    "concat_args": {"axis": -1},
                },
                "g_pooling_args": {"pooling_method": "mean"},
                "output_mlp": {
                    "use_bias": [True, True, False],
                    "units": [128, 64, 1],
                    "activation": ["swish", "swish", "linear"],
                },
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
                "class_name": "JarvisOptb88vdwBandGapDataset",
                "module_name": "kgcnn.data.datasets.JarvisOptb88vdwBandGapDataset",
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
            "data_unit": "GPa",
        },
        "info": {
            "postfix": "",
            "postfix_file": "_multimodal_v5_2",
            "kgcnn_version": "4.0.0",
        },
    },
}
