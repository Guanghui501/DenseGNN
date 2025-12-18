"""
Hyperparameter configuration for training DenseGNN on JARVIS mbj_bandgap dataset.
Dataset: JarvisMbjBandgapDataset
Task: Predict bandgap using modified Becke-Johnson (mBJ) functional
"""

hyper = {
    "DenseGNN": {
        "model": {
            "class_name": "make_model_asu",
            "module_name": "kgcnn.literature.DenseGNN",
            "config": {
                "name": "DenseGNN",
                "inputs": {
                    "offset": {"shape": (None, 3), "name": "offset", "dtype": "float32", "ragged": True},
                    "voronoi_ridge_area": {"shape": (None, ), "name": "voronoi_ridge_area", "dtype": "float32", "ragged": True},
                    "atomic_number": {"shape": (None,), "name": "atomic_number", "dtype": "int32", "ragged": True},
                    "AGNIFinger": {"shape": (None, 128), "name": "AGNIFinger", "dtype": "float32", "ragged": True},
                    "edge_indices": {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    "charge": {'shape': [1], 'name': "charge", 'dtype': 'float32', 'ragged': False},
                },

                "input_block_cfg": {
                    'node_size': 128,
                    'edge_size': 128,
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
                },

                "output_block_cfg": {
                    'edge_mlp': None,
                    'node_mlp': None,
                    'global_mlp': {
                        'units': [1],
                        'activation': ['linear']
                    },
                    'aggregate_edges_local': 'sum',
                    'aggregate_edges_global': 'mean',
                    'aggregate_nodes': 'mean',
                    'return_updated_edges': False,
                    'return_updated_nodes': True,
                    'return_updated_globals': True,
                    'edge_attention_mlp_local': {
                        'units': [32, 1],
                        'activation': ['swish', 'swish']
                    },
                    'edge_attention_mlp_global': {
                        'units': [32, 1],
                        'activation': ['swish', 'swish']
                    },
                    'node_attention_mlp': {
                        'units': [32, 1],
                        'activation': ['swish', 'swish']
                    },
                    'edge_gate': None,
                    'node_gate': None,
                    'global_gate': None,
                    'residual_node_update': False,
                    'residual_edge_update': False,
                    'residual_global_update': False,
                    'update_edges_input': [True, True, True, True],
                    'update_nodes_input': [True, True, True],
                    'update_global_input': [True, True, True],
                    'multiplicity_readout': False
                },

                "input_embedding": {
                    "node": {"input_dim": 96, "output_dim": 64},
                    "graph": {"input_dim": 100, "output_dim": 64}
                },
                "depth": 5,
                "n_units": 128,

                "gin_mlp": {
                    "units": [128],
                    "use_bias": True,
                    "activation": ["swish"]
                },
                "graph_mlp": {
                    "units": [128],
                    "use_bias": True,
                    "activation": ["swish"]
                },

                "gin_args": {
                    "pooling_method": "sum",
                    "g_pooling_method": "mean",
                    "edge_mlp_args": {
                        "units": [128, 128, 128],
                        "use_bias": True,
                        "activation": ["swish", "swish", "swish"]
                    },
                    "concat_args": {"axis": -1},
                    "node_mlp_args": {
                        "units": [128],
                        "use_bias": True,
                        "activation": ["swish"]
                    },
                    "graph_mlp_args": {
                        "units": [128],
                        "use_bias": True,
                        "activation": ["swish"]
                    }
                }
            }
        },

        "training": {
            "fit": {
                "batch_size": 64,
                "epochs": 300,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler",
                        "config": {
                            "learning_rate_start": 0.001,
                            "learning_rate_stop": 1e-05,
                            "epo_min": 100,
                            "epo": 300,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam",
                    "config": {
                        "lr": {
                            "class_name": "ExponentialDecay",
                            "config": {
                                "initial_learning_rate": 0.001,
                                "decay_steps": 5800,
                                "decay_rate": 0.5,
                                "staircase": False
                            }
                        }
                    }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {
                "class_name": "KFold",
                "config": {
                    "n_splits": 5,
                    "random_state": 42,
                    "shuffle": True
                }
            },
            "scaler": {
                "class_name": "StandardScaler",
                "module_name": "kgcnn.data.transform.scaler.standard",
                "config": {
                    "with_std": True,
                    "with_mean": True,
                    "copy": True
                }
            },
            "multi_target_indices": None
        },

        "data": {
            "dataset": {
                "class_name": "JarvisMbjBandgapDataset",
                "module_name": "kgcnn.data.datasets.JarvisMbjBandgapDataset",
                "config": {},
                "methods": [
                    {
                        "set_representation": {
                            "pre_processor": {
                                "class_name": "VoronoiUnitCell",
                                "module_name": "kgcnn.crystal.preprocessor",
                                "config": {"min_ridge_area": 0.01}
                            },
                            "reset_graphs": False
                        }
                    }
                ]
            },
            "data_unit": "eV"
        },

        "info": {
            "postfix": "",
            "postfix_file": "_jarvis_mbj_bandgap",
            "kgcnn_version": "3.1.0"
        }
    }
}
