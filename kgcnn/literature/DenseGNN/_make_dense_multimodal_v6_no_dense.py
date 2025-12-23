"""
DenseGNN Multimodal V6 - NO Dense Connections

Based on discovery that ALIGNN (no Dense) works but DenseGNN (Dense) fails,
this version removes Dense connections to prevent text accumulation.

Key changes from original v6:
1. Remove Dense connections (LazyConcatenate)
2. Each layer is independent
3. Middle fusion after convolution (like ALIGNN)
"""

import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.pooling import PoolingGlobalEdges
from ._dense_gnn_conv import DenseGNN
from ._multimodal_fusion import ProjectionHead, GatedFusion, MiddleFusionModule

ks = tf.keras


def get_features(x):
    if isinstance(x, dict):
        assert "features" in x.keys()
        return x["features"]
    return x


def make_model_multimodal_v6_no_dense(
    inputs=None,
    name=None,
    input_block_cfg=None,
    input_embedding: dict = None,
    depth: int = None,
    gin_args: dict = None,
    gin_mlp: dict = None,
    graph_mlp: dict = None,
    output_mlp: dict = None,
    g_pooling_args: dict = None,
    # Multimodal config
    text_embedding_dim: int = 768,
    text_projection_dim: int = 128,
    graph_projection_dim: int = 128,
    use_middle_fusion: bool = True,
    middle_fusion_layers: list = None,
    middle_fusion_cfg: dict = None,
    late_fusion_type: str = "gated",
    late_fusion_cfg: dict = None,
):
    """DenseGNN v6 without Dense connections - prevents text accumulation.

    Key differences from original v6:
    - NO Dense connections (no LazyConcatenate)
    - Each layer operates independently
    - Middle fusion AFTER convolution (like ALIGNN)
    - Text information doesn't accumulate across layers
    """

    if middle_fusion_layers is None:
        middle_fusion_layers = [2]
    if middle_fusion_cfg is None:
        middle_fusion_cfg = {}
    if late_fusion_cfg is None:
        late_fusion_cfg = {'output_dim': 128, 'dropout': 0.1}
    if g_pooling_args is None:
        g_pooling_args = {'pooling_method': 'mean'}

    # === Graph Inputs ===
    def in_inputs(key):
        return key in inputs and inputs[key] is not None

    edge_indices = ks.Input(**inputs['edge_indices'])
    atomic_number = ks.Input(**inputs['atomic_number'])
    edge_inputs, node_inputs = [], [atomic_number]

    if in_inputs('offset'):
        offset = ks.Input(**inputs['offset'])
        edge_inputs.append(offset)
    else:
        raise ValueError('Model needs "offset" input.')

    if in_inputs('voronoi_ridge_area'):
        inp_voronoi_ridge_area = ks.Input(**inputs['voronoi_ridge_area'])
        edge_inputs.append(inp_voronoi_ridge_area)

    if in_inputs('AGNIFinger'):
        inp_AGNIFinger = ks.Input(**inputs['AGNIFinger'])
        node_inputs.append(inp_AGNIFinger)

    if in_inputs('charge'):
        charge_input = ks.Input(**inputs['charge'])
        node_inputs.append(charge_input)
    else:
        raise ValueError('Model needs "charge" input for global state.')

    # === Text Input ===
    text_spec = inputs.get('text_embedding')
    if text_spec:
        text_input = ks.Input(**text_spec)
    else:
        text_input = ks.Input(shape=(text_embedding_dim,), name='text_embedding')

    # === Process Graph ===
    euclidean_norm = EuclideanNorm()
    distance = euclidean_norm(offset)

    if in_inputs('voronoi_ridge_area'):
        edge_input = (distance, inp_voronoi_ridge_area)
    else:
        edge_input = distance

    if in_inputs('AGNIFinger'):
        node_input = {'features': atomic_number, 'AGNIFinger': inp_AGNIFinger}
    else:
        node_input = {'features': atomic_number}

    if in_inputs('charge'):
        global_input = charge_input
    else:
        global_input = None

    # === Input Block ===
    from ._make_asu import GraphNetworkConfigurator
    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    edge_features, node_features, global_features, _ = crystal_input_block([
        edge_input, node_input, global_input, edge_indices
    ])

    n = get_features(node_features)
    ed = edge_features
    ud = global_features
    edi = edge_indices

    node_dim = gin_mlp.get('units', [64])[-1]

    # === Text Projection ===
    text_projection = ProjectionHead(
        embedding_dim=text_embedding_dim,
        projection_dim=text_projection_dim,
        dropout=0.1
    )
    text_emb = text_projection(text_input)

    # === Middle Fusion Modules ===
    node_fusion_modules = {}
    global_fusion_modules = {}

    if use_middle_fusion:
        for layer_idx in middle_fusion_layers:
            node_fusion_modules[layer_idx] = MiddleFusionModule(
                node_dim=node_dim,
                text_dim=text_projection_dim,
                hidden_dim=middle_fusion_cfg.get('hidden_dim', node_dim * 2),
                dropout=middle_fusion_cfg.get('dropout', 0.1),
                num_heads=middle_fusion_cfg.get('num_heads', 0),
                use_gate_norm=middle_fusion_cfg.get('use_gate_norm', False),
                use_learnable_scale=middle_fusion_cfg.get('use_learnable_scale', False),
                initial_scale=middle_fusion_cfg.get('initial_scale', 1.0),
            )

            global_fusion_modules[layer_idx] = MiddleFusionModule(
                node_dim=gin_mlp.get('units', [64])[-1],
                text_dim=text_projection_dim,
                hidden_dim=middle_fusion_cfg.get('hidden_dim', node_dim * 2),
                dropout=middle_fusion_cfg.get('dropout', 0.1),
                num_heads=0,
                use_gate_norm=middle_fusion_cfg.get('use_gate_norm', False),
                use_learnable_scale=middle_fusion_cfg.get('use_learnable_scale', False),
                initial_scale=middle_fusion_cfg.get('initial_scale', 1.0),
            )

    # === GNN Layers WITHOUT Dense Connections ===
    for i in range(depth):
        if i > 0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
            ud = GraphMLP(**graph_mlp)(ud)

        # 🔧 KEY CHANGE: Convolution FIRST (like ALIGNN)
        np, ep, up = DenseGNN(**gin_args)([n, edi, ed, ud])

        # 🔧 KEY CHANGE: Fusion AFTER convolution (like ALIGNN)
        if use_middle_fusion and i in middle_fusion_layers:
            np = node_fusion_modules[i]([np, text_emb])
            up = global_fusion_modules[i]([up, text_emb])

        # 🔧 KEY CHANGE: NO Dense connections!
        # Just use the new values directly
        n = np
        ed = ep
        ud = up

    # === Output ===
    graph_out = PoolingGlobalEdges(**g_pooling_args)(ep)

    # === Graph Projection ===
    graph_projection = ProjectionHead(
        embedding_dim=graph_out.shape[-1] if graph_out.shape[-1] else 128,
        projection_dim=graph_projection_dim,
        dropout=0.1
    )
    graph_emb = graph_projection(graph_out)

    # === Late Fusion ===
    if late_fusion_type == "gated":
        gated_fusion = GatedFusion(
            graph_dim=graph_projection_dim,
            text_dim=text_projection_dim,
            **late_fusion_cfg
        )
        fused = gated_fusion([graph_emb, text_emb])
        out = MLP(units=[late_fusion_cfg.get('output_dim', 128), 64, 1],
                  activation=['swish', 'swish', 'linear'])(fused)
    elif late_fusion_type == "concat":
        fused = tf.concat([graph_emb, text_emb], axis=-1)
        out = MLP(units=[128, 64, 1], activation=['swish', 'swish', 'linear'])(fused)
    else:
        out = MLP(**output_mlp)(graph_out)

    # === Build Model ===
    input_list = edge_inputs + node_inputs + [edge_indices, text_input]
    return ks.Model(inputs=input_list, outputs=out, name=name)
