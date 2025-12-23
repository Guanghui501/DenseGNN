"""Multimodal fusion modules for DenseGNN.

Implements text-graph multimodal fusion following SGA-fusion framework:
- ProjectionHead: Maps embeddings to common space
- MiddleFusionModule: Mid-level gated fusion
- GatedFusion: Late fusion with learned weights
"""

import tensorflow as tf
from tensorflow.keras import layers


class ProjectionHead(layers.Layer):
    """Projection head for mapping embeddings to a common space."""

    def __init__(self, embedding_dim, projection_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.projection = layers.Dense(self.projection_dim)
        self.gelu = layers.Activation('gelu')
        self.fc = layers.Dense(self.projection_dim)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x, training=training)
        x = x + projected
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'dropout': self.dropout_rate
        })
        return config


class MiddleFusionModule(layers.Layer):
    """Middle fusion module for injecting text information into graph encoding.

    Uses gated fusion mechanism:
    enhanced = node_feat + gate_values * text_broadcasted
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, dropout=0.1,
                 use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_gate_norm = use_gate_norm
        self.use_learnable_scale = use_learnable_scale
        self.initial_scale = initial_scale

    def build(self, input_shape):
        # Text transformation
        self.text_dense1 = layers.Dense(self.hidden_dim, activation='relu')
        self.text_dropout = layers.Dropout(self.dropout_rate)
        self.text_dense2 = layers.Dense(self.node_dim)

        # Gate mechanism
        self.gate_dense = layers.Dense(self.node_dim, activation='sigmoid')

        # Layer normalization
        self.layer_norm = layers.LayerNormalization()
        self.output_dropout = layers.Dropout(self.dropout_rate)

        if self.use_gate_norm:
            self.gate_norm = layers.LayerNormalization()

        if self.use_learnable_scale:
            self.text_scale = self.add_weight(
                name='text_scale', shape=(),
                initializer=tf.constant_initializer(self.initial_scale),
                trainable=True
            )
        else:
            self.text_scale = tf.constant(1.0, dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [node_feat, text_feat]
                - node_feat: [batch, N, node_dim] or RaggedTensor
                - text_feat: [batch, text_dim]
        Returns:
            Enhanced node features
        """
        node_feat, text_feat = inputs

        # Transform text features
        text_transformed = self.text_dense1(text_feat)
        text_transformed = self.text_dropout(text_transformed, training=training)
        text_transformed = self.text_dense2(text_transformed)  # [batch, node_dim]
        text_transformed = text_transformed * self.text_scale

        # Broadcast text to all nodes
        if isinstance(node_feat, tf.RaggedTensor):
            # Handle RaggedTensor
            def process_sample(args):
                nodes, text = args
                num_nodes = tf.shape(nodes)[0]
                text_exp = tf.repeat(tf.expand_dims(text, 0), num_nodes, axis=0)
                gate_in = tf.concat([nodes, text_exp], axis=-1)
                if self.use_gate_norm:
                    gate_in = self.gate_norm(gate_in)
                gate_val = self.gate_dense(gate_in)
                enhanced = nodes + gate_val * text_exp
                return enhanced

            enhanced = tf.map_fn(
                process_sample,
                (node_feat, text_transformed),
                fn_output_signature=tf.RaggedTensorSpec(
                    shape=[None, self.node_dim], dtype=tf.float32, ragged_rank=0
                )
            )
        else:
            # Check tensor rank
            node_rank = len(node_feat.shape)
            if node_rank == 2:
                # Graph-level features [batch, node_dim] - no broadcasting needed
                gate_input = tf.concat([node_feat, text_transformed], axis=-1)
                if self.use_gate_norm:
                    gate_input = self.gate_norm(gate_input)
                gate_values = self.gate_dense(gate_input)
                enhanced = node_feat + gate_values * text_transformed
            else:
                # Node-level features [batch, N, node_dim]
                text_broadcasted = tf.expand_dims(text_transformed, 1)
                text_broadcasted = tf.broadcast_to(text_broadcasted, tf.shape(node_feat))

                gate_input = tf.concat([node_feat, text_broadcasted], axis=-1)
                if self.use_gate_norm:
                    gate_input = self.gate_norm(gate_input)

                gate_values = self.gate_dense(gate_input)
                enhanced = node_feat + gate_values * text_broadcasted

        enhanced = self.layer_norm(enhanced)
        enhanced = self.output_dropout(enhanced, training=training)
        return enhanced

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_dim': self.node_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'use_gate_norm': self.use_gate_norm,
            'use_learnable_scale': self.use_learnable_scale,
            'initial_scale': self.initial_scale
        })
        return config


class GatedFusion(layers.Layer):
    """Gated fusion module - learns dynamic weights to balance two modalities.

    gate_g, gate_t = normalized sigmoid outputs
    fused = gate_g * graph_transformed + gate_t * text_transformed
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Gate for graph
        self.gate_graph_dense1 = layers.Dense(self.graph_dim // 2, activation='relu')
        self.gate_graph_dropout = layers.Dropout(self.dropout_rate)
        self.gate_graph_dense2 = layers.Dense(1, activation='sigmoid')

        # Gate for text
        self.gate_text_dense1 = layers.Dense(self.text_dim // 2, activation='relu')
        self.gate_text_dropout = layers.Dropout(self.dropout_rate)
        self.gate_text_dense2 = layers.Dense(1, activation='sigmoid')

        # Feature transformation
        self.graph_transform = layers.Dense(self.output_dim)
        self.text_transform = layers.Dense(self.output_dim)

        # Fusion transformation
        self.fusion_dense = layers.Dense(self.output_dim)
        self.fusion_norm = layers.LayerNormalization()
        self.fusion_activation = layers.Activation('relu')
        self.fusion_dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [graph_feat, text_feat]
                - graph_feat: [batch, graph_dim]
                - text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        graph_feat, text_feat = inputs

        # Compute gate weights
        gate_g = self.gate_graph_dense1(graph_feat)
        gate_g = self.gate_graph_dropout(gate_g, training=training)
        gate_g = self.gate_graph_dense2(gate_g)

        gate_t = self.gate_text_dense1(text_feat)
        gate_t = self.gate_text_dropout(gate_t, training=training)
        gate_t = self.gate_text_dense2(gate_t)

        # Normalize gates
        gate_sum = gate_g + gate_t + 1e-8
        gate_g = gate_g / gate_sum
        gate_t = gate_t / gate_sum

        # Transform features
        graph_transformed = self.graph_transform(graph_feat)
        text_transformed = self.text_transform(text_feat)

        # Gated fusion
        fused = gate_g * graph_transformed + gate_t * text_transformed

        # Final transformation
        fused = self.fusion_dense(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_dropout(fused, training=training)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout_rate
        })
        return config


class CrossModalAttention(layers.Layer):
    """Cross-modal attention between graph and text features."""

    def __init__(self, graph_dim=64, text_dim=64, hidden_dim=256,
                 num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.scale = (hidden_dim // num_heads) ** -0.5

    def build(self, input_shape):
        # Graph-to-Text attention
        self.g2t_query = layers.Dense(self.hidden_dim)
        self.g2t_key = layers.Dense(self.hidden_dim)
        self.g2t_value = layers.Dense(self.hidden_dim)

        # Text-to-Graph attention
        self.t2g_query = layers.Dense(self.hidden_dim)
        self.t2g_key = layers.Dense(self.hidden_dim)
        self.t2g_value = layers.Dense(self.hidden_dim)

        # Output projections
        self.graph_output = layers.Dense(self.graph_dim)
        self.text_output = layers.Dense(self.text_dim)

        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm_graph = layers.LayerNormalization()
        self.layer_norm_text = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [graph_feat, text_feat]
        Returns:
            enhanced_graph, enhanced_text
        """
        graph_feat, text_feat = inputs

        # Add sequence dimension
        graph_seq = tf.expand_dims(graph_feat, 1)
        text_seq = tf.expand_dims(text_feat, 1)

        # Graph-to-Text Attention
        Q_g2t = self.g2t_query(graph_seq)
        K_g2t = self.g2t_key(text_seq)
        V_g2t = self.g2t_value(text_seq)

        attn_g2t = tf.matmul(Q_g2t, K_g2t, transpose_b=True) * self.scale
        attn_g2t = tf.nn.softmax(attn_g2t, axis=-1)
        attn_g2t = self.dropout(attn_g2t, training=training)
        context_g2t = tf.squeeze(self.graph_output(tf.matmul(attn_g2t, V_g2t)), axis=1)

        # Text-to-Graph Attention
        Q_t2g = self.t2g_query(text_seq)
        K_t2g = self.t2g_key(graph_seq)
        V_t2g = self.t2g_value(graph_seq)

        attn_t2g = tf.matmul(Q_t2g, K_t2g, transpose_b=True) * self.scale
        attn_t2g = tf.nn.softmax(attn_t2g, axis=-1)
        attn_t2g = self.dropout(attn_t2g, training=training)
        context_t2g = tf.squeeze(self.text_output(tf.matmul(attn_t2g, V_t2g)), axis=1)

        # Residual + LayerNorm
        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)
        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

        return enhanced_graph, enhanced_text

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate
        })
        return config
