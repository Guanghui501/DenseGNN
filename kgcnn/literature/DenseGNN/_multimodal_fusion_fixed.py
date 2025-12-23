"""Multimodal fusion modules for DenseGNN - FIXED VERSION

Fixed based on diagnostic results showing optimal fusion weight α=0.86:
- Graph should contribute 86%
- Text should contribute 14%

This version uses a simplified GatedFusion with correct initial weight.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ProjectionHead(layers.Layer):
    """Projection head for mapping embeddings to a common space.

    UNCHANGED from original - this works fine.
    """

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
    """Middle fusion module - UNCHANGED from original.

    Note: Consider disabling middle fusion entirely (use_middle_fusion=False)
    as it may give text too much influence.
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, dropout=0.1,
                 use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0,
                 num_heads=0, **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_gate_norm = use_gate_norm
        self.use_learnable_scale = use_learnable_scale
        self.initial_scale = initial_scale
        self.num_heads = num_heads

    def build(self, input_shape):
        # Text transformation
        self.text_dense1 = layers.Dense(self.hidden_dim, activation='relu')
        self.text_dropout = layers.Dropout(self.dropout_rate)
        self.text_dense2 = layers.Dense(self.node_dim)

        # Multi-head attention (if enabled)
        if self.num_heads > 0:
            self.query_dense = layers.Dense(self.hidden_dim)
            self.key_dense = layers.Dense(self.hidden_dim)
            self.value_dense = layers.Dense(self.hidden_dim)
            self.attention_output = layers.Dense(self.node_dim)
            self.attention_dropout = layers.Dropout(self.dropout_rate)
            self.attention_norm = layers.LayerNormalization()

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
        node_feat, text_feat = inputs

        # Transform text features
        text_transformed = self.text_dense1(text_feat)
        text_transformed = self.text_dropout(text_transformed, training=training)
        text_transformed = self.text_dense2(text_transformed)
        text_transformed = text_transformed * self.text_scale

        # Multi-head attention (if enabled)
        if self.num_heads > 0:
            text_transformed = self._apply_attention(node_feat, text_transformed, training)

        # Broadcast text to all nodes
        if isinstance(node_feat, tf.RaggedTensor):
            enhanced = self._process_ragged(node_feat, text_transformed, training)
        else:
            enhanced = self._process_dense(node_feat, text_transformed, training)

        enhanced = self.layer_norm(enhanced)
        enhanced = self.output_dropout(enhanced, training=training)
        return enhanced

    def _apply_attention(self, node_feat, text_feat, training):
        if isinstance(node_feat, tf.RaggedTensor):
            node_pooled = tf.reduce_mean(node_feat, axis=1)
        else:
            if len(node_feat.shape) == 3:
                node_pooled = tf.reduce_mean(node_feat, axis=1)
            else:
                node_pooled = node_feat

        Q = self.query_dense(node_pooled)
        K = self.key_dense(text_feat)
        V = self.value_dense(text_feat)

        batch_size = tf.shape(Q)[0]
        head_dim = self.hidden_dim // self.num_heads

        Q = tf.reshape(Q, [batch_size, self.num_heads, head_dim])
        K = tf.reshape(K, [batch_size, self.num_heads, head_dim])
        V = tf.reshape(V, [batch_size, self.num_heads, head_dim])

        scale = tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attention_scores = tf.einsum('bnh,bnh->bn', Q, K) / scale
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights, training=training)

        context = tf.einsum('bn,bnh->bnh', attention_weights, V)
        context = tf.reshape(context, [batch_size, self.hidden_dim])

        attended = self.attention_output(context)
        attended = self.attention_norm(text_feat + attended)

        return attended

    def _process_ragged(self, node_feat, text_transformed, training):
        flat_nodes = node_feat.flat_values
        row_splits = node_feat.row_splits
        row_lengths = node_feat.row_lengths()

        text_repeated = tf.repeat(text_transformed, row_lengths, axis=0)

        gate_input = tf.concat([flat_nodes, text_repeated], axis=-1)
        if self.use_gate_norm:
            gate_input = self.gate_norm(gate_input)
        gate_values = self.gate_dense(gate_input)

        enhanced_flat = flat_nodes + gate_values * text_repeated

        enhanced = tf.RaggedTensor.from_row_splits(
            values=enhanced_flat,
            row_splits=row_splits
        )
        return enhanced

    def _process_dense(self, node_feat, text_transformed, training):
        node_rank = len(node_feat.shape)
        if node_rank == 2:
            gate_input = tf.concat([node_feat, text_transformed], axis=-1)
            if self.use_gate_norm:
                gate_input = self.gate_norm(gate_input)
            gate_values = self.gate_dense(gate_input)
            enhanced = node_feat + gate_values * text_transformed
        else:
            text_broadcasted = tf.expand_dims(text_transformed, 1)
            text_broadcasted = tf.broadcast_to(text_broadcasted, tf.shape(node_feat))

            gate_input = tf.concat([node_feat, text_broadcasted], axis=-1)
            if self.use_gate_norm:
                gate_input = self.gate_norm(gate_input)

            gate_values = self.gate_dense(gate_input)
            enhanced = node_feat + gate_values * text_broadcasted
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
            'initial_scale': self.initial_scale,
            'num_heads': self.num_heads
        })
        return config


class GatedFusion(layers.Layer):
    """🔧 FIXED: Gated fusion with optimal weight initialization.

    Based on diagnostic results showing optimal weight α=0.86:
    - Graph contributes 86%
    - Text contributes 14%

    This simplified version uses a single learnable gate weight initialized
    to the optimal value, and can be fine-tuned during training.

    Formula: fused = gate * graph + (1-gate) * text
    where gate is initialized to 0.86 (can be adjusted via initial_gate parameter)
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1,
                 initial_gate=0.86, trainable_gate=True, **kwargs):
        """
        Args:
            graph_dim: Graph embedding dimension
            text_dim: Text embedding dimension
            output_dim: Output dimension
            dropout: Dropout rate
            initial_gate: Initial gate weight for graph (0-1)
                         Default 0.86 based on diagnostic results
                         Higher = more graph, Lower = more text
            trainable_gate: Whether gate can be fine-tuned during training
        """
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.initial_gate = initial_gate
        self.trainable_gate = trainable_gate

    def build(self, input_shape):
        # Single learnable gate weight (graph contribution)
        # Text contribution = 1 - gate
        self.gate = self.add_weight(
            name='fusion_gate',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_gate),
            trainable=self.trainable_gate,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)  # Keep in [0,1]
        )

        # Feature transformation to common space
        self.graph_transform = layers.Dense(self.output_dim, name='graph_transform')
        self.text_transform = layers.Dense(self.output_dim, name='text_transform')

        # Post-fusion processing
        self.fusion_dense = layers.Dense(self.output_dim, name='fusion_dense')
        self.fusion_norm = layers.LayerNormalization(name='fusion_norm')
        self.fusion_activation = layers.Activation('relu', name='fusion_activation')
        self.fusion_dropout = layers.Dropout(self.dropout_rate, name='fusion_dropout')

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

        # Transform to common space
        graph_transformed = self.graph_transform(graph_feat)
        text_transformed = self.text_transform(text_feat)

        # Weighted fusion: α * graph + (1-α) * text
        # where α = self.gate (initialized to 0.86)
        fused = self.gate * graph_transformed + (1.0 - self.gate) * text_transformed

        # Post-fusion processing
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
            'dropout': self.dropout_rate,
            'initial_gate': self.initial_gate,
            'trainable_gate': self.trainable_gate
        })
        return config


class ResidualFusion(layers.Layer):
    """Alternative fusion: Graph主预测 + Text小幅修正

    This treats graph as the primary modality and text as a small correction.

    Formula: final = graph_prediction + correction_weight * text_correction
    where correction_weight = 0.14 (corresponding to α=0.86 for graph)
    """

    def __init__(self, graph_dim=64, text_dim=64, correction_weight=0.14,
                 trainable_weight=False, **kwargs):
        """
        Args:
            graph_dim: Graph embedding dimension
            text_dim: Text embedding dimension
            correction_weight: Weight for text correction (default 0.14 = 1-0.86)
            trainable_weight: Whether correction weight can be trained
        """
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.correction_weight_init = correction_weight
        self.trainable_weight = trainable_weight

    def build(self, input_shape):
        # Graph main prediction head
        self.graph_head = layers.Dense(1, name='graph_prediction')

        # Text correction head
        self.text_head = layers.Dense(1, name='text_correction')

        # Learnable correction weight (optional)
        if self.trainable_weight:
            self.correction_weight = self.add_weight(
                name='correction_weight',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(self.correction_weight_init),
                trainable=True,
                constraint=lambda x: tf.clip_by_value(x, 0.0, 0.5)  # Max 50% correction
            )
        else:
            self.correction_weight = tf.constant(self.correction_weight_init, dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [graph_emb, text_emb]
        Returns:
            final_prediction: [batch, 1]
        """
        graph_emb, text_emb = inputs

        # Main prediction from graph
        graph_pred = self.graph_head(graph_emb)

        # Correction from text
        text_correction = self.text_head(text_emb)

        # Residual combination
        final_pred = graph_pred + self.correction_weight * text_correction

        return final_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'correction_weight': float(self.correction_weight_init),
            'trainable_weight': self.trainable_weight
        })
        return config
