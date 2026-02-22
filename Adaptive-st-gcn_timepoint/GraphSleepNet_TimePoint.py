"""
GraphSleepNet_TimePoint.py

Multi-resolution architecture for time-point spindle detection.
Combines:
1. Graph-based spatial-temporal features (from DE)
2. Raw EEG temporal refinement
3. Multi-scale fusion for precise time-point predictions

Key idea: Use GraphSleepNet for coarse detection + raw EEG CNN for fine-grained timing
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Import original components
from GraphSleepNet import (
    TemporalAttention, SpatialAttention, Graph_Learn,
    cheb_conv_with_SAt_GL, cheb_conv_with_SAt_static,
    AddReluLayerNorm, reshape_dot, GraphSleepBlock
)


def build_GraphSleepNet_TimePoint(
    # Graph conv params
    k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
    cheb_polynomials, time_conv_kernel, sample_shape, num_block,
    # Dense params
    dense_size, opt, useGL, GLalpha, regularizer, dropout,
    # NEW: time-point prediction params
    samples_per_window=400,  # 2 sec * 200 Hz
    use_raw_eeg=True,
    raw_eeg_channels=19,
):

    
    context, n_channels, n_bands = sample_shape
    
    # ========================================================================
    # BRANCH 1: DE Features (Graph-based coarse detection)
    # ========================================================================
    de_input = layers.Input(shape=sample_shape, name='DE_Input')
    
    # GraphSleepNet blocks
    x_de = GraphSleepBlock(
        de_input, k, num_of_chev_filters, num_of_time_filters,
        time_conv_strides, cheb_polynomials, time_conv_kernel,
        useGL, GLalpha, i=0
    )
    
    for i in range(1, num_block):
        x_de = GraphSleepBlock(
            x_de, k, num_of_chev_filters, num_of_time_filters,
            1, cheb_polynomials, time_conv_kernel, useGL, GLalpha, i=i
        )
    
    # x_de shape: (B, context, n_channels, num_of_time_filters)
    
    # Compress to temporal features
    x_de_flat = layers.Reshape((context, -1))(x_de)  # (B, context, C*F)
    x_de_features = layers.Dense(128, activation='relu', name='DE_Dense')(x_de_flat)
    
    # Coarse window-level predictions (one per context timestep)
    coarse_pred = layers.TimeDistributed(
        layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer),
        name='Coarse_Predictions'
    )(x_de_features)  # (B, context, 1)
    
    # ========================================================================
    # BRANCH 2: Raw EEG (1D CNN for fine-grained timing)
    # ========================================================================
    if use_raw_eeg:
        # Input: raw EEG for center window only
        # Shape: (B, samples_per_window, n_channels)
        raw_input = layers.Input(
            shape=(samples_per_window, raw_eeg_channels),
            name='Raw_EEG_Input'
        )
        
        # Multi-scale 1D convolutions
        # Scale 1: Capture fast oscillations (spindle frequency ~11-16 Hz)
        x_raw_1 = layers.Conv1D(32, kernel_size=25, padding='same', activation='relu')(raw_input)
        x_raw_1 = layers.BatchNormalization()(x_raw_1)
        x_raw_1 = layers.MaxPooling1D(pool_size=2)(x_raw_1)  # (B, 200, 32)
        
        # Scale 2: Medium-term patterns
        x_raw_2 = layers.Conv1D(64, kernel_size=50, padding='same', activation='relu')(x_raw_1)
        x_raw_2 = layers.BatchNormalization()(x_raw_2)
        x_raw_2 = layers.MaxPooling1D(pool_size=2)(x_raw_2)  # (B, 100, 64)
        
        # Scale 3: Long-term context
        x_raw_3 = layers.Conv1D(64, kernel_size=25, padding='same', activation='relu')(x_raw_2)
        x_raw_3 = layers.BatchNormalization()(x_raw_3)
        
        # Upsample back to original resolution
        x_raw_up = layers.UpSampling1D(size=4)(x_raw_3)  # (B, 400, 64)
        
        # ====================================================================
        # FUSION: Combine coarse predictions with fine features
        # ====================================================================
        
        # Expand coarse predictions to match raw resolution
        # coarse_pred is (B, context, 1), we want center window
        center_idx = context // 2
        center_coarse = coarse_pred[:, center_idx:center_idx+1, :]  # (B, 1, 1)
        
        # Broadcast to all samples in window
        coarse_broadcast = layers.Lambda(
            lambda x: tf.tile(x, [1, samples_per_window, 1]),
            name='Coarse_Broadcast'
        )(center_coarse)  # (B, 400, 1)
        
        # Concatenate coarse + fine features
        fusion = layers.Concatenate(axis=-1)([x_raw_up, coarse_broadcast])
        
        # Final refinement layers
        x_fused = layers.Conv1D(32, kernel_size=15, padding='same', activation='relu')(fusion)
        x_fused = layers.BatchNormalization()(x_fused)
        if dropout > 0:
            x_fused = layers.Dropout(dropout)(x_fused)
        
        # Per-sample predictions
        fine_pred = layers.Conv1D(
            1, kernel_size=1, activation='sigmoid',
            kernel_regularizer=regularizer,
            name='Fine_Predictions'
        )(x_fused)  # (B, 400, 1)
        
        # Build model with both inputs
        model = models.Model(
            inputs=[de_input, raw_input],
            outputs={'coarse': coarse_pred, 'fine': fine_pred}
        )
        
    else:
        # Fallback: Just upsample coarse predictions
        # This is NOT recommended but provided for comparison
        
        # Extract center window prediction
        center_idx = context // 2
        center_pred = coarse_pred[:, center_idx:center_idx+1, :]  # (B, 1, 1)
        
        # Simple upsampling
        fine_pred = layers.Lambda(
            lambda x: tf.tile(x, [1, samples_per_window, 1]),
            name='Upsampled_Predictions'
        )(center_pred)
        
        model = models.Model(
            inputs=de_input,
            outputs={'coarse': coarse_pred, 'fine': fine_pred}
        )
    
    # ========================================================================
    # COMPILE with multi-output loss
    # ========================================================================
    
    model.compile(
        optimizer=opt,
        loss={
            'coarse': 'binary_crossentropy',
            'fine': 'binary_crossentropy'
        },
        loss_weights={
            'coarse': 0.3,  # Lower weight for coarse (auxiliary task)
            'fine': 1.0     # Main task: fine-grained prediction
        },
        metrics={
            'coarse': [
                keras.metrics.BinaryAccuracy(name='acc'),
                keras.metrics.AUC(curve='PR', name='pr_auc')
            ],
            'fine': [
                keras.metrics.BinaryAccuracy(name='acc'),
                keras.metrics.AUC(curve='PR', name='pr_auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        }
    )
    
    return model

