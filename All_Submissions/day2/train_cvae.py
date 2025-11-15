"""
Conditional Variational Autoencoder (CVAE) Training Script
===========================================================
Purpose: Train a CVAE model for learning latent representations of images
Output: Trained encoder and decoder models saved to disk

Usage: python train_cvae.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from datetime import datetime
import gc

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU found - training will use CPU")

# Clear memory
tf.keras.backend.clear_session()
gc.collect()

# Set random seeds for reproducibility
np.random.seed(716)
tf.random.set_seed(716)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'image_size': (224, 224),
    'latent_dim': 150,
    'num_classes': 10,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'data_path': '/home/ubuntu/deep_learning_exam1/Data',
    'excel_path': '/home/ubuntu/deep_learning_exam1/excel/train_test_cleaned.xlsx',
    'log_dir': './logs_cvae',
    'model_dir': './models_cvae',
    'nickname': 'Andrew',
    'kl_weight': 1.0,  # Weight for KL loss
}

# Create directories
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)

# ============================================================================
# DATA LOADING WITH tf.data (GPU-ACCELERATED & PARALLEL)
# ============================================================================

def load_and_preprocess_image(image_path, label, image_size):
    """
    Load and preprocess image using TensorFlow operations (GPU-compatible)
    Runs in parallel across multiple CPU cores
    """
    # Read and decode image
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    
    # Ensure RGB (handle grayscale, RGBA, etc.)
    img = img[:, :, :3]
    
    # Resize to target size
    img = tf.image.resize(img, image_size)
    
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label


def create_dataset(excel_path, data_path, image_size, batch_size, 
                   num_classes, split='train', shuffle=True):
    """
    Create GPU-accelerated tf.data.Dataset pipeline with:
    - Parallel image loading (uses all CPU cores)
    - Automatic prefetching (overlaps CPU/GPU work)
    - Memory efficient streaming
    """
    # Read Excel and filter by split
    df = pd.read_excel(excel_path)
    df_split = df[df['split'] == split].copy()
    
    print(f"\n{split.upper()} Dataset: {len(df_split)} samples")
    
    # Map class labels
    class_mapping = {f'class{i+1}': i for i in range(num_classes)}
    df_split['class_label'] = df_split['target'].map(class_mapping)
    
    # Get paths and labels
    image_paths = [os.path.join(data_path, img_id) for img_id in df_split['id'].values]
    labels = df_split['class_label'].values
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  class{cls+1}: {count} samples")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_onehot))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(image_paths)), 
                                 seed=716, reshuffle_each_iteration=True)
    
    # Parallel loading
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(df_split)

# ============================================================================
# VAE SAMPLING LAYER
# ============================================================================

class Sampling(layers.Layer):
    """Reparameterization trick: sample z from N(z_mean, exp(z_log_var))"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ============================================================================
# KL DIVERGENCE LOSS LAYER
# ============================================================================

class KLDivergenceLayer(layers.Layer):
    """
    Custom layer that adds KL divergence loss to the model
    This allows us to use standard Keras compile/fit without overriding train_step
    """
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        
        # Calculate KL divergence
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1
        )
        kl_loss = tf.reduce_mean(kl_loss)
        
        # Add as a loss to the layer
        self.add_loss(self.weight * kl_loss)
        
        # Also add as a metric for monitoring
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        
        # Pass through z_mean for further use
        return z_mean

# ============================================================================
# TENSORBOARD CALLBACK FOR IMAGE VISUALIZATION
# ============================================================================

class CVAETensorBoardCallback(keras.callbacks.Callback):
    """Visualize original, reconstructed, and generated images during training"""
    
    def __init__(self, val_dataset, decoder, log_dir, latent_dim, num_classes, num_examples=5):
        super().__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_examples = num_examples
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'images'))
        
        # Get validation samples
        for imgs, labels in val_dataset.take(1):
            self.val_images = imgs[:num_examples]
            self.val_labels = labels[:num_examples]
            break
        
    def on_epoch_end(self, epoch, logs=None):
        # Get reconstructions from the full model
        reconstructions = self.model([self.val_images, self.val_labels], training=False)
        
        # Generate random samples
        z_random = np.random.normal(size=(self.num_examples, self.latent_dim))
        random_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, self.num_classes, self.num_examples),
            num_classes=self.num_classes
        )
        generated = self.decoder([z_random, random_labels], training=False)
        
        # Log to TensorBoard
        with self.file_writer.as_default():
            tf.summary.image("1_Original", self.val_images, max_outputs=self.num_examples, step=epoch)
            tf.summary.image("2_Reconstructed", reconstructions, max_outputs=self.num_examples, step=epoch)
            tf.summary.image("3_Generated", generated, max_outputs=self.num_examples, step=epoch)
            
            comparison = tf.concat([self.val_images, reconstructions], axis=2)
            tf.summary.image("4_Comparison", comparison, max_outputs=self.num_examples, step=epoch)
        
        self.file_writer.flush()
    
    def on_train_end(self, logs=None):
        self.file_writer.close()

# ============================================================================
# BUILD CVAE MODEL (FUNCTIONAL API - NO CUSTOM train_step)
# ============================================================================

def build_cvae(image_size, latent_dim, num_classes, kl_weight=1.0):
    """
    Build complete CVAE model using Functional API
    Uses custom KL loss layer instead of overriding train_step
    """
    
    # ========== ENCODER ==========
    image_input = layers.Input(shape=(*image_size, 3), name='image_input')
    class_input = layers.Input(shape=(num_classes,), name='class_input')
    
    # Flatten and concatenate
    img_flat = layers.Flatten(name='flatten')(image_input)
    combined = layers.Concatenate(name='concat_encoder')([img_flat, class_input])
    
    # Encoder network (simplified for memory)
    x = layers.Dense(1024, activation='relu', name='enc_dense1')(combined)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = layers.Dropout(0.3, name='enc_dropout1')(x)
    
    x = layers.Dense(512, activation='relu', name='enc_dense2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = layers.Dropout(0.3, name='enc_dropout2')(x)
    
    x = layers.Dense(256, activation='relu', name='enc_dense3')(x)
    x = layers.BatchNormalization(name='enc_bn3')(x)

def build_encoder(image_size, latent_dim, num_classes):
    """Build conditional encoder with FEWER parameters"""
    
    # Inputs
    image_input = layers.Input(shape=(*image_size, 3), name='image_input')
    class_input = layers.Input(shape=(num_classes,), name='class_input')
    
    # Flatten and concatenate
    img_flat = layers.Flatten()(image_input)
    combined = layers.Concatenate()([img_flat, class_input])
    
    # REDUCED network - fewer layers and neurons
    x = layers.Dense(624, activation='relu')(combined)  # Was 2048
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, activation='relu')(x)  # Was 1024
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)  # Was 512
    x = layers.BatchNormalization()(x)
    
    # Latent parameters
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    return models.Model(
        inputs=[image_input, class_input],
        outputs=[z_mean, z_log_var, z],
        name='encoder'
    )


def build_decoder(image_size, latent_dim, num_classes):
    """Build conditional decoder with FEWER parameters"""
    
    # Inputs
    latent_input = layers.Input(shape=(latent_dim,), name='latent_input')
    class_input = layers.Input(shape=(num_classes,), name='class_input')
    
    # Concatenate
    combined = layers.Concatenate()([latent_input, class_input])
    
    # REDUCED network - fewer layers and neurons
    x = layers.Dense(256, activation='relu')(combined)  # Was 512
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, activation='relu')(x)  # Was 1024
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(624, activation='relu')(x)  # Was 2048
    x = layers.BatchNormalization()(x)
    
    # Output
    img_flat_dim = image_size[0] * image_size[1] * 3
    x = layers.Dense(img_flat_dim, activation='sigmoid')(x)
    output = layers.Reshape((*image_size, 3))(x)
    
    return models.Model(
        inputs=[latent_input, class_input],
        outputs=output,
        name='decoder'
    )

# ============================================================================
# CVAE MODEL
# ============================================================================

class CVAE(keras.Model):
    """Conditional Variational Autoencoder"""
    
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        # Training metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        # Validation metrics
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_recon_loss_tracker = keras.metrics.Mean(name="val_recon_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker,
            self.val_total_loss_tracker, self.val_recon_loss_tracker, self.val_kl_loss_tracker,
        ]
    
    def train_step(self, data):
        images, class_labels = data
        
        with tf.GradientTape() as tape:
            # Encode and decode
            z_mean, z_log_var, z = self.encoder([images, class_labels])
            reconstruction = self.decoder([z, class_labels])
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(images, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = recon_loss + kl_loss
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        images, class_labels = data
        
        # Encode and decode
        z_mean, z_log_var, z = self.encoder([images, class_labels], training=False)
        reconstruction = self.decoder([z, class_labels], training=False)
        
        # Losses
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(images, reconstruction),
                axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = recon_loss + kl_loss
        
        # Update validation metrics
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_recon_loss_tracker.update_state(recon_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        
        return {
            "val_total_loss": self.val_total_loss_tracker.result(),
            "val_recon_loss": self.val_recon_loss_tracker.result(),
            "val_kl_loss": self.val_kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        images, class_labels = inputs
        z_mean, z_log_var, z = self.encoder([images, class_labels])
        return self.decoder([z, class_labels])

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def save_model_summary(model, filepath):
    """Save model summary to text file"""
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def train_cvae(config):
    """Train CVAE model"""
    
    print("="*80)
    print("CVAE TRAINING - GPU ACCELERATED")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ Training on GPU: {gpus[0].name}")
    else:
        print("\n⚠ Training on CPU")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, train_size = create_dataset(
        config['excel_path'], config['data_path'], config['image_size'],
        config['batch_size'], config['num_classes'], split='train', shuffle=True
    )
    
    val_dataset, val_size = create_dataset(
        config['excel_path'], config['data_path'], config['image_size'],
        config['batch_size'], config['num_classes'], split='train', shuffle=False
    )
    
    # Calculate steps
    steps_per_epoch = train_size // config['batch_size']
    validation_steps = val_size // config['batch_size']
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Parallel loading: ✓ ENABLED")
    print(f"  Prefetching: ✓ ENABLED")
    
    # Build models
    print("\nBuilding CVAE...")
    encoder = build_encoder(config['image_size'], config['latent_dim'], config['num_classes'])
    decoder = build_decoder(config['image_size'], config['latent_dim'], config['num_classes'])
    cvae = CVAE(encoder, decoder)
    cvae.compile(optimizer=keras.optimizers.Adam(config['learning_rate']))
    
    # Save model summaries
    save_model_summary(encoder, os.path.join(config['model_dir'], f"encoder_summary_{config['nickname']}.txt"))
    save_model_summary(decoder, os.path.join(config['model_dir'], f"decoder_summary_{config['nickname']}.txt"))
    print(f"\n✓ Model summaries saved to {config['model_dir']}/")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config['log_dir'], timestamp)
    
    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch='10,20',
        ),
        CVAETensorBoardCallback(
            val_dataset, log_dir, config['latent_dim'], 
            config['num_classes'], num_examples=5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['model_dir'], f"cvae_best_{config['nickname']}.keras"),
            monitor='val_total_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"\nTensorBoard: tensorboard --logdir={config['log_dir']}")
    print("Then open: http://localhost:6006\n")
    
    history = cvae.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final models
    encoder.save(os.path.join(config['model_dir'], f"cvae_encoder_{config['nickname']}.keras"))
    decoder.save(os.path.join(config['model_dir'], f"cvae_decoder_{config['nickname']}.keras"))
    cvae.save(os.path.join(config['model_dir'], f"cvae_full_{config['nickname']}.keras"))
    
    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(config['model_dir'], f"training_history_{config['nickname']}.csv"), index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to: {config['model_dir']}/")
    print(f"  - cvae_encoder_{config['nickname']}.keras")
    print(f"  - cvae_decoder_{config['nickname']}.keras")
    print(f"  - cvae_full_{config['nickname']}.keras")
    print(f"  - cvae_best_{config['nickname']}.keras")
    print(f"  - training_history_{config['nickname']}.csv")
    print(f"\nUse generate_synthetic_data.py to create synthetic images")
    
    return cvae, encoder, decoder, history

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_cvae(CONFIG)
