
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# 0. GPU CONFIGURATION AND VERIFICATION
# ============================================================================

def setup_gpu():
    """Configure GPU settings and verify GPU availability"""
    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")

        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n✓ Memory growth enabled for all GPUs")

            # Set visible devices
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✓ {len(logical_gpus)} Logical GPU(s) available")

        except RuntimeError as e:
            print(f"\n✗ Error configuring GPU: {e}")
    else:
        print("\n✗ No GPU detected. Training will run on CPU.")
        print("  To use GPU, ensure CUDA and cuDNN are properly installed.")

    print("=" * 80)
    return len(gpus) > 0

# Set random seeds for reproducibility
np.random.seed(716)
tf.random.set_seed(716)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'image_size': (100, 100),      # Image dimensions
    'latent_dim': 128,            # REDUCED from 128 to 64
    'num_classes': 10,           # 10 classes
    'batch_size': 128,           # INCREASED for better GPU utilization
    'epochs': 100,
    'learning_rate': 0.001,
    'data_path': '/home/ubuntu/deep_learning_exam1/Data',
    'excel_path': '/home/ubuntu/deep_learning_exam1/excel/train_test_cleaned.xlsx',
    'prefetch_size': tf.data.AUTOTUNE,  # Automatic prefetching
    'num_parallel_calls': tf.data.AUTOTUNE  # Automatic parallelization
}

# ============================================================================
# 2. PARALLEL DATA LOADING WITH tf.data
# ============================================================================

def create_tf_dataset(excel_path, data_path, image_size, batch_size, 
                     num_parallel_calls, prefetch_size):
    """
    Create optimized tf.data pipeline for parallel image loading
    FIXED: Returns dataset in format expected by VAE (inputs, outputs)
    """
    print("\nCreating optimized data pipeline...")

    # Read Excel file
    df = pd.read_excel(excel_path)
    train_df = df[df['split'] == 'train'].copy()

    print(f"Total training samples: {len(train_df)}")

    # Parse class labels
    class_mapping = {f'class{i+1}': i for i in range(10)}
    train_df['class_label'] = train_df['target'].map(class_mapping)

    # Get file paths and labels
    file_paths = [os.path.join(data_path, img_id) for img_id in train_df['id'].values]
    labels = train_df['class_label'].values

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  class{cls+1}: {count} samples")

    def load_and_preprocess_image(file_path, label):
        """Load and preprocess a single image"""
        # Read image file
        image = tf.io.read_file(file_path)

        # Decode image (handles jpg, png, etc.)
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize to target size
        image = tf.image.resize(image, image_size)

        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Convert label to one-hot encoding
        label_onehot = tf.one_hot(label, depth=CONFIG['num_classes'])

        return image, label_onehot

    def format_for_vae(image, label):
        """
        Format data for VAE training
        Input: (image, label)
        Output: ((image, label), image)
        The VAE takes [image, label] as input and reconstructs image as output
        """
        return (image, label), image

    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42)

    # Parallel image loading and preprocessing
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=num_parallel_calls
    )

    # Split into train and validation
    train_size = int(0.9 * len(file_paths))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Format for VAE: ((image, label), image)
    train_dataset = train_dataset.map(format_for_vae)
    val_dataset = val_dataset.map(format_for_vae)

    # Batch the datasets
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)

    # Cache datasets in memory for faster access
    train_dataset = train_dataset.cache()
    val_dataset = val_dataset.cache()

    # Prefetch for performance (overlap data loading and training)
    train_dataset = train_dataset.prefetch(prefetch_size)
    val_dataset = val_dataset.prefetch(prefetch_size)

    print(f"\n✓ Data pipeline created with parallel loading")
    print(f"  Training batches: ~{train_size // batch_size}")
    print(f"  Validation batches: ~{(len(file_paths) - train_size) // batch_size}")

    return train_dataset, val_dataset, train_size, len(file_paths) - train_size

# ============================================================================
# 3. SAMPLING LAYER FOR VAE
# ============================================================================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ============================================================================
# 4. KL DIVERGENCE LAYER (CUSTOM LAYER FOR FUNCTIONAL API)
# ============================================================================

class KLDivergenceLayer(layers.Layer):
    """Custom layer to compute KL divergence loss"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 
                axis=1
            )
        )
        self.add_loss(kl_loss)
        return z_mean  # Pass through z_mean (not used, just for connectivity)

# ============================================================================
# 5. SIMPLIFIED CVAE ARCHITECTURE
# ============================================================================

def build_simplified_cvae(image_size, latent_dim, num_classes):
    """
    Build SIMPLIFIED CVAE with FEWER layers for faster training
    """

    # ========== ENCODER (SIMPLIFIED) ==========
    image_input = layers.Input(shape=(*image_size, 3), name='image_input')
    class_input = layers.Input(shape=(num_classes,), name='class_input')

    # Flatten image
    img_flat = layers.Flatten()(image_input)

    # Concatenate image and class label
    combined = layers.Concatenate()([img_flat, class_input])

    # SIMPLIFIED encoder: 2 layers instead of 4
    x = layers.Dense(1024, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Latent space
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # Add KL divergence loss using custom layer
    _ = KLDivergenceLayer()([z_mean, z_log_var])

    # Sample from latent space
    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(
        inputs=[image_input, class_input],
        outputs=[z_mean, z_log_var, z],
        name='encoder'
    )

    # ========== DECODER (SIMPLIFIED) ==========
    latent_input = layers.Input(shape=(latent_dim,), name='latent_input')
    decoder_class_input = layers.Input(shape=(num_classes,), name='decoder_class_input')

    decoder_combined = layers.Concatenate()([latent_input, decoder_class_input])

    # SIMPLIFIED decoder: 2 layers instead of 4
    d = layers.Dense(512, activation='relu')(decoder_combined)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.2)(d)

    d = layers.Dense(1024, activation='relu')(d)
    d = layers.BatchNormalization()(d)

    # Output layer
    img_flat_dim = image_size[0] * image_size[1] * 3
    d = layers.Dense(img_flat_dim, activation='sigmoid')(d)
    decoder_output = layers.Reshape((*image_size, 3))(d)

    decoder = models.Model(
        inputs=[latent_input, decoder_class_input],
        outputs=decoder_output,
        name='decoder'
    )

    # ========== FULL VAE MODEL ==========
    z_mean_out, z_log_var_out, z_out = encoder([image_input, class_input])
    reconstruction = decoder([z_out, class_input])

    vae = models.Model(
        inputs=[image_input, class_input],
        outputs=reconstruction,
        name='cvae'
    )

    return encoder, decoder, vae

# ============================================================================
# 6. LOSS FUNCTION
# ============================================================================

def vae_reconstruction_loss(y_true, y_pred):
    """Reconstruction loss (binary crossentropy)"""
    return tf.reduce_mean(
        tf.reduce_sum(
            keras.losses.binary_crossentropy(y_true, y_pred),
            axis=(1, 2)
        )
    )

# ============================================================================
# 7. TRAINING WITH GPU
# ============================================================================

def train_cvae_optimized(config):
    """Train CVAE with GPU optimization and parallel data loading"""

    # Setup GPU
    has_gpu = setup_gpu()

    # Create optimized data pipeline
    train_dataset, val_dataset, train_size, val_size = create_tf_dataset(
        config['excel_path'],
        config['data_path'],
        config['image_size'],
        config['batch_size'],
        config['num_parallel_calls'],
        config['prefetch_size']
    )

    print(f"\nTraining samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Build simplified CVAE
    print("\nBuilding SIMPLIFIED CVAE architecture...")
    encoder, decoder, vae = build_simplified_cvae(
        config['image_size'],
        config['latent_dim'],
        config['num_classes']
    )

    print("\nEncoder summary:")
    encoder.summary()
    print("\nDecoder summary:")
    decoder.summary()

    # Compile with mixed precision for faster GPU training
    if has_gpu:
        print("\n✓ Enabling mixed precision for faster GPU training")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    vae.compile(
        optimizer=keras.optimizers.Adam(config['learning_rate']),
        loss=vae_reconstruction_loss,
        metrics=['mse']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'cvae_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    print("\n" + "="*80)
    print("STARTING TRAINING ON", "GPU" if has_gpu else "CPU")
    print("="*80)

    # Train
    # Dataset is now formatted as: ((image, label), image)
    # which matches the VAE's expectation: inputs=[image, label], output=image
    history = vae.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Save models
    encoder.save('cvae_encoder.keras')
    decoder.save('cvae_decoder.keras')
    vae.save('cvae_full_model.keras')

    print("\n✓ Models saved:")
    print("  - cvae_encoder.keras")
    print("  - cvae_decoder.keras")
    print("  - cvae_full_model.keras")
    print("  - cvae_best.keras")

    return encoder, decoder, vae, history

# ============================================================================
# 8. PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('CVAE Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if 'mse' in history.history:
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Reconstruction MSE')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('cvae_training_history.png', dpi=150)
    print("\n✓ Training history saved: cvae_training_history.png")
    plt.close()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("OPTIMIZED CVAE - GPU + PARALLEL LOADING + SIMPLIFIED ARCHITECTURE")
    print("="*80)
    print("\nOptimizations:")
    print("  ✓ GPU training with memory growth")
    print("  ✓ Mixed precision for faster training")
    print("  ✓ Parallel image loading with tf.data")
    print("  ✓ Prefetching and caching")
    print("  ✓ Simplified architecture (fewer layers)")
    print("  ✓ Reduced latent dimension (128 → 64)")
    print("  ✓ Larger batch size (64 → 128)")
    print("="*80)

    # Train CVAE
    encoder, decoder, vae, history = train_cvae_optimized(CONFIG)

    # Plot history
    plot_training_history(history)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext: Run generate_synthetic_data.py to create balanced dataset")
    print("="*80)
