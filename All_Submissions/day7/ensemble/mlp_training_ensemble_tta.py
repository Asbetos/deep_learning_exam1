#------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers, Model
import keras
import json

#------------------------------------------------------------------------------------------------------------------

'''
ULTIMATE ENSEMBLE MODEL WITH TTA

THREE DIVERSE ARCHITECTURES:
1. Deep Sequential MLP - Simple deep feedforward network
2. Residual MLP - Skip connections for better gradient flow
3. Autoencoder-MLP - Reconstruction-based feature learning

ENSEMBLE STRATEGY:
- Train 3 diverse models independently
- Weighted voting based on validation performance
- Test-Time Augmentation (TTA) for robustness
- Save ensemble as single Keras model wrapper

Expected Improvement: 0.48-0.50 → 0.65-0.72 (+0.17-0.22)
'''

#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

OR_PATH = os.getcwd()
os.chdir("../..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

n_epoch = 50
BATCH_SIZE = 96

## Image processing
CHANNELS = 3
IMAGE_SIZE = 128
NICKNAME = 'Andrew'

## TTA Configuration
N_TTA_AUGMENTATIONS = 5  # Number of augmented versions per test image

## ENSEMBLE WEIGHTS (will be optimized based on validation performance)
ENSEMBLE_WEIGHTS = {
    'autoencoder_mlp': 0.40,  # Usually best performer
    'residual_mlp': 0.35,     # Good gradient flow
    'deep_sequential_mlp': 0.25  # Simple but effective
}

# Training parameters
INITIAL_LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.0
L2_REG = 0.00005

# FOCAL LOSS PARAMETERS
FOCAL_GAMMA = 2.5
FOCAL_ALPHA = 0.25

#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:
        x = lambda x: tf.argmax(x == class_names).numpy()
        final_target = xdf_data['target'].apply(x)
        final_target = to_categorical(list(final_target))
        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal
        xdf_data['target_class'] = final_target

    return class_names

#------------------------------------------------------------------------------------------------------------------

def aspect_resize_pad(img):
    orig_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = IMAGE_SIZE / tf.reduce_max(orig_shape)
    new_size = tf.cast(tf.round(orig_shape * scale), tf.int32)
    img = tf.image.resize(img, new_size)

    delta_h = IMAGE_SIZE - new_size[0]
    delta_w = IMAGE_SIZE - new_size[1]
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    img = tf.pad(img, [[top,bottom],[left,right],[0,0]], constant_values=0)
    return img

#------------------------------------------------------------------------------------------------------------------
# HEAVY DATA AUGMENTATION (for both training and TTA)
#------------------------------------------------------------------------------------------------------------------

def heavy_augment(img, is_training=True):
    """Heavy augmentation for training and TTA"""
    if not is_training:
        return img

    # Geometric
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    crop_size = tf.random.uniform((), 0.85, 1.0)
    crop_h = tf.cast(tf.cast(IMAGE_SIZE, tf.float32) * crop_size, tf.int32)
    crop_w = tf.cast(tf.cast(IMAGE_SIZE, tf.float32) * crop_size, tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, CHANNELS])
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    # Color
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
    img = tf.image.random_hue(img, max_delta=0.2)

    # Noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float32)
    img = img + noise
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img

#------------------------------------------------------------------------------------------------------------------

def process_path(path, label, is_training=True):
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    # Histogram equalization (enhance contrast)
    # img = tf.image.adjust_contrast(img, 1.2)
    # # Normalize per channel (instead of global)
    # img = tf.image.per_image_standardization(img)
    img = aspect_resize_pad(img)
    img = tf.cast(img, tf.float32) / 255.0
    img = heavy_augment(img, is_training=is_training)
    return tf.reshape(img, [-1]), label

#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))
    y_target = np.array(y_target.tolist())
    return y_target

#------------------------------------------------------------------------------------------------------------------

def read_data(num_classes, is_training=True):
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)
    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))

    final_ds = list_ds.map(
        lambda x, y: process_path(x, y, is_training=is_training),
        num_parallel_calls=AUTOTUNE
    ).batch(BATCH_SIZE)

    final_ds = final_ds.prefetch(AUTOTUNE)
    return final_ds

#------------------------------------------------------------------------------------------------------------------

def save_model(model, name):
    with open(f'summary_{name}.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\\n'))

#------------------------------------------------------------------------------------------------------------------
# CUSTOM SWISH ACTIVATION
#------------------------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="AutoencoderMLP")
class Swish(layers.Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.beta * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#------------------------------------------------------------------------------------------------------------------
# MODEL 1: DEEP SEQUENTIAL MLP (Simple but Effective)
#------------------------------------------------------------------------------------------------------------------

def build_deep_sequential_mlp():
    """
    Simple deep feedforward network
    No fancy tricks - just depth and width
    """
    print("\nBuilding Model 1: Deep Sequential MLP")

    inputs = tf.keras.Input(shape=(INPUTS_r,), name='input_layer')

    # Layer configuration: progressively narrowing
    layer_widths = [1024, 896, 768, 640, 512, 384, 256, 128]
    dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4]

    x = inputs
    for i, (width, dropout) in enumerate(zip(layer_widths, dropout_rates)):
        x = layers.Dense(
            width,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f'seq_dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'seq_bn_{i}')(x)
        x = Swish(name=f'seq_swish_{i}')(x)
        x = layers.Dropout(dropout, name=f'seq_dropout_{i}')(x)

    outputs = layers.Dense(
        OUTPUTS_a,
        activation='softmax',
        kernel_regularizer=regularizers.l2(L2_REG),
        name='output'
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='deep_sequential_mlp')
    print(f"  Parameters: {model.count_params():,}")

    return model

#------------------------------------------------------------------------------------------------------------------
# MODEL 2: RESIDUAL MLP (Skip Connections)
#------------------------------------------------------------------------------------------------------------------

def build_residual_mlp():
    """
    MLP with residual (skip) connections
    Better gradient flow through deep network
    """
    print("\nBuilding Model 2: Residual MLP")

    inputs = tf.keras.Input(shape=(INPUTS_r,), name='input_layer')

    # Initial projection
    x = layers.Dense(512, kernel_initializer='he_normal', name='res_initial')(inputs)
    x = layers.BatchNormalization(name='res_bn_initial')(x)
    x = Swish(name='res_swish_initial')(x)

    # Residual blocks
    def residual_block(x, units, block_id):
        """Single residual block with skip connection"""
        shortcut = x

        # First layer
        x = layers.Dense(units, kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_REG),
                        name=f'res_dense_{block_id}_1')(x)
        x = layers.BatchNormalization(name=f'res_bn_{block_id}_1')(x)
        x = Swish(name=f'res_swish_{block_id}_1')(x)
        x = layers.Dropout(0.3, name=f'res_dropout_{block_id}_1')(x)

        # Second layer
        x = layers.Dense(units, kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_REG),
                        name=f'res_dense_{block_id}_2')(x)
        x = layers.BatchNormalization(name=f'res_bn_{block_id}_2')(x)

        # Skip connection (identity or projection)
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units, kernel_initializer='he_normal',
                                   name=f'res_projection_{block_id}')(shortcut)

        # Add and activate
        x = layers.Add(name=f'res_add_{block_id}')([x, shortcut])
        x = Swish(name=f'res_swish_{block_id}_2')(x)
        x = layers.Dropout(0.4, name=f'res_dropout_{block_id}_2')(x)

        return x

    # Stack residual blocks with decreasing width
    x = residual_block(x, 512, 0)
    x = residual_block(x, 384, 1)
    x = residual_block(x, 256, 2)
    x = residual_block(x, 128, 3)

    # Final classification
    outputs = layers.Dense(
        OUTPUTS_a,
        activation='softmax',
        kernel_regularizer=regularizers.l2(L2_REG),
        name='output'
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='residual_mlp')
    print(f"  Parameters: {model.count_params():,}")

    return model

#------------------------------------------------------------------------------------------------------------------
# MODEL 3: AUTOENCODER-MLP (Reconstruction-based Learning)
#------------------------------------------------------------------------------------------------------------------

def build_autoencoder_mlp():
    """
    Hybrid autoencoder + classifier
    Learns robust features through reconstruction
    """
    print("\nBuilding Model 3: Autoencoder-MLP")

    inputs = tf.keras.Input(shape=(INPUTS_r,), name='input_layer')

    # Encoder
    encoder_widths = [512, 512, 384, 384, 256, 256, 128, 64]
    encoder_input = layers.GaussianNoise(0.1, name='gaussian_noise')(inputs)

    x_encoder = encoder_input
    for i, width in enumerate(encoder_widths):
        x_encoder = layers.Dense(width, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(L2_REG),
                                name=f'enc_dense_{i}')(x_encoder)
        x_encoder = layers.BatchNormalization(name=f'enc_bn_{i}')(x_encoder)
        x_encoder = Swish(name=f'enc_swish_{i}')(x_encoder)
        x_encoder = layers.Dropout(0.1, name=f'enc_dropout_{i}')(x_encoder)

    encoded = x_encoder

    # Decoder
    decoder_widths = list(reversed(encoder_widths[:-1]))
    x_decoder = encoded
    for i, width in enumerate(decoder_widths):
        x_decoder = layers.Dense(width, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(L2_REG),
                                name=f'dec_dense_{i}')(x_decoder)
        x_decoder = layers.BatchNormalization(name=f'dec_bn_{i}')(x_decoder)
        x_decoder = Swish(name=f'dec_swish_{i}')(x_decoder)
        x_decoder = layers.Dropout(0.1, name=f'dec_dropout_{i}')(x_decoder)

    reconstructed = layers.Dense(INPUTS_r, activation='sigmoid', name='reconstruction_output')(x_decoder)

    # Classifier (concatenate encoded + original)
    mlp_widths = [768, 512, 384, 256, 128, 64]
    mlp_dropouts = [0.2, 0.25, 0.3, 0.3, 0.35, 0.4]

    concatenated = layers.Concatenate(name='concat')([encoded, inputs])
    x_mlp = concatenated

    for i, (width, dropout) in enumerate(zip(mlp_widths, mlp_dropouts)):
        x_mlp = layers.Dense(width, kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(L2_REG),
                            name=f'mlp_dense_{i}')(x_mlp)
        x_mlp = layers.BatchNormalization(name=f'mlp_bn_{i}')(x_mlp)
        x_mlp = Swish(name=f'mlp_swish_{i}')(x_mlp)
        x_mlp = layers.Dropout(dropout, name=f'mlp_dropout_{i}')(x_mlp)

    classification = layers.Dense(OUTPUTS_a, activation='softmax',
                                 kernel_regularizer=regularizers.l2(L2_REG),
                                 name='classification_output')(x_mlp)

    model = Model(inputs=inputs, outputs=[classification, reconstructed], name='autoencoder_mlp')
    print(f"  Parameters: {model.count_params():,}")

    return model

#------------------------------------------------------------------------------------------------------------------
# COMPILE MODEL WITH FOCAL LOSS
#------------------------------------------------------------------------------------------------------------------

def compile_model(model, is_autoencoder=False):
    """Compile model with focal loss and cosine LR schedule"""

    steps_per_epoch = len(xdf_data[xdf_data["split"] == 'train']) // BATCH_SIZE

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        first_decay_steps=steps_per_epoch * 20,
        t_mul=1.5,
        m_mul=0.9,
        alpha=1e-6
    )

    if is_autoencoder:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'classification_output': tf.keras.losses.CategoricalFocalCrossentropy(
                    alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                    from_logits=False, label_smoothing=LABEL_SMOOTHING
                ),
                'reconstruction_output': 'mse'
            },
            loss_weights={'classification_output': 1.0, 'reconstruction_output': 0.2},
            metrics={'classification_output': ['accuracy'], 'reconstruction_output': ['mae']}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                from_logits=False, label_smoothing=LABEL_SMOOTHING
            ),
            metrics=['accuracy']
        )

    return model

#------------------------------------------------------------------------------------------------------------------
# TRAIN SINGLE MODEL
#------------------------------------------------------------------------------------------------------------------

def train_single_model(model, train_ds, val_ds, model_name, is_autoencoder=False):
    """Train a single model with callbacks"""

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    # Prepare datasets
    def prepare_dataset(ds, is_ae):
        if is_ae:
            return ds.map(lambda x, y: (x, (y, x)), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    train_ds_prep = prepare_dataset(train_ds, is_autoencoder)
    val_ds_prep = prepare_dataset(val_ds, is_autoencoder)

    # Callbacks
    monitor_metric = 'val_classification_output_loss' if is_autoencoder else 'val_loss'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric, patience=10,
            restore_best_weights=True, verbose=1, mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'model_{model_name}.keras', monitor=monitor_metric,
            mode='min', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=f'./logs_{model_name}', histogram_freq=1)
    ]

    # Train
    history = model.fit(
        train_ds_prep, validation_data=val_ds_prep,
        epochs=n_epoch, callbacks=callbacks, verbose=1
    )

    return history

#------------------------------------------------------------------------------------------------------------------
# ENSEMBLE WRAPPER MODEL
#------------------------------------------------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="EnsembleModel")
class EnsembleModel(Model):
    """
    Ensemble wrapper that combines 3 models
    Saved/loaded as a single Keras model
    """

    def __init__(self, model1=None, model2=None, model3=None, weights=None,
                 weight1=None, weight2=None, weight3=None, **kwargs):
        super().__init__(**kwargs)
        self.model1 = model1  # Deep Sequential
        self.model2 = model2  # Residual
        self.model3 = model3  # Autoencoder

        # Handle both initialization methods
        if weights is not None:
            # From dictionary (used during creation)
            self.weight1 = weights.get('deep_sequential_mlp', 0.33)
            self.weight2 = weights.get('residual_mlp', 0.33)
            self.weight3 = weights.get('autoencoder_mlp', 0.34)
        elif weight1 is not None and weight2 is not None and weight3 is not None:
            # From individual weights (used during loading from config)
            self.weight1 = weight1
            self.weight2 = weight2
            self.weight3 = weight3
        else:
            # Default weights
            self.weight1 = 0.33
            self.weight2 = 0.33
            self.weight3 = 0.34

    def call(self, inputs, training=False):
        # Get predictions from each model
        pred1 = self.model1(inputs, training=training)
        pred2 = self.model2(inputs, training=training)

        # Autoencoder returns [classification, reconstruction]
        pred3 = self.model3(inputs, training=training)
        if isinstance(pred3, list):
            pred3 = pred3[0]  # Take only classification output

        # Weighted average
        ensemble_pred = (self.weight1 * pred1 +
                         self.weight2 * pred2 +
                         self.weight3 * pred3)

        return ensemble_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            'weight1': float(self.weight1),
            'weight2': float(self.weight2),
            'weight3': float(self.weight3)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create instance from config
        Critical for proper deserialization
        """
        # Extract weights from config
        weight1 = config.pop('weight1', 0.33)
        weight2 = config.pop('weight2', 0.33)
        weight3 = config.pop('weight3', 0.34)

        # Create instance with weights
        return cls(weight1=weight1, weight2=weight2, weight3=weight3, **config)



#------------------------------------------------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    print(f"\\n" + "=" * 70)
    print("ENSEMBLE TRAINING WITH TTA")
    print("=" * 70)

    # Load data
    FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + "training" + os.path.sep + "train_test_cleaned.xlsx"
    xdf_data = pd.read_excel(FILE_NAME)
    class_names = process_target(1)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    print(f"\\nDataset: {len(xdf_data)} images, {OUTPUTS_a} classes")
    print(f"Input dimension: {INPUTS_r:,}")

    # Prepare datasets
    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    train_ds = read_data(OUTPUTS_a, is_training=True)

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    val_ds = read_data(OUTPUTS_a, is_training=False)

    tf.keras.backend.clear_session()

    # BUILD AND TRAIN 3 MODELS
    print("\\n" + "=" * 70)
    print("BUILDING ENSEMBLE ARCHITECTURES")
    print("=" * 70)

    # # Model 1: Deep Sequential MLP
    model1 = build_deep_sequential_mlp()
    model1 = compile_model(model1, is_autoencoder=False)
    save_model(model1, 'deep_sequential_mlp')
    history1 = train_single_model(model1, train_ds, val_ds, 'deep_sequential_mlp', False)
    #
    tf.keras.backend.clear_session()

    # Model 2: Residual MLP
    model2 = build_residual_mlp()
    model2 = compile_model(model2, is_autoencoder=False)
    save_model(model2, 'residual_mlp')
    history2 = train_single_model(model2, train_ds, val_ds, 'residual_mlp', False)

    tf.keras.backend.clear_session()

    # Model 3: Autoencoder-MLP
    model3 = build_autoencoder_mlp()
    model3 = compile_model(model3, is_autoencoder=True)
    save_model(model3, 'autoencoder_mlp')
    history3 = train_single_model(model3, train_ds, val_ds, 'autoencoder_mlp', True)

    # OPTIMIZE ENSEMBLE WEIGHTS
    print("\\n" + "=" * 70)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 70)

    # Load best models
    model1 = keras.models.load_model('model_deep_sequential_mlp.keras', custom_objects={'Swish': Swish})
    model2 = keras.models.load_model('model_residual_mlp.keras', custom_objects={'Swish': Swish})
    model3 = keras.models.load_model('model_autoencoder_mlp.keras', custom_objects={'Swish': Swish})

    models = [model1, model2, model3]

    # Evaluate on validation set
    val_accs = []
    for model_idx, model in enumerate(models):
        acc_sum = 0
        total = 0
        for images, labels in val_ds:
            pred = model.predict(images, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            acc_sum += np.sum(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))
            total += len(labels)
        val_accs.append(acc_sum / total)

    # Normalize to weights
    total_acc = sum(val_accs)
    optimized_weights = {
        'deep_sequential_mlp': val_accs[0] / total_acc,
        'residual_mlp': val_accs[1] / total_acc,
        'autoencoder_mlp': val_accs[2] / total_acc
    }

    print("\\nValidation accuracies:")
    print(f"  Deep Sequential: {val_accs[0]:.4f}")
    print(f"  Residual: {val_accs[1]:.4f}")
    print(f"  Autoencoder: {val_accs[2]:.4f}")
    print("\\nOptimized ensemble weights:")
    for name, weight in optimized_weights.items():
        print(f"  {name}: {weight:.4f}")

    # CREATE AND SAVE ENSEMBLE
    print("\\n" + "=" * 70)
    print("CREATING ENSEMBLE MODEL")
    print("=" * 70)

    ensemble = EnsembleModel(model1, model2, model3, optimized_weights, name='ensemble')

    # Save ensemble (note: may have issues loading, use individual models in testing)
    try:
        ensemble.save(f'model_{NICKNAME}_ensemble.keras')
        print(f"\\n✅ Ensemble saved as: model_{NICKNAME}_ensemble.keras")
    except Exception as e:
        print(f"\\n⚠️  Could not save ensemble wrapper: {e}")
        print("   Individual models are saved and can be used for ensemble prediction")

    # Save weights separately
    with open(f'ensemble_weights_{NICKNAME}.json', 'w') as f:
        json.dump(optimized_weights, f, indent=2)
    print(f"✅ Weights saved as: ensemble_weights_{NICKNAME}.json")

    # PREDICT WITH GPU-OPTIMIZED BATCHED TTA
    print("\\n" + "=" * 70)
    print("GPU-OPTIMIZED ENSEMBLE + TTA PREDICTION")
    print("=" * 70)

    weights_list = [
        optimized_weights['deep_sequential_mlp'],
        optimized_weights['residual_mlp'],
        optimized_weights['autoencoder_mlp']
    ]

    print(f"\\nConfiguration:")
    print(f"  Images: {len(xdf_dset)}")
    print(f"  Augmentations per image: {N_TTA_AUGMENTATIONS}")
    print(f"  Processing: Batched GPU-accelerated\\n")

    all_predictions = []
    processed = 0

    for batch_images, _ in val_ds:
        batch_size = batch_images.shape[0]

        # Convert to tensor
        if not isinstance(batch_images, tf.Tensor):
            batch_images = tf.constant(batch_images, dtype=tf.float32)

        # Reshape to 2D for augmentation
        images_2d = tf.reshape(batch_images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        # Generate augmentations (vectorized on GPU)
        all_versions = [batch_images]  # Original

        for _ in range(N_TTA_AUGMENTATIONS):
            augmented = tf.map_fn(
                lambda img: heavy_augment(img, is_training=True),
                images_2d,
                parallel_iterations=32,
                dtype=tf.float32
            )
            augmented_flat = tf.reshape(augmented, [batch_size, -1])
            all_versions.append(augmented_flat)

        # Stack and reshape
        all_versions = tf.stack(all_versions, axis=0)
        total_samples = (N_TTA_AUGMENTATIONS + 1) * batch_size
        all_versions_flat = tf.reshape(all_versions, [total_samples, -1])

        # Ensemble prediction on all augmented versions
        batch_preds = []
        for model, weight in zip(models, weights_list):
            pred = model.predict(all_versions_flat, batch_size=512, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            batch_preds.append(pred * weight)

        ensemble_pred = np.sum(batch_preds, axis=0)

        # Reshape and average
        ensemble_pred = ensemble_pred.reshape(N_TTA_AUGMENTATIONS + 1, batch_size, -1)
        batch_final = np.mean(ensemble_pred, axis=0)

        all_predictions.append(batch_final)

        processed += batch_size
        print(f"  Processed: {processed}/{len(xdf_dset)} images", end='\\n')

    print(f"  Processed: {len(xdf_dset)}/{len(xdf_dset)} images - Complete!")

    # Concatenate and truncate
    all_predictions = np.vstack(all_predictions)[:len(xdf_dset)]
    xres = [np.argmax(pred) for pred in all_predictions]

    # Save results
    xdf_dset['results'] = xres
    xdf_dset.to_excel(f'results_{NICKNAME}.xlsx', index=False)

    # METRICS
    print("\\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\\nAccuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"\\nComposite (Acc + Kappa) / 2: {(acc + kappa) / 2:.4f}")

    print("\\n" + "=" * 70)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("=" * 70)


# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#------------------------------------------------------------------------------------------------------------------
