#------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers
import keras

#------------------------------------------------------------------------------------------------------------------

'''
OPTIMIZED SUPERVISED AUTOENCODER-MLP ARCHITECTURE:

Key Improvements:
1. ✅ Focal Loss for handling class imbalance (gamma=2.0, alpha=0.25)
2. ✅ Deep-narrow architecture for better feature learning with fewer parameters
3. ✅ Reduced image size (96x96) optimized for 500x400 original images
4. ✅ Fixed Swish serialization with @keras.saving.register_keras_serializable
5. ✅ Memory-efficient streaming data pipeline

Architecture Philosophy:
- DEEP (9+ encoder layers): Learns hierarchical features progressively
- NARROW (256→128→64): Fewer parameters, less overfitting
- INFORMATION BOTTLENECK: Forces learning of compressed representations

Input: 96x96x3 images (27,648 dims) → Encoder → 32D bottleneck → Classifier → 10 classes
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

n_epoch = 200  # Increased for deep network
BATCH_SIZE = 128

## Image processing - OPTIMIZED for 500x400 images
CHANNELS = 3
IMAGE_SIZE = 150      # Reduced from 128 → 44% less features, faster training
NICKNAME = 'Andrew'

## DEEP-NARROW Architecture (Inspired by ResNet/DenseNet principles)
# Encoder: Progressive compression with many layers
ENCODER_LAYERS = [256, 256, 128, 128, 64, 64, 32]  # 7 layers, narrow width

# Decoder: Mirror encoder for reconstruction
DECODER_LAYERS = list(reversed(ENCODER_LAYERS[:-1]))  # [64, 64, 128, 128, 256, 256]

# MLP Classifier: Deep processing of concatenated features
MLP_LAYERS = [400, 300, 100, 100, 80, 64, 32]  # 4 layers

# Regularization parameters
GAUSSIAN_NOISE_STD = 0.1
ENCODER_DROPOUT = 0.1      # Lighter dropout in encoder
MLP_DROPOUT = [0.2, 0.3, 0.4, 0.4, 0.45, 0.5, 0.55]  # Heavier dropout in classifier
L2_REG = 0.0001  # Reduced L2 since we have more depth

# Training parameters
LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.1
AUTOENCODER_LOSS_WEIGHT = 0.3 # Lower weight for reconstruction

# FOCAL LOSS PARAMETERS (KEY FOR IMBALANCED DATA)
FOCAL_GAMMA = 2.0 # Focusing parameter (standard) default = 2.0
FOCAL_ALPHA = 0.25 # Class balancing weight default = 0.25

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

    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        xdepth = len(class_names)
        final_target = tf.one_hot(target, xdepth)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
        final_target = xfinal
        xdf_data['target_class'] = final_target

    if target_type == 3:
        pass

    return class_names

#------------------------------------------------------------------------------------------------------------------

def aspect_resize_pad(img):
    # Preserve aspect ratio then pad to square
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

def process_path(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    img = aspect_resize_pad(img)
    img = tf.cast(img, tf.float32) / 255.0

    return tf.reshape(img, [-1]), label

#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))
    y_target = np.array(y_target.tolist())

    return y_target

#------------------------------------------------------------------------------------------------------------------

def read_data(num_classes):
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets))

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    final_ds = final_ds.prefetch(AUTOTUNE)

    return final_ds

#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\\n'))

#------------------------------------------------------------------------------------------------------------------
# CUSTOM SWISH ACTIVATION - FIXED FOR PROPER SERIALIZATION
#------------------------------------------------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="AutoencoderMLP")
class Swish(layers.Layer):
    """
    Swish activation: x * sigmoid(beta * x)
    Fixed with proper from_config for Keras 3.x compatibility
    """
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
        """Required for proper deserialization in Keras 3.x"""
        return cls(**config)

#------------------------------------------------------------------------------------------------------------------
# DEEP-NARROW AUTOENCODER-MLP MODEL WITH FOCAL LOSS
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    """
    Deep-Narrow Supervised Autoencoder-MLP with Focal Loss

    Architecture:
    - Input: 96×96×3 = 27,648 dimensions
    - Encoder: 7 deep layers → 32D bottleneck
    - Decoder: 6 layers → reconstruct input
    - Classifier: Concat(bottleneck + input) → 4 deep MLP layers → 10 classes

    Total depth: 17+ layers (very deep for MLP!)
    """

    inputs = tf.keras.Input(shape=(INPUTS_r,), name='input_layer')

    print(f"\nBuilding DEEP-NARROW Architecture:")
    print(f"  Input dimension: {INPUTS_r}")
    print(f"  Encoder layers: {len(ENCODER_LAYERS)} (depths: {ENCODER_LAYERS})")
    print(f"  Bottleneck: {ENCODER_LAYERS[-1]}D")
    print(f"  MLP layers: {len(MLP_LAYERS)} (depths: {MLP_LAYERS})")

    # ============= ENCODER BRANCH (DEEP COMPRESSION) =============
    encoder_input = layers.GaussianNoise(GAUSSIAN_NOISE_STD, name='gaussian_noise')(inputs)

    x_encoder = encoder_input
    for i, units in enumerate(ENCODER_LAYERS):
        x_encoder = layers.Dense(
            units, 
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f'encoder_dense_{i}'
        )(x_encoder)
        x_encoder = layers.BatchNormalization(name=f'encoder_bn_{i}')(x_encoder)
        x_encoder = Swish(name=f'encoder_swish_{i}')(x_encoder)
        x_encoder = layers.Dropout(ENCODER_DROPOUT, name=f'encoder_dropout_{i}')(x_encoder)

    encoded = x_encoder  # Bottleneck representation

    # ============= DECODER BRANCH (RECONSTRUCTION) =============
    x_decoder = encoded
    for i, units in enumerate(DECODER_LAYERS):
        x_decoder = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f'decoder_dense_{i}'
        )(x_decoder)
        x_decoder = layers.BatchNormalization(name=f'decoder_bn_{i}')(x_decoder)
        x_decoder = Swish(name=f'decoder_swish_{i}')(x_decoder)
        x_decoder = layers.Dropout(ENCODER_DROPOUT, name=f'decoder_dropout_{i}')(x_decoder)

    reconstructed = layers.Dense(
        INPUTS_r, 
        activation='sigmoid',
        name='reconstruction_output'
    )(x_decoder)

    # ============= CLASSIFICATION BRANCH (DEEP MLP) =============
    concatenated = layers.Concatenate(name='concat_encoded_original')([encoded, inputs])

    x_mlp = concatenated
    for i, (units, dropout_rate) in enumerate(zip(MLP_LAYERS, MLP_DROPOUT)):
        x_mlp = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f'mlp_dense_{i}'
        )(x_mlp)
        x_mlp = layers.BatchNormalization(name=f'mlp_bn_{i}')(x_mlp)
        x_mlp = Swish(name=f'mlp_swish_{i}')(x_mlp)
        x_mlp = layers.Dropout(dropout_rate, name=f'mlp_dropout_{i}')(x_mlp)

    classification_output = layers.Dense(
        OUTPUTS_a,
        activation='softmax',
        kernel_regularizer=regularizers.l2(L2_REG),
        name='classification_output'
    )(x_mlp)

    # ============= CREATE MODEL WITH FOCAL LOSS =============
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[classification_output, reconstructed],
        name='deep_narrow_autoencoder_mlp'
    )

    # Compile with FOCAL LOSS for imbalanced data
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'classification_output': tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=FOCAL_ALPHA,           # Class balancing (0.25 = focus on minority)
                gamma=FOCAL_GAMMA,            # Focusing parameter (2.0 = standard)
                from_logits=False,            # Using softmax output
                label_smoothing=LABEL_SMOOTHING,
                name='focal_loss'
            ),
            'reconstruction_output': 'mse'
        },
        loss_weights={
            'classification_output': 1.0,
            'reconstruction_output': AUTOENCODER_LOSS_WEIGHT
        },
        metrics={
            'classification_output': ['accuracy'],
            'reconstruction_output': ['mae']
        }
    )

    # Calculate and print parameter count
    total_params = model.count_params()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size: {total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"  Total layers: {len(model.layers)}")

    model.summary()
    save_model(model)

    return model

#------------------------------------------------------------------------------------------------------------------

def calculate_class_weights_from_dataset(train_ds):
    """Calculate class weights from tf.Dataset WITHOUT loading all data into memory"""
    print("\nCalculating class weights from training data...")

    class_counts = np.zeros(OUTPUTS_a)
    total_samples = 0

    for _, labels_batch in train_ds:
        class_counts += np.sum(labels_batch.numpy(), axis=0)
        total_samples += labels_batch.shape[0]

    # Calculate weights
    class_weights = {}
    for i in range(OUTPUTS_a):
        class_weights[i] = total_samples / (OUTPUTS_a * class_counts[i]) if class_counts[i] > 0 else 1.0

    print(f"Total training samples: {int(total_samples)}")
    print(f"\nClass distribution and weights:")
    for i in range(OUTPUTS_a):
        print(f"  Class {i}: {int(class_counts[i])} samples ({100*class_counts[i]/total_samples:.1f}%) → weight: {class_weights[i]:.4f}")

    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")

    if imbalance_ratio > 3:
        print("Severe class imbalance detected! Focal Loss will help significantly.")

    return class_weights

#------------------------------------------------------------------------------------------------------------------

def add_sample_weights_to_dataset(ds, class_weights):
    """Add sample weights to tf.Dataset for handling class imbalance"""
    def add_weights(images, labels):
        class_indices = tf.argmax(labels, axis=1)
        weight_map = tf.constant([class_weights[i] for i in range(OUTPUTS_a)], dtype=tf.float32)
        classification_weights = tf.gather(weight_map, class_indices)
        reconstruction_weights = tf.ones_like(classification_weights)

        return images, (labels, images), (classification_weights, reconstruction_weights)

    return ds.map(add_weights, num_parallel_calls=AUTOTUNE)

#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds):
    """Memory-efficient training with Focal Loss and sample weights"""

    print("\n" + "="*70)
    print("DEEP-NARROW AUTOENCODER-MLP TRAINING WITH FOCAL LOSS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE} (optimized for 500×400 originals)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Focal Loss: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    print(f"  Architecture: Deep-Narrow (many layers, narrow width)")
    print(f"  Memory: Streaming data pipeline (low RAM usage)")

    def prepare_multitask_dataset(ds):
        """Convert single-output to multi-output format"""

        def convert_batch(images, labels):
            # Model expects TWO targets: [classification, reconstruction]
            # MUST use tuple, not list!
            return images, (labels, images)

        return ds.map(convert_batch, num_parallel_calls=AUTOTUNE)

    # Calculate class weights
    # class_weights = calculate_class_weights_from_dataset(train_ds)

    # Add sample weights to datasets
    # train_ds_weighted = add_sample_weights_to_dataset(train_ds, class_weights)
    # val_ds_weighted = add_sample_weights_to_dataset(val_ds, class_weights)

    # Optimize dataset performance


    # Apply transformation
    print("\n  Preparing multi-task datasets...")
    train_ds_multitask = prepare_multitask_dataset(train_ds)
    val_ds_multitask = prepare_multitask_dataset(val_ds)

    # Optimize dataset performance
    train_ds_multitask = train_ds_multitask.prefetch(AUTOTUNE)
    val_ds_multitask = val_ds_multitask.prefetch(AUTOTUNE)

    # Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_classification_output_loss',
        patience=20,  # Increased patience for deep network
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    check_point = tf.keras.callbacks.ModelCheckpoint(
        'model_{}.keras'.format(NICKNAME),
        monitor='val_classification_output_loss',  # ← Monitor composite metric
        mode='min',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_classification_output_loss',
        factor=0.25,
        patience=8,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs_{}'.format(NICKNAME),
        histogram_freq=1
    )

    # Clear session and build model
    tf.keras.backend.clear_session()
    final_model = model_definition()

    print(f"\nStarting training...")
    print(f"Expected memory per batch: ~{(BATCH_SIZE * INPUTS_r * 4) / (1024**2):.1f} MB")

    # Train
    history = final_model.fit(
        train_ds_multitask,
        validation_data=val_ds_multitask,
        epochs=n_epoch,
        callbacks=[early_stop, check_point, reduce_lr, tensorboard],
        verbose=1
    )

    return history

#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    """Predict function - memory efficient"""
    custom_objects = {'Swish': Swish}
    final_model = keras.models.load_model(
        'model_{}.keras'.format(NICKNAME),
        custom_objects=custom_objects
    )

    # Predict in batches
    all_predictions = []
    for images, _ in test_ds:
        predictions = final_model.predict(images, verbose=0)
        classification_predictions = predictions[0]
        all_predictions.append(classification_predictions)

    all_predictions = np.vstack(all_predictions)
    xres = [tf.argmax(f).numpy() for f in all_predictions]

    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):

    def f1_score_metric(y_true, y_pred, type):
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    xcont = 1
    xsum = 0

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)

#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    # for file in os.listdir(PATH+os.path.sep + "excel"):
    #     if file[-5:] == '.xlsx':
    #         FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + "training" + os.path.sep + "train_test_cleaned.xlsx"

    xdf_data = pd.read_excel(FILE_NAME)
    class_names= process_target(1)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    print(f"\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(f"  Total images: {len(xdf_data)}")
    print(f"  Number of classes: {OUTPUTS_a}")
    print(f"  Class names: {class_names}")
    print(f"  Input dimension: {INPUTS_r:,} ({IMAGE_SIZE}×{IMAGE_SIZE}×{CHANNELS})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {n_epoch}")

    print(f"\nClass distribution in training data:")
    train_data = xdf_data[xdf_data["split"] == 'train']
    class_dist = train_data['target'].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} images")

    # Create datasets
    xdf_dset = train_data.copy()
    train_ds = read_data(OUTPUTS_a)

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    val_ds = read_data(OUTPUTS_a)

    # Train model
    history = train_func(train_ds, val_ds)

    # Predictions
    predict_func(val_ds)

    # Metrics
    list_of_metrics = ['f1_micro', 'f1_macro', 'f1_weighted', 'coh', 'acc']
    list_of_agg = ['avg','sum']
    metrics_func(list_of_metrics, list_of_agg)

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#------------------------------------------------------------------------------------------------------------------
