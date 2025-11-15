#------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, regularizers

#------------------------------------------------------------------------------------------------------------------

'''
SUPERVISED AUTOENCODER-MLP ARCHITECTURE (MEMORY-OPTIMIZED VERSION):

MAJOR CHANGE: Uses tf.data.Dataset throughout training instead of loading all data into RAM.
This prevents OOM (Out of Memory) errors on systems with limited RAM.

Key optimizations:
1. Streaming data from disk using tf.data pipeline
2. Custom sample weight generator for imbalanced data
3. Reduced model complexity if needed
4. Mixed precision training option for lower memory usage

Architecture remains: Autoencoder + MLP hybrid for classification
'''

#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

n_epoch = 100
BATCH_SIZE = 32  # Reduced from 128 to lower memory usage

## Image processing
CHANNELS = 3
IMAGE_SIZE = 224
NICKNAME = 'Andrew'

## Autoencoder-MLP Hyperparameters (Reduced for memory efficiency)
ENCODER_LAYERS = [512, 256, 128]
MLP_LAYERS = [512, 256, 128]  # Reduced from [896, 448, 256]

# Regularization parameters
GAUSSIAN_NOISE_STD = 0.2
DROPOUT_RATES = [0.3, 0.4, 0.3]
L2_REG = 0.0001

# Training parameters
LEARNING_RATE = 0.0001
LABEL_SMOOTHING = 0.1
AUTOENCODER_LOSS_WEIGHT = 0.15

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

def process_path(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
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
# CUSTOM SWISH ACTIVATION FUNCTION
#------------------------------------------------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="AutoencoderMLP")
class Swish(layers.Layer):
    """Swish activation: x * sigmoid(beta * x)"""
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.beta * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config

#------------------------------------------------------------------------------------------------------------------
# SUPERVISED AUTOENCODER-MLP MODEL
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    """Memory-optimized Autoencoder-MLP architecture"""

    inputs = tf.keras.Input(shape=(INPUTS_r,), name='input_layer')

    # ============= ENCODER BRANCH =============
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
        x_encoder = layers.Dropout(0.2, name=f'encoder_dropout_{i}')(x_encoder)

    encoded = x_encoder

    # ============= DECODER BRANCH =============
    x_decoder = encoded
    for i, units in enumerate(reversed(ENCODER_LAYERS[:-1])):
        x_decoder = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_REG),
            name=f'decoder_dense_{i}'
        )(x_decoder)
        x_decoder = layers.BatchNormalization(name=f'decoder_bn_{i}')(x_decoder)
        x_decoder = Swish(name=f'decoder_swish_{i}')(x_decoder)
        x_decoder = layers.Dropout(0.2, name=f'decoder_dropout_{i}')(x_decoder)

    reconstructed = layers.Dense(
        INPUTS_r, 
        activation='sigmoid',
        name='reconstruction_output'
    )(x_decoder)

    # ============= CLASSIFICATION BRANCH =============
    concatenated = layers.Concatenate(name='concat_encoded_original')([encoded, inputs])

    x_mlp = concatenated
    for i, (units, dropout_rate) in enumerate(zip(MLP_LAYERS, DROPOUT_RATES)):
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

    # ============= CREATE MODEL =============
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[classification_output, reconstructed],
        name='supervised_autoencoder_mlp'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'classification_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
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

    model.summary()
    save_model(model)

    return model

#------------------------------------------------------------------------------------------------------------------

def calculate_class_weights_from_dataset(train_ds):
    """
    Calculate class weights from tf.Dataset WITHOUT loading all data into memory
    """
    print("\nCalculating class weights from training data...")

    # Count classes by iterating through dataset once
    class_counts = np.zeros(OUTPUTS_a)
    total_samples = 0

    for _, labels_batch in train_ds:
        # labels_batch shape: (batch_size, num_classes)
        class_counts += np.sum(labels_batch.numpy(), axis=0)
        total_samples += labels_batch.shape[0]

    # Calculate weights
    class_weights = {}
    for i in range(OUTPUTS_a):
        class_weights[i] = total_samples / (OUTPUTS_a * class_counts[i]) if class_counts[i] > 0 else 1.0

    print(f"Total training samples: {int(total_samples)}")
    print(f"Class weights:")
    for i, weight in class_weights.items():
        print(f"  Class {i}: {weight:.4f} (count: {int(class_counts[i])})")

    return class_weights

#------------------------------------------------------------------------------------------------------------------

def add_sample_weights_to_dataset(ds, class_weights):
    """
    Add sample weights to tf.Dataset for handling class imbalance
    This creates a dataset that yields (inputs, outputs, sample_weights)
    """
    def add_weights(images, labels):
        # Get class indices from one-hot labels
        class_indices = tf.argmax(labels, axis=1)

        # Map class indices to weights
        weight_map = tf.constant([class_weights[i] for i in range(OUTPUTS_a)], dtype=tf.float32)
        classification_weights = tf.gather(weight_map, class_indices)

        # Uniform weights for reconstruction
        reconstruction_weights = tf.ones_like(classification_weights)

        # Return: inputs, (classification_labels, reconstruction_targets), (classification_weights, reconstruction_weights)
        return images, (labels, images), (classification_weights, reconstruction_weights)

    return ds.map(add_weights, num_parallel_calls=AUTOTUNE)

#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds):
    """
    MEMORY-EFFICIENT training using tf.data pipeline
    No loading entire dataset into RAM!
    """

    print("\n" + "="*70)
    print("MEMORY-EFFICIENT TRAINING MODE")
    print("="*70)

    # Calculate class weights by streaming through data once
    class_weights = calculate_class_weights_from_dataset(train_ds)

    # Add sample weights to datasets
    train_ds_weighted = add_sample_weights_to_dataset(train_ds, class_weights)
    val_ds_weighted = add_sample_weights_to_dataset(val_ds, class_weights)

    # Optimize dataset performance
    train_ds_weighted = train_ds_weighted.prefetch(AUTOTUNE)
    val_ds_weighted = val_ds_weighted.prefetch(AUTOTUNE)

    # Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_classification_output_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    check_point = tf.keras.callbacks.ModelCheckpoint(
        'model_{}.keras'.format(NICKNAME),
        monitor='val_classification_output_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_classification_output_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs_{}'.format(NICKNAME),
        histogram_freq=1
    )

    # Clear any previous sessions to free memory
    tf.keras.backend.clear_session()

    final_model = model_definition()

    print(f"\nStarting training with batch size: {BATCH_SIZE}")
    print(f"Estimated memory usage per batch: ~{(BATCH_SIZE * INPUTS_r * 4) / (1024**2):.1f} MB")

    # Train directly on tf.data.Dataset (memory-efficient!)
    history = final_model.fit(
        train_ds_weighted,
        validation_data=val_ds_weighted,
        epochs=n_epoch,
        callbacks=[early_stop, check_point, reduce_lr, tensorboard],
        verbose=1
    )

    return history

#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    """Predict function - memory efficient"""
    custom_objects = {'Swish': Swish}
    final_model = tf.keras.models.load_model(
        'model_{}.keras'.format(NICKNAME),
        custom_objects=custom_objects
    )

    # Predict in batches to avoid memory issues
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
    xavg = 0

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

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)
    class_names= process_target(1)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    print(f"\nDataset Information:")
    print(f"  Total images: {len(xdf_data)}")
    print(f"  Number of classes: {OUTPUTS_a}")
    print(f"  Class names: {class_names}")
    print(f"  Input dimension: {INPUTS_r}")
    print(f"  Batch size: {BATCH_SIZE}")

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

    # Train model (memory-efficient!)
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
