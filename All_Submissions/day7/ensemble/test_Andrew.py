# ------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
import argparse
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
import json

# ------------------------------------------------------------------------------------------------------------------

'''
Testing script for ENSEMBLE MODEL with Test-Time Augmentation (TTA)

Loads trained ensemble model (3 architectures + weighted voting)
Applies TTA for robust predictions
Evaluates performance with comprehensive metrics
'''

# ------------------------------------------------------------------------------------------------------------------

CHANNELS = 3
IMAGE_SIZE = 128  # Must match training script
BATCH_SIZE = 96  # Match training script
USE_TTA = True  # Change to False to disable TTA
N_TTA_AUGMENTATIONS = 3  # Number of augmented versions per test image

## Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train
args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

NICKNAME = 'Andrew'

# ------------------------------------------------------------------------------------------------------------------
# CUSTOM LAYERS - MUST BE DEFINED BEFORE LOADING MODEL
# ------------------------------------------------------------------------------------------------------------------

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

    @classmethod
    def from_config(cls, config):
        """Required for proper deserialization in Keras 3.x"""
        return cls(**config)

# ------------------------------------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------------------------------------

def aspect_resize_pad(img):
    """Preserve aspect ratio then pad to square"""
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

# ------------------------------------------------------------------------------------------------------------------

def heavy_augment(img, is_training=True):
    """Heavy augmentation for TTA"""
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

# ------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:
        x = lambda x: tf.argmax(x == class_names).numpy()
        final_target = xdf_data['target'].apply(x)
        final_target = to_categorical(list(final_target))
        xfinal = []
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal
        xdf_data['target_class'] = final_target

    return class_names

# ------------------------------------------------------------------------------------------------------------------

def process_path(path, label):
    """Process image for testing (NO augmentation)"""
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    img = aspect_resize_pad(img)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.reshape(img, [-1]), label

# ------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))
    y_target = np.array(y_target.tolist())
    return y_target

# ------------------------------------------------------------------------------------------------------------------

def read_data(num_classes):
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)
    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))
    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    final_ds = final_ds.prefetch(AUTOTUNE)
    return final_ds

# ------------------------------------------------------------------------------------------------------------------

def save_model(model):
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\\n'))

# ------------------------------------------------------------------------------------------------------------------
# TEST-TIME AUGMENTATION
# ------------------------------------------------------------------------------------------------------------------

def create_tta_dataset(test_ds, n_augmentations=N_TTA_AUGMENTATIONS):
    """
    Create TTA dataset pipeline with prefetching
    GPU preprocessing happens in parallel with prediction!
    """

    def augment_batch(images, labels):
        """Generate augmented versions of batch"""
        batch_size = tf.shape(images)[0]

        # Reshape to 2D images
        images_2d = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

        # Original images
        all_versions = [images]

        # Generate augmented versions
        for _ in range(n_augmentations):
            # Vectorized augmentation on entire batch
            augmented = tf.map_fn(
                lambda img: heavy_augment(img, is_training=True),
                images_2d,
                parallel_iterations=32,
                dtype=tf.float32
            )
            # Flatten back
            augmented_flat = tf.reshape(augmented, [batch_size, -1])
            all_versions.append(augmented_flat)

        # Stack: (n_aug+1, batch_size, features)
        all_versions = tf.stack(all_versions, axis=0)

        # Reshape to: ((n_aug+1)*batch_size, features)
        total_samples = (n_augmentations + 1) * batch_size
        all_versions_flat = tf.reshape(all_versions, [total_samples, -1])

        # Create labels for tracking which augmentation belongs to which image
        batch_indices = tf.tile(tf.range(batch_size), [n_augmentations + 1])

        return all_versions_flat, batch_indices

    # Apply augmentation and prefetch
    tta_ds = test_ds.map(
        augment_batch,
        num_parallel_calls=AUTOTUNE  # Parallel preprocessing
    ).prefetch(AUTOTUNE)  # Overlap preprocessing with prediction

    return tta_ds


def predict_func(test_ds, use_tta=True):
    """
    Ultimate optimized prediction with proper batch tracking
    Fixed: Correctly handles all images in dataset
    """
    print("\\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    # Define custom objects
    custom_objects = {'Swish': Swish, 'EnsembleModel': EnsembleModel}

    # Load individual models
    models = []
    # model_names = ['deep_sequential_mlp', 'residual_mlp', 'autoencoder_mlp']

    model_names = ['Andrew1', 'Andrew2', 'Andrew3']
    for model_name in model_names:
        try:
            model = tf.keras.models.load_model(
                f'model_{model_name}.keras',
                custom_objects=custom_objects
            )
            models.append(model)
            print(f"✅ Loaded: model_{model_name}.keras")
        except Exception as e:
            print(f"⚠️  Could not load model_{model_name}.keras: {e}")
            models.append(None)

    # Load ensemble weights
    try:
        with open(f'ensemble_weights_{NICKNAME}.json', 'r') as f:
            ensemble_weights = json.load(f)
        print("\\nEnsemble weights:")
        for name, weight in ensemble_weights.items():
            print(f"  {name}: {weight:.4f}")

        weights = [
            ensemble_weights.get('deep_sequential_mlp', 0.33),
            ensemble_weights.get('residual_mlp', 0.33),
            ensemble_weights.get('autoencoder_mlp', 0.34)
        ]
    except FileNotFoundError:
        print("\\nUsing default equal weights (0.33, 0.33, 0.34)")
        weights = [0.33, 0.33, 0.34]

    # Normalize weights
    valid_models = [m for m in models if m is not None]
    if len(valid_models) == 0:
        raise RuntimeError("No models could be loaded!")

    if len(valid_models) < len(models):
        total_weight = sum([w for m, w in zip(models, weights) if m is not None])
        weights = [w / total_weight if m is not None else 0 for m, w in zip(models, weights)]

    print(f"\\nSuccessfully loaded {len(valid_models)}/{len(models)} models")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\\n🚀 GPU acceleration: {len(gpus)} GPU(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass

    # Make predictions
    if use_tta:
        print(f"\\n{'=' * 70}")
        print(f"TTA PREDICTION (GPU-OPTIMIZED)")
        print(f"{'=' * 70}")
        print(f"\\nConfiguration:")
        print(f"  Images: {len(xdf_dset)}")
        print(f"  Augmentations per image: {N_TTA_AUGMENTATIONS}")
        print(f"  Processing: Batched with prefetch\\n")

        all_predictions = []  # Store predictions in order
        processed = 0

        # Process each batch with TTA
        for batch_images, _ in test_ds:
            batch_size = batch_images.shape[0]

            # Convert to tensor for GPU operations
            if not isinstance(batch_images, tf.Tensor):
                batch_images = tf.constant(batch_images, dtype=tf.float32)

            # Reshape to 2D for augmentation
            images_2d = tf.reshape(batch_images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

            # Generate augmentations (vectorized)
            all_versions = [batch_images]  # Original

            for _ in range(N_TTA_AUGMENTATIONS):
                # Augment entire batch at once
                augmented = tf.map_fn(
                    lambda img: heavy_augment(img, is_training=True),
                    images_2d,
                    parallel_iterations=32,
                    dtype=tf.float32
                )
                augmented_flat = tf.reshape(augmented, [batch_size, -1])
                all_versions.append(augmented_flat)

            # Stack and reshape: (n_aug+1, batch_size, features) -> (total, features)
            all_versions = tf.stack(all_versions, axis=0)
            total_samples = (N_TTA_AUGMENTATIONS + 1) * batch_size
            all_versions_flat = tf.reshape(all_versions, [total_samples, -1])

            # Ensemble prediction on all augmented versions
            batch_preds = []
            for model, weight in zip(models, weights):
                if model is not None:
                    pred = model.predict(all_versions_flat, batch_size=1024, verbose=0)
                    if isinstance(pred, list):
                        pred = pred[0]
                    batch_preds.append(pred * weight)

            ensemble_pred = np.sum(batch_preds, axis=0)

            # Reshape back: (n_aug+1, batch_size, n_classes)
            ensemble_pred = ensemble_pred.reshape(N_TTA_AUGMENTATIONS + 1, batch_size, -1)

            # Average over augmentations: (batch_size, n_classes)
            batch_final = np.mean(ensemble_pred, axis=0)

            # Store predictions for this batch
            all_predictions.append(batch_final)

            processed += batch_size
            print(f"  Processed: {processed}/{len(xdf_dset)} images", end='\\r')

        print(f"  Processed: {processed}/{len(xdf_dset)} images - Complete!")

        # Concatenate all batch predictions
        all_predictions = np.vstack(all_predictions)

        # Truncate to exact dataset size (in case of padding in last batch)
        all_predictions = all_predictions[:len(xdf_dset)]

    else:
        print(f"\\n{'=' * 70}")
        print(f"FAST PREDICTION (NO TTA)")
        print(f"{'=' * 70}")

        all_predictions = []
        processed = 0

        for batch_images, _ in test_ds:
            # Ensemble prediction
            batch_preds = []
            for model, weight in zip(models, weights):
                if model is not None:
                    pred = model.predict(batch_images, batch_size=1024, verbose=0)
                    if isinstance(pred, list):
                        pred = pred[0]
                    batch_preds.append(pred * weight)

            batch_pred = np.sum(batch_preds, axis=0)
            all_predictions.append(batch_pred)

            processed += batch_images.shape[0]
            print(f"  Processed: {processed}/{len(xdf_dset)} images", end='\\r')

        print(f"  Processed: {processed}/{len(xdf_dset)} images - Complete!")
        all_predictions = np.vstack(all_predictions)

        # Truncate to exact size
        all_predictions = all_predictions[:len(xdf_dset)]

    # Verify length matches
    assert len(all_predictions) == len(xdf_dset), \
        f"Prediction count mismatch: {len(all_predictions)} vs {len(xdf_dset)}"

    # Get class labels
    xres = [np.argmax(pred) for pred in all_predictions]

    # Verify again before assignment
    assert len(xres) == len(xdf_dset), \
        f"Results count mismatch: {len(xres)} vs {len(xdf_dset)}"

    # Save results
    xdf_dset['results'] = xres
    output_filename = f'results_{NICKNAME}.xlsx'
    xdf_dset.to_excel(output_filename, index=False)

    print(f"\\n✅ Results saved to: {output_filename}")

    # Print distribution
    unique, counts = np.unique(xres, return_counts=True)
    print("\\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} predictions ({100 * count / len(xres):.1f}%)")

    print("=" * 70)


# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    """
    Calculate evaluation metrics
    Returns composite score (accuracy + kappa) / 2
    """

    def f1_score_metric(y_true, y_pred, type):
        res = f1_score(y_true, y_pred, average=type)
        print(f"  F1 Score ({type}): {res:.4f}")
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print(f"  Cohen's Kappa: {res:.4f}")
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print(f"  Accuracy: {res:.4f}")
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print(f"  Matthews Correlation: {res:.4f}")
        return res

    # Convert to class labels
    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    xcont = 0
    xsum = 0

    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    # Store accuracy and kappa for composite score
    acc_value = None
    kappa_value = None

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
            kappa_value = xmet
        elif xm == 'acc':
            xmet = accuracy_metric(y_true, y_pred)
            acc_value = xmet
        elif xm == 'mat':
            xmet = matthews_metric(y_true, y_pred)
        else:
            print(f'  Metric {xm} does not exist')
            continue

        xsum += xmet
        xcont += 1

    print("\n" + "-" * 70)

    if 'sum' in aggregates and xcont > 0:
        print(f"  Sum of Metrics: {xsum:.4f}")

    if 'avg' in aggregates and xcont > 0:
        avg_score = xsum / xcont
        print(f"  Average of Metrics: {avg_score:.4f}")

    # Calculate and display composite score
    if acc_value is not None and kappa_value is not None:
        composite = (acc_value + kappa_value) / 2
        print(f"\n  🎯 Composite Score (Acc + Kappa)/2: {composite:.4f}")

    print("=" * 70)

# ------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, xdf_dset

    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL + TTA TESTING SCRIPT")
    print("=" * 70)

    # Reading the excel file
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Load and process data
    xdf_data = pd.read_excel(FILE_NAME)
    class_names = process_target(1)

    print(f"\nDataset Information:")
    print(f"  Total images: {len(xdf_data)}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Class names: {class_names}")
    print(f"  Split: {SPLIT}")

    # Filter by split
    xdf_dset = xdf_data[xdf_data["split"] == SPLIT].copy()
    print(f"  {SPLIT} set size: {len(xdf_dset)}")

    # Class distribution
    print(f"\nClass distribution in {SPLIT} set:")
    class_dist = xdf_dset['target'].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} images")

    # Process dataset
    test_ds = read_data(len(class_names))

    # Predict with TTA (set to False for faster testing without TTA)

    predict_func(test_ds, use_tta=USE_TTA)

    # Calculate metrics
    list_of_metrics = ['acc', 'coh', 'f1_macro', 'f1_weighted']
    list_of_agg = ['avg', 'sum']
    metrics_func(list_of_metrics, list_of_agg)

    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------------------------
