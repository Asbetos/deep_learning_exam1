# ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import argparse
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------------------------------------------------------------

'''
Testing script for MLP-Mixer model
Loads trained model with custom PatchEmbedding and MixerBlock layers
'''

# ------------------------------------------------------------------------------------------------------------------

CHANNELS = 3
IMAGE_SIZE = 128  # Must match training script
BATCH_SIZE = 128

# MLP-Mixer parameters (must match training script exactly)
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) * (IMAGE_SIZE // PATCH_SIZE)
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS

HIDDEN_DIM = 128  # Must match training
NUM_MIXER_BLOCKS = 6  # Must match training
TOKENS_MLP_DIM = 128  # Must match training
CHANNELS_MLP_DIM = 256  # Must match training
DROPOUT_RATE = 0.5  # Must match training
L2_REG = 0.001  # Must match training

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

@tf.keras.utils.register_keras_serializable(package="MLPMixer")
class PatchEmbedding(layers.Layer):
    """Extract and embed image patches with robust config handling"""

    def __init__(self,
                 num_patches=None,
                 patch_dim=None,
                 hidden_dim=None,
                 image_height=None,
                 image_width=None,
                 image_channels=None,
                 patch_size=None,
                 **kwargs):
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.patch_size = patch_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.projection = layers.Dense(
            self.hidden_dim,
            kernel_regularizer=regularizers.l2(L2_REG),
            name='projection'
        )
        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        images = tf.reshape(
            images,
            [-1, self.image_height, self.image_width, self.image_channels]
        )

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])
        embedded = self.projection(patches)
        return embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'patch_dim': self.patch_dim,
            'hidden_dim': self.hidden_dim,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'image_channels': self.image_channels,
            'patch_size': self.patch_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        init_params = {
            'num_patches': config.get('num_patches'),
            'patch_dim': config.get('patch_dim'),
            'hidden_dim': config.get('hidden_dim'),
            'image_height': config.get('image_height'),
            'image_width': config.get('image_width'),
            'image_channels': config.get('image_channels'),
            'patch_size': config.get('patch_size'),
        }
        parent_config = {k: v for k, v in config.items()
                         if k not in init_params and k != 'build_config'}
        final_config = {**init_params, **parent_config}
        return cls(**final_config)


@tf.keras.utils.register_keras_serializable(package="MLPMixer")
class MixerBlock(layers.Layer):
    """MLP-Mixer block with robust config handling and regularization"""

    def __init__(self,
                 num_patches=None,
                 hidden_dim=None,
                 tokens_mlp_dim=None,
                 channels_mlp_dim=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Layer normalizations
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='norm2')

        # Token-mixing with BatchNorm and L2 regularization
        self.token_dense1 = layers.Dense(
            self.tokens_mlp_dim,
            activation='gelu',
            kernel_regularizer=regularizers.l2(L2_REG),
            name='token_dense1'
        )
        self.token_bn1 = layers.BatchNormalization(name='token_bn1')
        self.token_dropout1 = layers.Dropout(self.dropout_rate, name='token_dropout1')
        self.token_dense2 = layers.Dense(
            self.num_patches,
            kernel_regularizer=regularizers.l2(L2_REG),
            name='token_dense2'
        )
        self.token_bn2 = layers.BatchNormalization(name='token_bn2')
        self.token_dropout2 = layers.Dropout(self.dropout_rate, name='token_dropout2')

        # Channel-mixing with BatchNorm and L2 regularization
        self.channel_dense1 = layers.Dense(
            self.channels_mlp_dim,
            activation='gelu',
            kernel_regularizer=regularizers.l2(L2_REG),
            name='channel_dense1'
        )
        self.channel_bn1 = layers.BatchNormalization(name='channel_bn1')
        self.channel_dropout1 = layers.Dropout(self.dropout_rate, name='channel_dropout1')
        self.channel_dense2 = layers.Dense(
            self.hidden_dim,
            kernel_regularizer=regularizers.l2(L2_REG),
            name='channel_dense2'
        )
        self.channel_bn2 = layers.BatchNormalization(name='channel_bn2')
        self.channel_dropout2 = layers.Dropout(self.dropout_rate, name='channel_dropout2')

        super().build(input_shape)

    def call(self, x, training=False):
        # Token-mixing
        y = self.norm1(x)
        y = tf.transpose(y, perm=[0, 2, 1])
        y = self.token_dense1(y)
        y = self.token_bn1(y, training=training)
        y = self.token_dropout1(y, training=training)
        y = self.token_dense2(y)
        y = self.token_bn2(y, training=training)
        y = self.token_dropout2(y, training=training)
        y = tf.transpose(y, perm=[0, 2, 1])
        x = x + y  # Skip connection

        # Channel-mixing
        y = self.norm2(x)
        y = self.channel_dense1(y)
        y = self.channel_bn1(y, training=training)
        y = self.channel_dropout1(y, training=training)
        y = self.channel_dense2(y)
        y = self.channel_bn2(y, training=training)
        y = self.channel_dropout2(y, training=training)
        x = x + y  # Skip connection

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'hidden_dim': self.hidden_dim,
            'tokens_mlp_dim': self.tokens_mlp_dim,
            'channels_mlp_dim': self.channels_mlp_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        init_params = {
            'num_patches': config.get('num_patches'),
            'hidden_dim': config.get('hidden_dim'),
            'tokens_mlp_dim': config.get('tokens_mlp_dim'),
            'channels_mlp_dim': config.get('channels_mlp_dim'),
            'dropout_rate': config.get('dropout_rate', 0.0),
        }
        parent_config = {k: v for k, v in config.items()
                         if k not in init_params and k != 'build_config'}
        final_config = {**init_params, **parent_config}
        return cls(**final_config)


# ------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
    1- Multiclass target = (1...n, text1...textn)
    2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    3- Binary target = (1,0)
    :return:
    '''
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

    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        xdepth = len(class_names)
        final_target = tf.one_hot(target, xdepth)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
        final_target = xfinal
        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names


# ------------------------------------------------------------------------------------------------------------------

def process_path(path, label):
    '''
    feature is the path and id of the image
    target is the result
    returns the image and the target as label
    '''
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    return tf.reshape(img, [-1]), label


# ------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''
    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))
    y_target = np.array(y_target.tolist())

    return y_target


# ------------------------------------------------------------------------------------------------------------------

def read_data(num_classes):
    '''
    reads the dataset and process the target
    '''
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))
    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    final_ds = final_ds.prefetch(AUTOTUNE)

    return final_ds


# ------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
    receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# ------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
    Load MLP-Mixer model and make predictions
    '''
    print("\n" + "=" * 60)
    print("Loading MLP-Mixer Model")
    print("=" * 60)

    # Load model - custom layers are automatically registered via decorators
    try:
        final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
        print(f"Successfully loaded: model_{NICKNAME}.keras")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and custom layers are properly defined.")
        raise

    print(f"Model loaded successfully!")
    print(f"Making predictions on {len(xdf_dset)} samples...")

    # Make predictions
    predictions = final_model.predict(test_ds, verbose=1)
    xres = [tf.argmax(f).numpy() for f in predictions]

    # Save model summary
    save_model(final_model)

    # Save results to Excel
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    print(f"\nResults saved to: results_{NICKNAME}.xlsx")

    # Print prediction distribution
    unique, counts = np.unique(xres, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} predictions")

    print("=" * 60)


# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functions of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
        type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
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

    # For multiclass
    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass
    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)


# ------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, xdf_dset

    # Reading the excel file from a directory
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    # Multiclass, verify the classes, change from strings to numbers
    class_names = process_target(1)

    # Filtering the information
    xdf_dset = xdf_data[xdf_data["split"] == SPLIT].copy()

    # Processing the information
    test_ds = read_data(len(class_names))

    # Predict
    predict_func(test_ds)

    # Metrics Function over the result of the test dataset
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    list_of_metrics = ['coh', 'acc']
    list_of_agg = ['avg','sum']
    metrics_func(list_of_metrics, list_of_agg)

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------------------------
