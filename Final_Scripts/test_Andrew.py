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
Testing script for Supervised Autoencoder-MLP model
Loads trained model with custom Swish activation layer
'''

# ------------------------------------------------------------------------------------------------------------------

CHANNELS = 3
IMAGE_SIZE = 224  # Must match training script
BATCH_SIZE = 32   # Match training script

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
# CUSTOM SWISH ACTIVATION - MUST BE DEFINED BEFORE LOADING MODEL
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
        model.summary(print_fn=lambda x: fh.write(x + '\\n'))

# ------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
    Load Supervised Autoencoder-MLP model and make predictions
    NOTE: Model outputs TWO values [classification, reconstruction], we only need classification
    '''
    print("\\n" + "=" * 60)
    print("Loading Supervised Autoencoder-MLP Model")
    print("=" * 60)

    # Load model with custom Swish activation
    custom_objects = {'Swish': Swish}

    try:
        final_model = tf.keras.models.load_model(
            'model_{}.keras'.format(NICKNAME),
            custom_objects=custom_objects
        )
        print(f"Successfully loaded: model_{NICKNAME}.keras")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and Swish layer is properly defined.")
        raise

    print(f"Model loaded successfully!")
    print(f"Model architecture: Supervised Autoencoder-MLP")
    print(f"Output layers: {len(final_model.outputs)} (classification + reconstruction)")
    print(f"\\nMaking predictions on {len(xdf_dset)} samples...")

    # Make predictions - model returns [classification_output, reconstruction_output]
    # We only need classification predictions (index 0)
    all_predictions = []

    for batch_images, _ in test_ds:
        batch_predictions = final_model.predict(batch_images, verbose=0)
        # batch_predictions is a list: [classification, reconstruction]
        classification_preds = batch_predictions[0]  # Extract only classification
        all_predictions.append(classification_preds)

    # Combine all batches
    all_predictions = np.vstack(all_predictions)

    # Get class labels from predictions
    xres = [tf.argmax(f).numpy() for f in all_predictions]

    # Save model summary
    save_model(final_model)

    # Save results to Excel
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    print(f"\\nResults saved to: results_{NICKNAME}.xlsx")

    # Print prediction distribution
    unique, counts = np.unique(xres, return_counts=True)
    print("\\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} predictions ({100*count/len(xres):.1f}%)")

    print("=" * 60)

# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functions of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_weighted, coh, acc, mat
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
            # f1 score average = weighted
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

    print("\\n" + "=" * 60)
    print("SUPERVISED AUTOENCODER-MLP TESTING SCRIPT")
    print("=" * 60)

    # Reading the excel file from a directory
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    # Multiclass, verify the classes, change from strings to numbers
    class_names = process_target(1)

    print(f"\\nDataset Information:")
    print(f"  Total images: {len(xdf_data)}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Class names: {class_names}")
    print(f"  Split: {SPLIT}")

    # Filtering the information
    xdf_dset = xdf_data[xdf_data["split"] == SPLIT].copy()
    print(f"  {SPLIT} set size: {len(xdf_dset)}")

    # Processing the information
    test_ds = read_data(len(class_names))

    # Predict
    predict_func(test_ds)

    # Metrics Function over the result of the test dataset
    print("\\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    list_of_metrics = ['coh', 'acc']
    list_of_agg = ['avg', 'sum']
    metrics_func(list_of_metrics, list_of_agg)

    print("\\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------------------------
