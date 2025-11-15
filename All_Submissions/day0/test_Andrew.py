# ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------------------------------------------------------------------

'''

'''
# ------------------------------------------------------------------------------------------------------------------

CHANNELS = 3
IMAGE_SIZE = 200
BATCH_SIZE = 32

##  0, 1, 3
## Review documentation on tersorflow https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

NICKNAME = 'andrew'


# ------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

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

def process_path(feature, target):
    '''
             feature is the path and id of the image
             target is the result
             returns the image and the target as label
    '''

    label = target

    # Processing feature

    # load the raw data from the file as a string
    file_path = feature

    ## Read the image from disk
    img = tf.io.read_file(file_path)

    ## Decode the image
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    ## Reshape the image to get the right dimensions for the initial input in the model
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    # Normalize pixels to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0

    ## Flatten the image to match the input shape expected by the model
    img = tf.reshape(img, [-1])

    return img, label


# ------------------------------------------------------------------------------------------------------------------


def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    # removed unnecessary steps
    # end = np.zeros(num_classes)
    # for s1 in y_target:
    #     end = np.vstack([end, s1])
    #
    # y_target = np.array(end[1:])

    y_target = np.array(y_target.tolist())
    return y_target


# ------------------------------------------------------------------------------------------------------------------


def read_data(num_classes):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    ## Make the channel as a list to make it variable
    ## Create the data set and call the function map to create the dataloader using
    ## tf.data.Dataset
    ## dataset.map
    ## map creates an iterable
    ## More information on https://www.tensorflow.org/tutorials/images/classification

    # Create TensorFlow dataset from inputs and targets
    dataset = tf.data.Dataset.from_tensor_slices((ds_inputs, ds_targets))

    # Map the process_path function to load and process images
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch for performance
    final_ds = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

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
    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
    xres = [tf.argmax(f).numpy() for f in final_model.predict(test_ds)]

    save_model(final_model)

    ## write the rsults to a results_<nickname> xlsx
    ## the function will return a probability if using softmax
    ## The answers should be the the label not the probability
    ## for more information on activation functions and the results https://www.tensorflow.org/api_docs/python/tf/keras/activations

    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)


# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
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
    # Ask for arguments for each metric


# ------------------------------------------------------------------------------------------------------------------

### Main

## Reading the excel file from a directory
for file in os.listdir(PATH + os.path.sep + "excel"):
    if file[-5:] == '.xlsx':
        FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

# Reading and filtering Excel file

xdf_data = pd.read_excel(FILE_NAME)

## Multiclass , verify the classes , change from strings to numbers

class_names = process_target(1)

# Filtering the information
xdf_dset = xdf_data[xdf_data["split"] == SPLIT].copy()

# processing the information
test_ds = read_data(len(class_names))

# predict
predict_func(test_ds)
## Metrics Function over the result of the test dataset

list_of_metrics = ['f1_macro', 'f1_weighted', 'coh', 'acc']
list_of_agg = ['avg']
metrics_func(list_of_metrics, list_of_agg)
