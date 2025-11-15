#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
from tensorflow.keras.utils import to_categorical

#------------------------------------------------------------------------------------------------------------------

'''
PHASE 1 IMPROVEMENTS IMPLEMENTED:
1. Increased epochs from 1 to 100 with early stopping
2. Increased batch size from 2 to 32
3. Added pixel normalization (divide by 255.0)
4. Implemented He initialization for all layers
5. Switched to Adam optimizer with default parameters
6. Added ReduceLROnPlateau callback
7. Added validation split for monitoring
'''
#------------------------------------------------------------------------------------------------------------------

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("../..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 500
BATCH_SIZE = 32
L2_REG = 0.0001
## Image processing
CHANNELS = 3
IMAGE_SIZE = 128
INITIAL_LEARNING_RATE = 1e-5
NICKNAME = 'Andrew'
#------------------------------------------------------------------------------------------------------------------

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

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
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
        # target_class is already done
        pass

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

def process_path(path, label):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''

    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    # Histogram equalization (enhance contrast)
    # img = tf.image.adjust_contrast(img, 1.2)
    # # Normalize per channel (instead of global)
    # img = tf.image.per_image_standardization(img)
    img = aspect_resize_pad(img)
    img = tf.cast(img, tf.float32) / 255.0
    # img = heavy_augment(img, is_training=is_training)
    return tf.reshape(img, [-1]), label
#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target
#------------------------------------------------------------------------------------------------------------------


def read_data(num_classes, shuffle=False):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    # Add shuffling for training
    if shuffle:
        list_ds = list_ds.shuffle(buffer_size=1000, seed=42)

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    final_ds = final_ds.prefetch(AUTOTUNE)  #Add prefetching for performance
    return final_ds
#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
#------------------------------------------------------------------------------------------------------------------

def model_definition():
    # =====He Initialization + Adam Optimizer + BatchNorm + Dropout + L2 Regularization =====
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # First layer with input shape
    model.add(tf.keras.layers.Dense(1024,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG),
                                    input_shape=(INPUTS_r,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    # Hidden layers
    model.add(tf.keras.layers.Dense(896,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(768,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.15))

    model.add(tf.keras.layers.Dense(640,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(512,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(384,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(256,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.35))

    model.add(tf.keras.layers.Dense(128,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.L2(L2_REG)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.35))

    # Output layer (no batch norm, dropout, or regularization on output layer)
    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax'))

    steps_per_epoch = len(xdf_data[xdf_data["split"] == 'train']) // BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        first_decay_steps=steps_per_epoch * 20,
        t_mul=1.5,
        m_mul=0.9,
        alpha=1e-6
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.F1Score(average='macro')])

    save_model(model)
    return model


#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, val_ds):
    '''
        train the model
    '''

    # ===== PHASE 1 IMPROVEMENTS: Added callbacks for better training =====


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True, verbose=1, mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'model_{NICKNAME}.keras', monitor='val_loss',
            mode='min', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=f'./logs_{NICKNAME}', histogram_freq=1),

        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,  # Reduce LR by half
        #     patience=5,  # Wait 5 epochs before reducing
        #     min_lr=1e-7,
        #     verbose=1
        # )
    ]

    final_model = model_definition()

    # IMPROVED: Added validation_data and all callbacks
    history = final_model.fit(
        train_ds,
        validation_data=val_ds,  # IMPROVED: Added validation data
        epochs=n_epoch,
        callbacks=callbacks,  # IMPROVED: All callbacks
        verbose=1
    )

    return history
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#------------------------------------------------------------------------------------------------------------------

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
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric
#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    # Create validation split from training data =====
    train_data = xdf_data[xdf_data["split"] == 'train'].copy()

    # Shuffle and split training data for validation (80/20 split)
    train_data_shuffled = train_data.sample(frac=1, random_state=752).reset_index(drop=True)
    split_idx = int(0.8 * len(train_data_shuffled))

    # Create training subset
    xdf_dset = train_data_shuffled.iloc[:split_idx].copy()
    train_ds = read_data(OUTPUTS_a, shuffle=True)

    # Create validation subset
    xdf_dset = train_data_shuffled.iloc[split_idx:].copy()
    val_ds = read_data(OUTPUTS_a, shuffle=False)

    history = train_func(train_ds, val_ds)

    # Preprocessing Test dataset
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    test_ds = read_data(OUTPUTS_a)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'f1_weighted', 'coh', 'acc']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------

