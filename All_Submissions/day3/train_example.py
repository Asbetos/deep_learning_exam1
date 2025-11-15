#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
# from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
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

PHASE 2 IMPROVEMENTS IMPLEMENTED:
1. Class weight balancing for imbalanced dataset
2. Batch Normalization after each Dense layer
3. Dropout regularization (0.5, 0.4, 0.3 progressive rates)
4. L2 weight regularization on all Dense layers
5. Data augmentation (flips, rotation, brightness, contrast)
6. Label smoothing in loss function
7. Improved architecture with better layer organization

PHASE 3 IMPROVEMENTS:
1. Simplified architecture: 2 hidden layers (256→128)
2. Input size: 128×128 with aspect-ratio resizing + padding
3. BatchNorm + Dropout(0.4) on both hidden layers
4. L2 regularization 0.001
5. Class weights
6. Label smoothing 0.1
7. Data augmentation using native TF ops
8. 100 epochs, EarlyStopping, ReduceLROnPlateau, TensorBoard

# TODO
1. Autoencoder to generate synthetic data for minority classes -- model generates images with no variance
1a. Augmentation for minority classes - Done
2. MLP mixer layer for better feature extraction
3. Branching networks for multi-task learning
4. Input image data to every hidden layer - Model performs very poorly

--FIX IMAGE_SIZE to 400*400 - Done
--REMOVE 1 Instance of duplicate image - Done
-- REMOVE ALL LOW VARIANCE IMAGES - Done
-- CONVERT ALL IMAGES TO RGB
-- USE opencv to get outlines, face recognition?
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
DATA_DIR = os.getcwd() + os.path.sep + 'Balanced_Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LABEL_SMOOTHING = 0.1  # NEW: Reduces overconfidence
L2_REG = 0.01  # NEW: L2 regularization strength

## Image processing
CHANNELS = 3
IMAGE_SIZE = 128

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

# def aspect_resize_pad(img):
#     # Preserve aspect ratio then pad to square
#     orig_shape = tf.cast(tf.shape(img)[:2], tf.float32)
#     scale = IMAGE_SIZE / tf.reduce_max(orig_shape)
#     new_size = tf.cast(tf.round(orig_shape * scale), tf.int32)
#     img = tf.image.resize(img, new_size)
#     delta_h = IMAGE_SIZE - new_size[0]
#     delta_w = IMAGE_SIZE - new_size[1]
#     top = delta_h // 2
#     bottom = delta_h - top
#     left = delta_w // 2
#     right = delta_w - left
#     img = tf.pad(img, [[top,bottom],[left,right],[0,0]], constant_values=0)
#     return img


#------------------------------------------------------------------------------------------------------------------
# def augment_image(img):
#     '''
#     NEW PHASE 2: Apply data augmentation to training images
#     Helps with class imbalance and improves generalization
#     Uses only native TensorFlow operations - NO tensorflow-addons required!
#     '''
#     # Random horizontal flip
#     img = tf.image.random_flip_left_right(img)
#     # Random vertical flip
#     img = tf.image.random_flip_up_down(img)
#     # Random brightness adjustment (±20%)
#     img = tf.image.random_brightness(img, max_delta=0.3)
#     # Random contrast adjustment (80% to 120%)
#     img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
#     # Random saturation adjustment (helps with color variations)
#     img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
#     # Random hue adjustment (slight color shift)
#     img = tf.image.random_hue(img, max_delta=0.1)
#     # Random rotation (using 90-degree increments - native TensorFlow)
#     # 90° rotation increments
#     k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
#     img = tf.image.rot90(img, k)
#     # Random zoom/crop (simulates zoom augmentation)
#     if tf.random.uniform([]) > 0.5:
#         # Random crop and resize back
#         crop_size = tf.random.uniform([], minval=int(IMAGE_SIZE * 0.8),
#                                       maxval=IMAGE_SIZE, dtype=tf.int32)
#         img = tf.image.random_crop(img, size=[crop_size, crop_size, CHANNELS])
#         img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
#
#     # Ensure values stay in [0, 1] range after augmentation
#     img = tf.clip_by_value(img, 0.0, 1.0)
#
#     return img


def process_path(path, label):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''
    img = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    # if augment:
    #     img = augment_image(img)
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

    # removed unnecessary steps
    # end = np.zeros(num_classes)
    # for s1 in y_target:
    #     end = np.vstack([end, s1])
    #
    # y_target = np.array(end[1:])

    y_target = np.array(y_target.tolist())

    return y_target
#------------------------------------------------------------------------------------------------------------------


def read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices(
        (ds_inputs, ds_targets))  # creates a tensor from the image paths and targets

    # Add shuffling for training
    # if shuffle:
    #     list_ds = list_ds.shuffle(buffer_size=1000, seed=42)

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    final_ds = final_ds.prefetch(AUTOTUNE)  # Add prefetching for performance
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
    """
    MLP-Mixer architecture for image classification.
    Pure MLP-based architecture with token-mixing and channel-mixing layers.
    Optimized for imbalanced multi-class image classification.
    """

    # ===== HYPERPARAMETERS =====
    # Adjust these based on your image dimensions
    # Assuming flattened input: INPUTS_r should be HEIGHT * WIDTH * CHANNELS
    # For example, if images are 64x64x3, then INPUTS_r = 12288

    # Calculate image dimensions from flattened input
    # You'll need to set these based on your actual image size

    PATCH_SIZE = 8  # Size of each image patch
    NUM_PATCHES = (IMAGE_SIZE// PATCH_SIZE) * (IMAGE_SIZE // PATCH_SIZE)
    PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS

    HIDDEN_DIM = 256  # Embedding dimension
    NUM_MIXER_BLOCKS = 8  # Number of mixer layers (depth)
    TOKENS_MLP_DIM = 256  # Token-mixing MLP hidden dimension
    CHANNELS_MLP_DIM = 512  # Channel-mixing MLP hidden dimension
    DROPOUT_RATE = 0.3

    # ===== CUSTOM LAYERS =====

    class PatchEmbedding(layers.Layer):
        """Extract and embed image patches"""

        def __init__(self, num_patches, patch_dim, hidden_dim):
            super().__init__()
            self.num_patches = num_patches
            self.patch_dim = patch_dim
            self.projection = layers.Dense(hidden_dim)

        def call(self, images):
            # Reshape flattened input to image format
            batch_size = tf.shape(images)[0]
            images = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

            # Extract patches
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )

            # Reshape patches: [batch, num_patches, patch_dim]
            patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])

            # Linear projection to hidden dimension
            embedded = self.projection(patches)
            return embedded

    class MLPBlock(layers.Layer):
        """MLP block with GELU activation and dropout"""

        def __init__(self, mlp_dim, dropout_rate):
            super().__init__()
            self.dense1 = layers.Dense(mlp_dim, activation='gelu')
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dense2 = layers.Dense(mlp_dim)
            self.dropout2 = layers.Dropout(dropout_rate)

        def call(self, x, training=False):
            x = self.dense1(x)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            return x

    class MixerBlock(layers.Layer):
        """MLP-Mixer block with token-mixing and channel-mixing"""

        def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout_rate):
            super().__init__()
            self.num_patches = num_patches
            self.hidden_dim = hidden_dim

            # Layer normalizations
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)

            # Token-mixing MLP
            self.token_mixing = keras.Sequential([
                layers.Dense(tokens_mlp_dim, activation='gelu'),
                layers.Dropout(dropout_rate),
                layers.Dense(num_patches),
                layers.Dropout(dropout_rate)
            ])

            # Channel-mixing MLP
            self.channel_mixing = keras.Sequential([
                layers.Dense(channels_mlp_dim, activation='gelu'),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_dim),
                layers.Dropout(dropout_rate)
            ])

        def call(self, x, training=False):
            # Token-mixing (operate across patches)
            y = self.norm1(x)
            y = tf.transpose(y, perm=[0, 2, 1])  # [batch, hidden_dim, num_patches]
            y = self.token_mixing(y, training=training)
            y = tf.transpose(y, perm=[0, 2, 1])  # [batch, num_patches, hidden_dim]
            x = x + y  # Skip connection

            # Channel-mixing (operate across channels/features)
            y = self.norm2(x)
            y = self.channel_mixing(y, training=training)
            x = x + y  # Skip connection

            return x

    # ===== BUILD MODEL =====

    # Input layer (flattened images)
    inputs = keras.Input(shape=(INPUTS_r,))

    # Patch embedding
    x = PatchEmbedding(NUM_PATCHES, PATCH_DIM, HIDDEN_DIM)(inputs)

    # Stack of Mixer blocks
    for _ in range(NUM_MIXER_BLOCKS):
        x = MixerBlock(
            num_patches=NUM_PATCHES,
            hidden_dim=HIDDEN_DIM,
            tokens_mlp_dim=TOKENS_MLP_DIM,
            channels_mlp_dim=CHANNELS_MLP_DIM,
            dropout_rate=DROPOUT_RATE
        )(x)

    # Global average pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head with dropout for regularization
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(OUTPUTS_a, activation='softmax', name='classifier')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='mlp_mixer')

    # Compile model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    save_model(model)
    return model


#------------------------------------------------------------------------------------------------------------------

# def compute_class_weights(labels):
#     '''
#     NEW PHASE 2: Compute class weights to handle imbalanced dataset
#     '''
#     # Check if labels are already class indices (1D) or one-hot encoded (2D)
#     if labels.ndim == 1:
#         y = labels
#         classes = np.unique(y)
#     else:
#         # Get unique classes and compute balanced weights
#         y = np.argmax(labels, axis=1)
#         classes = np.arange(labels.shape[1])
#
#     weights = compute_class_weight(
#         class_weight='balanced',
#         classes=classes,
#         y=y
#     )
#     return dict(enumerate(weights))


def train_func(train_ds, val_ds,class_weight_dict=None):
    '''
    train the model with class weights
    '''
    # Early stopping: stops training when validation loss stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience for Phase 2
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint: save best model based on validation loss
    check_point = tf.keras.callbacks.ModelCheckpoint(
        'model_{}.keras'.format(NICKNAME),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Increased patience
        min_lr=1e-7,
        verbose=1
    )

    # NEW PHASE 2: TensorBoard callback for visualization
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs_{}'.format(NICKNAME),
        histogram_freq=1
    )

    final_model = model_definition()

    # NEW PHASE 2: Train with class weights
    history = final_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epoch,
        callbacks=[early_stop, check_point, reduce_lr, tensorboard],
        class_weight=class_weight_dict,  # NEW: Apply class weights
        verbose=1
    )

    return history
#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.keras'.format(NICKNAME))
    xres = [tf.argmax(f).numpy() for f in final_model.predict(test_ds)]
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

    # for file in os.listdir(PATH+os.path.sep + "excel"):
    #     if file[-5:] == '.xlsx':
    #         FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file
    FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + "training" + os.path.sep + "train_test_balanced.xlsx"

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)
    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    # Create validation split from training data =====
    train_data = xdf_data[xdf_data["split"] == 'train'].copy()

    # Shuffle and split training data for validation (80/20 split)
    train_data_shuffled = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    # split_idx = int(0.8 * len(train_data_shuffled))

    # x = lambda x: tf.argmax(x == class_names).numpy()
    # train_labels = np.array(train_data_shuffled['target'].apply(x))

    # Compute class weights for imbalanced dataset
    # class_weight_dict = compute_class_weights(train_labels[:split_idx])

    # Create training subset
    xdf_dset = train_data_shuffled.copy()
    train_ds = read_data(OUTPUTS_a)

    # Create validation subset
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    val_ds = read_data(OUTPUTS_a)

    # Train model with class weights
    history = train_func(train_ds, val_ds)

    # Preprocessing Test dataset
    # xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()
    # test_ds = read_data(OUTPUTS_a)
    predict_func(val_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['coh', 'acc']
    list_of_agg = ['avg','sum']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
#------------------------------------------------------------------------------------------------------------------

