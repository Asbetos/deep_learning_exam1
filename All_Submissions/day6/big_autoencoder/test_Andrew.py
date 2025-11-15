# ------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import argparse
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generate and save confusion matrix visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{NICKNAME}.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: confusion_matrix_{NICKNAME}.png")
    plt.close()
    
    return cm

# ------------------------------------------------------------------------------------------------------------------

def analyze_confusion_matrix(cm, class_names):
    """
    Detailed confusion matrix analysis
    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Per-class analysis
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        print(f"\nClass: {class_name}")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Negatives:  {tn}")
        
        # Additional metrics
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  Precision: {precision:.4f}")
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"  Recall:    {recall:.4f}")
        if tp + fp + fn > 0:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"  Specificity: {specificity:.4f}")
    
    # Overall misclassification analysis
    print("\n" + "-" * 60)
    print("MISCLASSIFICATION PATTERNS")
    print("-" * 60)
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                total_true = cm[i, :].sum()
                error_rate = cm[i, j] / total_true if total_true > 0 else 0
                print(f"  {true_class} → {pred_class}: {cm[i, j]} "
                      f"({error_rate*100:.1f}% of all {true_class} samples)")

# ------------------------------------------------------------------------------------------------------------------

def per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate per-class performance metrics
    """
    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE METRICS")
    print("=" * 60)
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    # Create DataFrame for better visualization
    metrics_data = []
    for class_name in class_names:
        if class_name in report:
            metrics_data.append({
                'Class': class_name,
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': int(report[class_name]['support'])
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Print formatted table
    print("\n" + df_metrics.to_string(index=False))
    
    # Save to Excel
    df_metrics.to_excel(f'per_class_metrics_{NICKNAME}.xlsx', index=False)
    print(f"\nPer-class metrics saved to: per_class_metrics_{NICKNAME}.xlsx")
    
    # Overall metrics
    print("\n" + "-" * 60)
    print("OVERALL METRICS")
    print("-" * 60)
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"Macro Avg:")
    print(f"  Precision:  {report['macro avg']['precision']:.4f}")
    print(f"  Recall:     {report['macro avg']['recall']:.4f}")
    print(f"  F1-Score:   {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg:")
    print(f"  Precision:  {report['weighted avg']['precision']:.4f}")
    print(f"  Recall:     {report['weighted avg']['recall']:.4f}")
    print(f"  F1-Score:   {report['weighted avg']['f1-score']:.4f}")
    
    return df_metrics

# ------------------------------------------------------------------------------------------------------------------

def prediction_bias_analysis(y_true, y_pred, class_names):
    """
    Analyze prediction bias - which classes are over/under predicted
    """
    print("\n" + "=" * 60)
    print("PREDICTION BIAS ANALYSIS")
    print("=" * 60)
    
    # Count actual vs predicted
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    # Create bias analysis DataFrame
    bias_data = []
    for i, class_name in enumerate(class_names):
        true_count = counts_true[unique_true == i][0] if i in unique_true else 0
        pred_count = counts_pred[unique_pred == i][0] if i in unique_pred else 0
        
        bias = pred_count - true_count
        bias_pct = (bias / true_count * 100) if true_count > 0 else 0
        
        bias_data.append({
            'Class': class_name,
            'True Count': true_count,
            'Predicted Count': pred_count,
            'Bias': bias,
            'Bias %': bias_pct,
            'Status': 'Over-predicted' if bias > 0 else ('Under-predicted' if bias < 0 else 'Balanced')
        })
    
    df_bias = pd.DataFrame(bias_data)
    
    # Print formatted table
    print("\n" + df_bias.to_string(index=False))
    
    # Save to Excel
    df_bias.to_excel(f'prediction_bias_{NICKNAME}.xlsx', index=False)
    print(f"\nPrediction bias analysis saved to: prediction_bias_{NICKNAME}.xlsx")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, df_bias['True Count'], width, label='True Count', alpha=0.8)
    plt.bar(x + width/2, df_bias['Predicted Count'], width, label='Predicted Count', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Prediction Bias: True vs Predicted Counts')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'prediction_bias_{NICKNAME}.png', dpi=300, bbox_inches='tight')
    print(f"Prediction bias plot saved to: prediction_bias_{NICKNAME}.png")
    plt.close()
    
    # Summary
    print("\n" + "-" * 60)
    print("BIAS SUMMARY")
    print("-" * 60)
    over_predicted = df_bias[df_bias['Bias'] > 0]
    under_predicted = df_bias[df_bias['Bias'] < 0]
    
    if len(over_predicted) > 0:
        print("\nOver-predicted classes:")
        for _, row in over_predicted.iterrows():
            print(f"  {row['Class']}: +{row['Bias']} ({row['Bias %']:+.1f}%)")
    
    if len(under_predicted) > 0:
        print("\nUnder-predicted classes:")
        for _, row in under_predicted.iterrows():
            print(f"  {row['Class']}: {row['Bias']} ({row['Bias %']:.1f}%)")
    
    return df_bias

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

    # Prepare true and predicted labels for analysis
    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # ========== NEW COMPREHENSIVE METRICS ==========
    
    # 1. Per-Class Performance Metrics
    per_class_metrics(y_true, y_pred, class_names)
    
    # 2. Prediction Bias Analysis
    prediction_bias_analysis(y_true, y_pred, class_names)
    
    # 3. Confusion Matrix Analysis
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    analyze_confusion_matrix(cm, class_names)

    # Original Metrics Function
    print("\n" + "=" * 60)
    print("AGGREGATE EVALUATION METRICS")
    print("=" * 60)
    list_of_metrics = ['mat','f1_micro','f1_macro','f1_weighted','coh', 'acc']
    list_of_agg = ['avg', 'sum']
    metrics_func(list_of_metrics, list_of_agg)

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print(f"  - results_{NICKNAME}.xlsx")
    print(f"  - summary_{NICKNAME}.txt")
    print(f"  - per_class_metrics_{NICKNAME}.xlsx")
    print(f"  - prediction_bias_{NICKNAME}.xlsx")
    print(f"  - confusion_matrix_{NICKNAME}.png")
    print(f"  - prediction_bias_{NICKNAME}.png")

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------------------------
