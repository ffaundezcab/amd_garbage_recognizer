import os
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import itertools
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from keras.callbacks import EarlyStopping
from keras import mixed_precision
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score

from collections import Counter
import tensorflow as tf

image_size = (96, 96)
num_classes = 10

class_weights = {0: np.float64(1.6402457757296467),
 1: np.float64(1.6478395061728395),
 2: np.float64(0.8702526487367563),
 3: np.float64(0.6728418399495905),
 4: np.float64(0.6682102628285357),
 5: np.float64(1.3431446540880503),
 6: np.float64(0.9672101449275362),
 7: np.float64(0.7811265544989027),
 8: np.float64(0.8176110260336906),
 9: np.float64(2.7032911392405063)}

mixed_precision.set_global_policy('mixed_float16')
logs_path = "C:/Users/faarc/OneDrive/Escritorio/Uni/5th Trimester/Algorithms for massiva Data/amd_garbage_recognizer/notebooks/logs"

def count_label_distribution(dset):
    '''Takes a TensorFlow dataset and counts how many observations for each label are present. 
    
    Args:
        - dset: dataset with tensors.
        
    Returns:
        - total_observations (int): number of elements existing in the dataset.
        - labels_count (dict): dictionary with the label index and total count.
        
    '''
    label_array = [x[1] for x in dset]
    
    labels_in_dset = []
    for x in label_array:
        labels_in_dset.extend(np.argmax(x, axis = 1))
    total_observations = len(labels_in_dset)
    labels_count = Counter(labels_in_dset)
    
    return total_observations, labels_count

def check_jpeg(path):
    """
    cases where there are corrputed files
    todo: add docstring
    """
    with open(path, "rb") as f:
        first_bytes = f.read(3)
        return first_bytes == b'\xff\xd8\xff'
    
    

def root_logdir(architecture_name):
    '''Defines the logs directory for a specific architecture.
    
    Args:
        - architecture_name (str): name of the specific model that is being trained for the respective architecture.
        
    Returns:
        - Path for the logs of execution to be saved.
    
    '''
    return os.path.join(os.curdir, "logs/"+architecture_name)

def get_run_logdir(architecture_name, it):
    '''Creates a run id by adding a timestamp to the execution path of each log to better trace each training session.
    
    Args:
        - architecture_name (str): name of the specific model that is being trained for the respective architecture.
        - it (int): number of the architecture that is being trained.
        
    Returns:
        - run_id (str): combination of architecture number and timestamp.
        - new_path (str): updated log path with run_id.
    '''
    run_id = str(it) +"_"+ time.strftime("run_%Y_%m_%d-%H_%M_%S")
    new_path = os.path.join(root_logdir(architecture_name), run_id)
    
    return run_id, new_path

def table_from_history(history_table, run_id):
    '''(DEPRECATED) Converts history table of a training session to a DataFrame.
    *Note: Only used in non-paralellized GridSearch, deprecated.
    
    Args:
        - history_table: resulting history of a training session
        - run_id (str): combination of architecture number and timestamp.
        
    Returns:
        - dft (pd.DataFrame): dataframe containing all the information of the history report.
    
    '''
    dft = pd.DataFrame(history_table)
    dft['run'] = run_id
    dft = dft.reset_index(names = "epoch")
    return dft 


def pass_batchs_to_arrays(dset, pass_dummy=False):
    X_list = []
    y_list = []

    for x_batch, y_batch in dset.as_numpy_iterator():
        X_list.append(x_batch)
        y_list.append(y_batch)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if pass_dummy:
        y = np.argmax(y, axis=1)

    return X, y

def load_full_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1024)
    
    images = []
    targets = []
    
    for batch_images, batch_labels in ds:
        images.append(batch_images)
        targets.append(batch_labels)
        
    X = tf.concat(images, axis=0)
    y = tf.concat(targets, axis=0)
    
    return X, y

# pass thru tf to convert paths to images
def preprocess_image(file_path, label):
    ''' Reads a file and then converts the input bytes using the appropiate scale into a Tensor.
    
    Args:
        - file_path (str): directory address where the file is.
        - label (str): real label of the original image.
        - image_size (tuple): global tuple indicating the original size of the image.
        
    Returns:
        - img: correspondent tensor of the converted image.
        - tf.one_hot(...): one-hot tensor of the original label
    
    '''
    img = tf.io.read_file(file_path)
    # by default we read it on RGB
    # img = tf.image.decode_image(img, channels=3)
    img = tf.io.decode_jpeg(img, channels=3)        
    img = tf.image.resize(img, image_size)
    return img, tf.one_hot(label, depth=num_classes)

def save_params(run_logdir, name, best_params):
    '''Saves a dictionary in pickle format.
    
    Args:
        - run_logdir (str): directiory where the dictionary will be saved.
        - name (str): name of the file.
        - best_params (dict): dictionary to be saved.
    
    '''
    with open(run_logdir + '/' + name, 'wb') as f:
        pickle.dump(best_params, f)
        
def load_params(run_logdir):
    '''Loads a dictionary previously saved.
    
    Args:
        - run_logdir (str): path of the pickle file to be loaded.
        
    Returns:
        - loaded_params (dict): loaded dictionary.
    
    '''
    with open(run_logdir, 'rb') as f:
        loaded_params = pickle.load(f)
    
    return loaded_params 


def compute_classification_metrics(y_true, y_pred, class_names):
    '''Computes confusion matrix based on prediction and true labels.
    
    Args:
        - y_true (np.array): true labels of a dataset.
        - y_pred (np.array): predicted labels.
        - class_names (list): index-ordered class names.
        
    Returns:
        - cm: confusion matrix.
        - Also prints a classification report.
    
    '''
    cm = confusion_matrix(y_true, y_pred)
    str_summary = classification_report(y_true, y_pred, target_names=class_names)
    print(str_summary)
    
    # todo: manually compute other metrics
    return cm


def plot_conf_matrix(cm, class_names):
    ''' Plots a confusion matrix coming from the compute_classification_metrics function
    
    Args:
        - cm: confusion matrix.
    Returns:
        - Plot of the confusion matrix.
        
    '''
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.show()