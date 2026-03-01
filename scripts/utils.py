import os
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import itertools
import pickle
import time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from keras.callbacks import EarlyStopping
from keras import mixed_precision

from collections import Counter
import tensorflow as tf

image_size = (32,32)
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
    return os.path.join(os.curdir, "logs\\"+architecture_name)

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

def evaluate_combination(i_p, params, X_train, y_train, skf, build_model, batch_size=128):
    """
    Args:
        i_p (int): index of the hyperparameter combination
        params (dict): current hyperparameter values
        X_train_paths (np.array): array of image file paths
        y_train_array (np.array): array of corresponding labels
        skf (StratifiedKFold): cross-validation splitter
        build_model (function): function returning a compiled Keras model
        batch_size (int): batch size for training
        
    Returns:
        dict: {'index', 'params', 'val_loss', 'val_accuracy'}
    """
    
    history_tables = []
    
    print(f"combination {i_p}")
    print(f"Params: {params}")
    start_comb = time.time()

    for i, (train_i, val_i) in enumerate(skf.split(X_train, y_train)):
        fold = f"Fold{i+1}"
        
        # y_train_fold = tf.one_hot(y_train[train_i], depth=num_classes)
        train_ds = tf.data.Dataset.from_tensor_slices(
                (X_train[train_i], y_train[train_i])
            ).shuffle(len(train_i)).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        # y_val_fold = tf.one_hot(y_train[val_i], depth=num_classes)
        val_ds = tf.data.Dataset.from_tensor_slices(
            (X_train[val_i], y_train[val_i])
        ).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        
        tf.keras.backend.clear_session()
        model = build_model(**params)

        run_id, run_logdir = get_run_logdir("first_architecture", f"ip_{i_p}_{fold}")

        # train
        # print("training")
        start_fold = time.time()
        early_stopping = EarlyStopping(patience = 3, restore_best_weights=True)
        history = model.fit(train_ds, 
                            validation_data=val_ds, 
                            epochs=15, verbose=0, 
                            class_weight=class_weights, 
                            callbacks = [early_stopping])
        
        fold_time = time.time() - start_fold
        # print(f"[{fold}] time: {fold_time:.1f}")

        history_tables.append(table_from_history(history.history, run_id))

    df_history = pd.concat(history_tables)

    best_train_val = df_history.groupby('run').apply(
        lambda x: x.loc[x["val_loss"].idxmin()]
    ).reset_index(drop=True)

    val_loss = best_train_val['val_loss'].mean()
    val_accuracy = best_train_val['val_accuracy'].mean()

    return {
        'index': i_p,
        'params': params,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }

def CNN_GridSearchCV(X_train_paths, y_train_array, param_grid, build_model, random_state,
                     n_splits=5, shuffle=True, n_jobs=1, batch_size=32):
    """
    Runs GridSearchCV for a CNN
    
    Args:
        X_train_paths (np.array): array of image file paths
        y_train_array (np.array): array of labels
        param_grid (dict): dictionary of hyperparameter options
        build_model (function): returns compiled model
        random_state (int)
        n_splits (int): K in K-Fold
        shuffle (bool)
        n_jobs (int): parallel jobs (-1 uses all cores)
        batch_size (int)
    
    Returns:
        best_params, best_accuracy, mean_scores
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]
    total_start = time.time()
    results = []
    for i, params in enumerate(param_combinations):
        result = evaluate_combination(i, params, X_train_paths, y_train_array, skf, build_model, batch_size)
        results.append(result)
        
    total_time = time.time() - total_start
    # print(f"tot_time: {total_time:.1f}")
    best_result = max(results, key=lambda r: r['val_accuracy'])
    best_params = best_result['params']
    best_accuracy = best_result['val_accuracy']
    mean_scores = [(r['val_loss'], r['val_accuracy']) for r in results]

    return best_params, best_accuracy, mean_scores


def save_params(run_logdir, name, best_params):
    '''Saves a dictionary in pickle format.
    
    Args:
        - run_logdir (str): directiory where the dictionary will be saved.
        - name (str): name of the file.
        - best_params (dict): dictionary to be saved.
    
    '''
    with open(run_logdir + '\\' + name, 'wb') as f:
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
