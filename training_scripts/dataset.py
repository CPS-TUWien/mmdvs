import tarfile
import tensorflow as tf
import os
import numpy as np
import pathlib
import glob

# Dataset to use
dataset_name = "dvs_only_256p_100hz"

# Get list of files
filelist_all = sorted(glob.glob("./data/extracted/*.npz"))
filelist = filelist_all[:int(len(filelist_all)*0.2)]

# Splitting the dataset into train, validation and test
filelist_train = filelist[:int(len(filelist)*0.6)]
filelist_valid = filelist[int(len(filelist)*0.6):int(len(filelist)*0.8)]
filelist_test = filelist[int(len(filelist)*0.8):]

# Use tf.data.Dataset 
ds_train = tf.data.Dataset.from_tensor_slices(filelist_train).repeat() #use repeat() if you specify the number of steps per epoch
ds_valid = tf.data.Dataset.from_tensor_slices(filelist_valid).repeat() 
ds_test = tf.data.Dataset.from_tensor_slices(filelist_test)

def get_data_from_file(filename):
    # Load data from npz file
    # Return the time surface data and the steering action
    time_surface_data = np.load(filename.decode())
    return tf.cast(time_surface_data["data"] * time_surface_data["filtered_owp_mask"].astype(int), tf.float32), tf.cast(time_surface_data["action"][0], tf.float32)
    
@tf.function
def get_data_wrapper(filename):
    # Assuming here that both your data and label is float type.
    features, labels = tf.numpy_function(
        get_data_from_file, [filename], (tf.float32, tf.float32)) 
    return features, labels

@tf.function
def get_data_wrapper_with_filenames(filename):
    # Including the filename in the dataset for testing purposes
    features, labels = tf.numpy_function(
        get_data_from_file, [filename], (tf.float32, tf.float32)) 

    return features, labels, filename

class LineFollowingDataset:
    def __init__(self, directory) -> None:
        directory = pathlib.Path(directory).expanduser()

        if os.path.isfile(os.path.join(directory, dataset_name + ".tar.bz2")):
            print(f"Found data in '{directory}'")

            for filename in sorted(directory.glob("*.tar.bz2")):
                print(f"Extracting from {filename}")

                if os.path.exists(directory / 'extracted'):
                    print(f"Directory {directory / 'extracted'} already exists, skipping")
                else:
                    with tarfile.open(filename, "r:bz2") as tar:
                        tar.extractall(directory / 'extracted') 
        else:
            raise ValueError(
                f"Could not find *.tar.bz2 file in '{directory}'"
            )

    def load_train_dataset(self, seq_len = 16, batch_size = 4):
        # Load the training dataset
        ds = ds_train.map(get_data_wrapper)
        # Creating a dataset with sequence length
        ds = ds.batch(seq_len, drop_remainder= True)
        # Creating dataset with batch_size
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def load_valid_dataset(self, seq_len = 16, batch_size = 4):
        ds = ds_valid.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        # Creating a dataset with sequence length
        ds = ds.batch(seq_len, drop_remainder= True)
        # creating dataset with batch_size
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        return ds
    
    def load_test_dataset(self, seq_len = 16, batch_size = 4, include_filenames = False):
        # Load the test dataset, including the filenames for testing purposes
        if include_filenames:
            ds = ds_test.map(get_data_wrapper_with_filenames, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds_test.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        # Creating a dataset with sequence length
        ds = ds.batch(seq_len, drop_remainder= True)
        # # creating dataset with batch_size
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        return ds

# if __name__ == "__main__":
#     data = LineFollowingDataset('data')