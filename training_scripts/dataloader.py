import tensorflow as tf
import numpy as np
import os

def generator_function(file_list, sequence_length, with_filenames=False):
    for i in range(0, len(file_list) - sequence_length + 1):
        images = []
        labels = []
        if with_filenames:
            filenames = []
        for j in range(sequence_length):
            with np.load(file_list[i + j]) as data:
                images.append(data['data']*data["filtered_owp_mask"])
                labels.append(data['action'][0])
                if with_filenames:
                    filenames.append(file_list[i + j])
        if with_filenames:
            yield (tf.stack(images), tf.stack(labels), tf.stack(filenames))
        else:
            yield (tf.stack(images), tf.stack(labels))

def create_dataset(file_list, sequence_length, with_filenames=False):
    if with_filenames:
        return tf.data.Dataset.from_generator(
            lambda: generator_function(file_list, sequence_length, with_filenames = with_filenames),
            output_types=(tf.float32, tf.float32, tf.string),
            output_shapes=((sequence_length, None, None, None), (sequence_length,), (sequence_length,))
        )
    else:
        return tf.data.Dataset.from_generator(
            lambda: generator_function(file_list, sequence_length),
            output_types=(tf.float32, tf.float32),
            output_shapes=((sequence_length, None, None, None), (sequence_length,))
        )

def create_datasets(data_dir, sequence_length=16, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, with_filenames=False):
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
    
    n_files = len(all_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    train_dataset = create_dataset(train_files, sequence_length)
    val_dataset = create_dataset(val_files, sequence_length)
    test_dataset = create_dataset(test_files, sequence_length, with_filenames=with_filenames)
    
    return train_dataset, val_dataset, test_dataset

def configure_for_performance(dataset, batch_size=32):
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# # Usage
# data_dir = './data/extracted'
# sequence_length = 16
# with tf.device('CPU:0'):
#     train_dataset, val_dataset, test_dataset = create_datasets(data_dir, sequence_length)

# train_dataset = configure_for_performance(train_dataset, batch_size=2)
# val_dataset = configure_for_performance(val_dataset, batch_size=2)
# test_dataset = configure_for_performance(test_dataset, batch_size=2)

# # Example usage
# for image_sequence, label_sequence in val_dataset.take(1):
#     print(f"Image sequence shape: {image_sequence.shape}")
    
#     print(f"Label sequence shape: {label_sequence.shape}")
#     # print(f"Label sequence: {label_sequence}")