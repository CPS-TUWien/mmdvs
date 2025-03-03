import glob
import numpy as np
import tensorflow as tf

def get_data_from_file(filename):
    # Load data from npz file
    # Return the time surface data and the steering action
    time_surface_data = np.load(filename)
    return tf.cast(time_surface_data["data"] * time_surface_data["filtered_owp_mask"].astype(int), tf.float32), tf.cast(time_surface_data["action"][0], tf.float32)

filelist_all = sorted(glob.glob("./data/extracted/*.npz"))
filelist = filelist_all[:16] # Select the first 16 files

# Print the data from the first 16 files
for filename in filelist:
    print(filename)
    print(get_data_from_file(filename))