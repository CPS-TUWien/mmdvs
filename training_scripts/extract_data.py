import os 
import tarfile

tar_path = './data/dvs_only_256p_100hz.tar.bz2'
extracted_dir = './data/extracted'

if os.path.isfile(tar_path):

    print(f"Extracting from {tar_path}")

    if os.path.exists(extracted_dir):
        print(f"Directory {extracted_dir} already exists, skipping")
    else:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(extracted_dir) 
else:
    raise ValueError(
        f"Could not find *.tar.bz2 file '{tar_path}'"
    )