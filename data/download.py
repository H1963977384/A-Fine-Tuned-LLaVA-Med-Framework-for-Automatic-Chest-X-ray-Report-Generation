import kagglehub
import os
import shutil

# Define target directory for dataset storage in current working directory
target_path = os.path.join(os.getcwd(), 'image')

# Download the Indiana University Chest X-rays dataset using Kaggle Hub API
# Dataset contains chest radiograph images and associated medical data
dataset_path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")

# Move the downloaded dataset from temporary storage to target directory
# This organizes the dataset into a predictable location for subsequent processing
shutil.move(dataset_path, target_path)
