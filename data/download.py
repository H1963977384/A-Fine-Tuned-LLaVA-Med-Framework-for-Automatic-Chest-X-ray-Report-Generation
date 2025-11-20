import kagglehub
import os
import shutil

target_path = os.path.join(os.getcwd(), 'image')
dataset_path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
shutil.move(dataset_path, target_path)