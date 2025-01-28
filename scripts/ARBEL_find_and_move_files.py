import os
import sys

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import the module
from get_recording_files import get_output_files_by_exp_group
import shutil


data_dir = '/mnt/hd0/Pain_ML_data'
output_files = get_output_files_by_exp_group(data_dir)

for Experiment, files in output_files:
    Experiment_folder = os.path.join(data_dir, 'videos', Experiment)
    if os.path.exists(Experiment_folder):
        Experiment_videos = os.path.join(Experiment_folder, 'Videos') #ARBEL is expecting this
        os.makedirs(Experiment_videos, exist_ok=True)
        for file in files:
            trans, dlc = file['trans'], file['dlc']
            # Copy -trans.avi and dlc .h5 file to experiment videos folder
            shutil.copy2(trans, os.path.join(Experiment_videos, os.path.basename(trans)))
            shutil.copy2(dlc, os.path.join(Experiment_videos, os.path.basename(dlc)))


