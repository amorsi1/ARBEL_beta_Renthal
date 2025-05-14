import os
import sys

'''
For moving videos from file structure of Pain ML aggregated dataset to that expected by ARBEL's code logic. 

Pain ML format:
├── videos/
    │   ├── exp_group_1/
    │   │   ├── recording
    │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_trans.avi  
    │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_transDLC_resnet50_arcteryx500Nov4shuffle1_350000.h5         
    │   │   │   ├── other output files...
 ARBEL input format:
 ├── ARBEL_project_folder/
    │   ├── Exp_folder/
    │   │   ├── Videos
    │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_trans.avi  
    │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_transDLC_resnet50_arcteryx500Nov4shuffle1_350000.h5             
    │   │   │   ├── all other trans.avi files to analyze...
'''

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import the module
from get_recording_files import get_output_files_by_exp_group
import shutil
import pandas as pd

data_dir = '/mnt/hd0/Pain_ML_data'
save_dest = "/mnt/hd0/Pain_ML_data/ARBEL/Pain_ML/Videos"

def trim_DLC_bodyparts(dlc_file: os.PathLike,
                       target_bodyparts=('lfpaw','rfpaw','lhpaw','rhpaw','snout','neck','sternumtail','tailbase','tailtip')):

    if not dlc_file.endswith('DLC_resnet50_arcteryx500Nov4shuffle1_350000.h5'):
        print("Expecting the following DLC architecture: DLC_resnet50_arcteryx500Nov4shuffle1_350000 \n Ensure that the schema matches")
    h5_df = pd.read_hdf(dlc_file)
    # Create mask of columns and return filtered df
    mask = h5_df.columns.get_level_values(1).isin(target_bodyparts)
    return h5_df.loc[:, mask]

# file = '/mnt/hd0/Pain_ML_data/videos/formalin/Videos/formalin-trimmed_2024-08-06_18-06-36_f304-transDLC_resnet50_arcteryx500Nov4shuffle1_350000.h5'
# trim_DLC_bodyparts(file)
def copy_and_save(data_dir, save_dest):
    output_files = get_output_files_by_exp_group(data_dir)
    for _, files in output_files:
        for file in files:
            trans, dlc = file['trans'], file['dlc']
            # Copy -trans.avi and dlc .h5 file to experiment videos folder
            shutil.copy2(trans, os.path.join(save_dest, os.path.basename(trans)))
            # Process DLC file to cut down from ~50 bodyparts to the 9 bodyparts featured in the paper
            trimmed_df = trim_DLC_bodyparts(dlc)
            trimmed_df.to_hdf(os.path.join(save_dest, os.path.basename(dlc)), key='df_with_missing')

copy_and_save(data_dir, save_dest)
