"""
ARBEL Automatic Behavior Scoring Script

This script is designed for automatic behavior scoring using the ARBEL framework.
It integrates pose estimation, pixel brightness feature extraction, and behavior
classification through machine learning models.

Components Used:
- `MatlabLikeFunctions`: Provides utility functions similar to MATLAB for data processing.
- `AniML_VideoLabel`: Handles video labeling and visualization of behaviors.
- `ARBEL_Predict`: Runs the classification models to predict behaviors from extracted features.
- `AniML_utils_PoseFeatureExtraction`: Extracts pose-based features for classification.
- `AniML_utils_PixBrightnessFeatureExtraction`: Extracts pixel brightness features.

User-Defined Parameters:
- `Project`: Path to the main folder containing experimental data.
- `Experiments`: List of experimental folders to process. An experiment folder should contain all videos.
- `Behavior_classifiers`: List of pre-trained classifier files for behavior prediction.
- Feature extraction parameters such as `pix_threshold`, `bp_pixbrt_list`, and `square_size`.
- Video creation options such as resolution, FPS, and input/output formats.

Workflow:
1. Loop through experimental folders and identify videos for analysis.
2. Extract pose and pixel brightness features from video data.
3. Apply classifiers to predict behaviors and save results.
4. Compile and save summary statistics for each experiment.
5. (Optional) generate labeled videos for the detected behaviors.

Note:
- Ensure the required classifier files are available in the specified `ClassifierLibraryFolder`.
- Running DeepLabCut on the video is optional; uncomment and configure DLC lines if analysis is needed.

"""


from AniML_utils_GeneralFunctions import *
from AniML_VideoLabel import *
from ARBEL_utils_Predict import *
from AniML_utils_PoseFeatureExtraction import *
from AniML_utils_PixBrightnessFeatureExtraction import *
import glob
from tqdm import tqdm

tic()
#%% USER DEFINED FIELDS
#######################

create_videos=1 # Slower; Adjust resolution (0-to-1) for faster results.
runDLC=0 # If videos were not DeepLabCutted then true

fps = 45
PoseDataFileType='h5'
#Crop video scoring (from frame to frame)
From,To=(0, None)

''' 1. Define behavior video folder '''
# Form: Experiment  -> Experiments (mouse) -> i_Trial(Before, drug, after, timepoints, etc.)
# Example: Project (DARPA) > Experiment (Capsaicin 0.1%, morphine, Date) > Subject Trial files (*.H5, *.avi)
Project = r'/mnt/hd0/Pain_ML_data/ARBEL/' #Insert Project folder path
Rig='DarkBottomUp' #Insert classifier rig
Experiments =['02.24.24_SN'] #Insert experiment folder name

pix_threshold=0.3
bp_pixbrt_list=['rhpaw', 'lhpaw','snout'] #Refer to the DLC body part for hind left paw, hind right paw, and snout
                                          #You may need to change this for your dataset
trim_pose_data = True # Edits datafile in-place to only include inputs for ARBEL data. Useful if pose data tracks significantly more than 9 bodyparts.
if trim_pose_data:
    # Inclued four paws, snout, neck, centroid, tail base and tail end
    # These names should correspond to what the bodyparts are named in your DLC model
    bodyparts_to_include = ('lfpaw','rfpaw','lhpaw','rhpaw','snout','neck','sternumtail','tailbase','tailtip')
square_size=[40,40,40]


'''2. Define classifiers'''
ClassifierLibraryFolder = rf'/mnt/md0/dev/ARBEL_beta/ARBEL_Classifiers/DarkBottomUp/'
Behavior_classifiers = [
                       'ARBEL_Flinching.pkl',
                       'ARBEL_LickingBiting.pkl',
                       'ARBEL_Grooming.pkl',
                    ]
# OPTIONAL
# Rename columns in input dataframe (X) to match that of classifier features
rename_X_features = True
if rename_X_features:
    #Columns of dataframe X will be replaced according to this map
    rename_map = {
        'lhpaw': 'hlpaw',
        'rhpaw': 'hrpaw',
        'lfpaw': 'flpaw',
        'rfpaw': 'frpaw',
        'sternumtail': 'centroid'
        #Extend as needed
    }
#%% RUN! Automatic Recognition of Behavior Enhance with Light
#############################################################
timestamp=datetime.now().strftime("%H%M")
'''Run by the list of experiments (Folders)'''
for Experiment in Experiments:
    closeall()
    Experiment_path = Project + f'{Experiment}/'
    OutputFolder = 'ARBEL_output'
    os.makedirs(Experiment_path + OutputFolder, exist_ok=True)

    #DLC pose estimation labeling - DLC doesnt re-analyze if it recognized DLC files in the folder
    videos_folder = Experiment_path + f'/Videos/'
    if runDLC:
        print(f'DeepLabCut will run only on files that were not analyzed by DLC....')
        # Insert DeepLabCut config file to 'config_path'
        import deeplabcut
        dlc_config_path=r'\config.yaml'
        deeplabcut.analyze_videos(dlc_config_path,videos_folder, save_as_csv=False, gputouse=0)

    # Experiment in Folders
    pose_file_list = glob.glob(videos_folder + '/*.' + PoseDataFileType)
    pose_file_list = sorted([os.path.basename(file) for file in pose_file_list])  # for list of sessionfiles without folder
    pose_ID = sorted([os.path.basename(file).split('DLC')[0] for file in pose_file_list])
    data_summary=pd.DataFrame()
    data_subjects = pd.DataFrame(pose_ID, columns=[f'{Experiment}_Subject'])

    print(f'Experiment: {Experiment} - with {Behavior_classifiers}')
    for i_file in tqdm(range(0, len(pose_file_list)), desc="Processing pose files"):
        print(f'Preparing data and extracting features (X) for classification: {pose_file_list[i_file]}')
        pose_data_file = pose_file_list[i_file]
        vid_file = pose_data_file.split('DLC')[0] + '.avi'
        DataID = pose_data_file[0:pose_data_file.find('.')]
        if pose_data_file.find('DLC') > -1:
            DataID = DataID[:pose_data_file.find('DLC')]
        data_path = videos_folder + pose_data_file

        num_bodypart_columns = len(pd.read_hdf(data_path).columns.get_level_values(1))
        if num_bodypart_columns > 9:
            if trim_pose_data:
                print(f'Trimming pose_data_file...')
                trim_DLC_bodyparts(data_path, bodyparts_to_include, save_inplace=True)
            else:
                raise UserWarning(f'''
                        DLC file {pose_data_file} contains tracking data for {num_bodypart_columns}, but ARBEL is expecting
                        9. This may cause longer processing times during feature extraction.
                         
                        Set 'trim_pose_data' to True in ARBEL_AutoScore.py to fix this.
                        ''')

        X = ARBEL_ExtractFeatures(pose_data_file=data_path,
                                  video_file_path=videos_folder + '/' + vid_file,
                                  bp_pixbrt_list=bp_pixbrt_list,
                                  square_size=square_size,
                                  pix_threshold=pix_threshold)
        subject_summary = pd.DataFrame()
        if rename_X_features:
            # Rename features matrix X to match feature names in XGBoost object
            X = rename_feature_columns(X, rename_map)

        '''Run the data with all the selected classifiers'''
        subject_y_pred_filts=pd.DataFrame()
        for Behavior_classifier in Behavior_classifiers:
            clf_model_path = ClassifierLibraryFolder + Behavior_classifier

            with open(clf_model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            Behavior_type = loaded_model['Behavior_type']
            Behavior_join= ''.join(Behavior_type[:])
            if rename_X_features:
                # Check if renamed features successfully match the loaded model
                missing_features = set(loaded_model['clf_model'].feature_names_in_) - set(X.columns)
                if missing_features:
                    print(f'{len(missing_features)} features in {Behavior_classifier} not found in features matrix. solving...')
                    missing_cols = set(X.columns) - set(loaded_model['clf_model'].feature_names_in_)
                    if len(missing_features) == len(missing_cols):
                        # Discrepancy may be due to naming swap
                        X = swap_bodypart_names(missing_features, X, 'hrpaw','flpaw')
            y_pred_filt, y_pred = ARBEL_Predict(clf_model_path, X)

            y_pred=y_pred.iloc[From:To,:]
            y_pred_filt=y_pred_filt.iloc[From:To,:]

            ##### Save prediction to CSV file
            os.makedirs(Experiment_path + OutputFolder + f'/{DataID}_Behavior', exist_ok=True)
            y_pred_filt.columns=[''.join(Behavior_join)]
            y_pred_filt.to_csv(Experiment_path + OutputFolder + f'/{DataID}_Behavior/' + DataID + '_' + ''.join(Behavior_join) + '_ML.csv', index=False)
            subject_y_pred_filts = pd.concat([subject_y_pred_filts,y_pred_filt],axis=1)
            ##### 2.2 Save summary file
            total_behavior_time = pd.DataFrame([round(np.sum(np.array(y_pred_filt))/fps,2)], columns=[f'{Behavior_join} time (s)'])
            # mean_bout_duration = float(np.mean(np.transpose(find_consecutive_repeats(y_pred_filt)[np.where(find_consecutive_repeats(y_pred_filt)[:, 0] == True), 1])/fps))
            # new_col=pd.DataFrame([[total_behavior_time,mean_bout_duration]],
            #                      columns=[f'{Behavior_join} time (s)', f'{Behavior_join} bout mean (s)'])
            subject_summary =pd.concat([subject_summary, total_behavior_time],axis=1)

        ##### Create video (optional if 'create_videos==1')
        if create_videos==1:
                print("Writing video...")
                LabelVideo(VideoName=DataID,
                           Folder=videos_folder,
                           OutputFolder=Experiment_path + OutputFolder + f'/{DataID}_Behavior/',
                           BehaviorLabels=subject_y_pred_filts,
                           InputFileType='.avi',
                           OutputFileType='.avi',
                           FrameCount=True,
                           Resolution_factor=1,
                           pix_threshold=pix_threshold * 256,
                           )
        data_summary = pd.concat([data_summary, subject_summary], axis=0)
    data_summary = pd.concat([data_subjects, data_summary.reset_index(drop=True)], axis=1)
    data_summary.to_csv(Experiment_path + OutputFolder + '/Behavior_summary_' + Experiment + '_' + timestamp + '.csv')
print('Done.')
toc()
Done() #Play sounds when done.

