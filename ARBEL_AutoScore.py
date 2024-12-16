"""
ARBEL Automatic Behavior Scoring Script

This script is designed for automatic behavior scoring using the ARBEL framework.
It integrates pose estimation, pixel brightness feature extraction, and behavior
classification through machine learning models. The primary purpose of this
pipeline is to automate behavior analysis from video data efficiently.

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
- Feature extraction parameters such as `dt_before`, `pix_threshold`, `bp_list`, and `square_size`.
- Video creation options such as resolution, FPS, and input/output formats.

Workflow:
1. Loop through experimental folders and identify videos for analysis.
2. Extract pose and pixel brightness features from video data.
3. Apply classifiers to predict behaviors and save results.
4. Optionally, generate labeled videos for the detected behaviors.
5. Compile and save summary statistics for each experiment.

Note:
- Ensure the required classifier files are available in the specified `ClassifierLibraryFolder`.
- Running DeepLabCut on the video is optional; uncomment and configure DLC lines if analysis is needed.

"""


from AniML_utils_GeneralFunctions import *
from AniML_VideoLabel import *
from ARBEL_Predict import *
from AniML_utils_PoseFeatureExtraction import *
from AniML_utils_PixBrightnessFeatureExtraction import *
import glob

#%% USER DEFINED FIELDS
#######################

create_videos=1 # Slower; Adjust resolution (0-to-1) for faster results.
fps = 25
PoseDataFileType='h5'
#Crop video scoring (from frame to frame)
From,To=(0, None)

''' 1. Define behavior video folder '''
# Form: Experiment  -> Experiments (mouse) -> i_Trial(Before, drug, after, timepoints, etc.)
# Example: Project (DARPA) > Experiment (Capsaicin 0.1%, morphine, Date) > Subject Trial files (*.H5, *.avi)
Project = r'H:\Shared drives\WoolfLab\Sunze\SOD1 ALS\cohort5&6/'
Rig='BlackBox'
Experiments =['1wk'] #'4wk','6wk','7wk','8wk','9wk','10wk','11wk','12wk','13wk','14wk','15wk','16wk','17wk']

dt_before=2
pix_threshold=0.3
bp_list=['hrpaw', 'hlpaw','snout']
square_size=[40,40,40]


'''2. Define classifiers'''
ClassifierLibraryFolder = rf'C:\Users\ch226295\PycharmProjects\AniML\ARBEL_Classifiers\{Rig}/'
Behavior_classifiers = [
                       # 'ARBEL_Flinching.pkl',
                       # 'ARBEL_LickingBiting.pkl',
                       # 'ARBEL_Grooming.pkl',
                       'ARBEL_Rearing.pkl',
                    ]

#%% RUN! Automatic Recognition of Behavior Enhance with Light
#############################################################
timestamp=datetime.now().strftime("%H%M")
'''Run by the list of experiments (Folders)'''
for Experiment in Experiments:
    closeall()
    Experiment_path = Project + f'{Experiment}/'
    OutputFolder = 'ML_Output'
    os.makedirs(Experiment_path + OutputFolder, exist_ok=True)

    #DLC pose estimation labeling - DLC doesnt re-analyze if it recognized DLC files in the folder
    videos_folder = Experiment_path
    if len(glob.glob(videos_folder + '/*.' + PoseDataFileType)) != (len(glob.glob(videos_folder + '/body*.mp4')) + len(glob.glob(videos_folder + '/body*.avi'))):
        print(f'DeepLabCut will run only on files that were not analyzed by DLC....')
        # import deeplabcut
        # # config_path=r'H:\Shared drives\WoolfLab\Omer\DeepLabCutAlex_UpdatedJan24\config.yaml'
        # # config_path = r'H:\Shared drives\WoolfLab\Omer\PalmReaderEEG -Omer -2024-03-19\config.yaml'
        # deeplabcut.analyze_videos(config_path,videos_folder, save_as_csv=False, gputouse=0)
        # deeplabcut.create_labeled_video(config_path, videos_folder)

    # Experiment in Folders
    pose_file_list = glob.glob(videos_folder + '/*.' + PoseDataFileType)
    pose_file_list = sorted([os.path.basename(file) for file in pose_file_list])  # for list of sessionfiles without folder
    pose_ID = sorted([os.path.basename(file).split('DLC')[0] for file in pose_file_list])
    data_summary=pd.DataFrame()
    data_subjects = pd.DataFrame(pose_ID, columns=[f'{Experiment}_Subject'])

    print(f'Experiment: {Experiment} - with {Behavior_classifiers}')
    for i_file in range(0, len(pose_file_list)):
        print(f'Preparing data and extracting features (X) for classification: {pose_file_list[i_file]}')
        pose_data_file = pose_file_list[i_file]
        vid_file = pose_data_file.split('DLC')[0] + '.avi'
        DataID = pose_data_file[0:pose_data_file.find('.')]
        if pose_data_file.find('DLC') > -1:
            DataID = DataID[:pose_data_file.find('DLC')]
        data_path = videos_folder + pose_data_file
        print("Change the order of ['hrpaw', 'hlpaw','snout'] to  ['hlpaw', 'hrpaw','snout'] if there is a warning about it.")
        X = ARBEL_ExtractFeatures(data_path, video_file_path=videos_folder + '/' + vid_file, bp_list=bp_list,
                                  dt_before=dt_before, square_size=square_size, pix_threshold=pix_threshold, create_video=False,
                                  Flip=False, n_jobs=32, bp_include_list=None, save_feature_mat=0)
        subject_summary = pd.DataFrame()

        '''Run the data with all the selected classifiers'''
        for Behavior_classifier in Behavior_classifiers:
            clf_model_path = ClassifierLibraryFolder + Behavior_classifier

            with open(clf_model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            Behavior_type = loaded_model['Behavior_type']

            Behavior_join= ''.join(Behavior_type[:])

            y_pred_filt, y_pred = ARBEL_Predict(clf_model_path, X)

            y_pred=y_pred.iloc[From:To,:]
            y_pred_filt=y_pred_filt.iloc[From:To,:]

            ##### Save prediction to CSV file
            os.makedirs(Experiment_path + OutputFolder + f'/{DataID}_Behavior', exist_ok=True)
            y_pred_filt.columns=[''.join(Behavior_join)]
            y_pred_filt.to_csv(Experiment_path + OutputFolder + f'/{DataID}_Behavior/' + DataID + '_' + ''.join(Behavior_join) + '_ML.csv', index=False)

            ##### 2.2 Save summary file
            total_behavior_time = round(np.sum(np.array(y_pred_filt))/fps,2)
            mean_bout_duration = float(np.mean(np.transpose(find_consecutive_repeats(y_pred_filt)[np.where(find_consecutive_repeats(y_pred_filt)[:, 0] == True), 1])/fps))
            new_col=pd.DataFrame([[total_behavior_time,mean_bout_duration]],
                                 columns=[f'{Behavior_join} time (s)', f'{Behavior_join} bout mean (s)'])
            subject_summary =pd.concat([subject_summary ,new_col],axis=1)

            ##### Create video (optional if 'create_videos==1')
            if create_videos==1:
                    print("Writing video(s)...")
                    os.makedirs(Experiment_path + OutputFolder + f'/{DataID}_Behavior/' + OutputFolder, exist_ok=True)
                    LabelVideo(VideoName=DataID, Folder=videos_folder,
                               OutputFolder=Experiment_path + OutputFolder + f'/{DataID}_Behavior/',
                               BehaviorLabels = y_pred_filt,
                               BehaviorLabels1 = [],
                               InputFileType = '.avi',
                               Behavior_name = ''.join(Behavior_join[:]),
                               OutputFileType = '.mp4',
                               FrameCount = True,
                               Resolution_factor = 0.33, fromFrame = 0,  #toFrame=,
                               pix_threshold = 125, only_pix_threshold = False, colormap = 'coolwarm')
        data_summary = pd.concat([data_summary, subject_summary], axis=0)
    data_summary = pd.concat([data_subjects, data_summary.reset_index(drop=True)], axis=1)
    data_summary.to_csv(Experiment_path + OutputFolder + '/Behavior_summary_' + Experiment + '_' + timestamp + '.csv')
print('Done.')

Done() #Play sounds when done.


