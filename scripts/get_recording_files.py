import os
import warnings
from typing import Dict, Tuple, List, Union


def get_recording_outputs(recording_dir: Union[str, os.PathLike]) -> Dict[str, Union[str, os.PathLike]]:
    """
    Processes a recording dir to get the path of all output files to be used for downstream analysis

    Parameters:
    recording_dir (str): The path of the recording directory. function looks for the following files:
        palmreader : filename of 'features.h5'
        kpms : ends with 'kpms.csv'
        deg : ends with 'deg.csv'
        dlc : ends with '000.h5'
        trans : ends with 'trans.avi'
        ftir: ends with 'trans.avi'

        This sort of directory structrue is achieved by running standardize_data_filenames with copy_kpms_result

    Returns:
    output_files (Dict): A dictionary where the keys are the class of output files and the values are the corresponding filepath
    """

    def make_abs(path: Union[str, os.PathLike], root=recording_dir) -> Union[str, os.PathLike]:
        return os.path.join(root, path)

    keys = ['palmreader', 'kpms', 'deg', 'dlc', 'trans', 'ftir']
    output_files = {k: None for k in keys}
    output_files['filename'] = os.path.basename(recording_dir)

    for file in os.listdir(recording_dir):
        if file == 'features.h5':
            output_files['palmreader'] = make_abs(file)
        elif file.endswith('kpms.csv'):
            output_files['kpms'] = make_abs(file)
        elif file.endswith('deg.csv'):
            output_files['deg'] = make_abs(file)
        elif file.endswith('000.h5'):
            output_files['dlc'] = make_abs(file)
        elif file.endswith('trans.avi'):
            output_files['trans'] = make_abs(file)
        elif file.endswith('ftir.avi'):
            output_files['ftir'] = make_abs(file)
    for k, v in output_files.items():
        if v is None:
            message = f'No file found for output file {k} in folder {os.path.basename(recording_dir)}'
            warnings.warn(message)

    return output_files


def get_output_files_by_exp_group(data_dir: Union[str, os.PathLike]) -> List[Tuple[str, List[Dict[str, Union[str, os.PathLike]]]]]:
    """
    Gets paths of all output files for all recordings in data_dir. Output retains information on which experimental
    group the files are part of. Experimental group names are determined by folder in the data_dir directory. So it is
    essential to have the structure correctly.

    :param data_dir: path of root data dir. Expected to have structure similar to this
    ├── videos/
                │   ├── exp_group_1/
                │   │   ├── recording
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_body.avi
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_ftir.avi
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_bodyDLC_resnet50_arcteryx500Nov4shuffle1_350000.h5
                │   │   │   ├── other output files...

    :return: List of (group_name, recordings) tuples where:
    - group_name (str): Experimental group name (from parent directory)
    - recordings (List[Dict]): List of recording data, each a dict with keys:
        ['palmreader', 'kpms', 'deg', 'dlc', 'trans', 'ftir'] (subject to future changes)
      Values are absolute paths to the corresponding output files.
    """
    videos = os.path.join(data_dir, 'videos')
    exp_group_tuples = []

    # split by experimental group
    for exp_group in os.listdir(videos):
        exp_group_abs_path = os.path.join(videos, exp_group)
        if not os.path.isdir(exp_group_abs_path):
            continue
        recordings = os.listdir(exp_group_abs_path)
        if recordings is None:
            message = f'No files found in {exp_group}. Expecting a directory of recordings for a certain experimental group'
            warnings.warn(message)
            continue

        # get recordings within each experimental groups and append dicts of all their output file paths
        recording_output_dicts = []
        for recording in recordings:
            if recording[0] == '.':
                # allows us to skip hidden files
                print(f'skipping hidden folder {recording}')
                continue
            recording_path = os.path.join(exp_group_abs_path, recording)
            if os.path.isdir(recording_path):
                recording_output_dicts.append(get_recording_outputs(recording_path))
            else:
                print(f'argument is not a directory: recording "{recording}" in {exp_group}')
        # Append a tuple of (exp group name, list of dicts for all recordings found)
        exp_group_tuples.append((exp_group, recording_output_dicts))
    return exp_group_tuples


def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    get_output_files_by_exp_group(data_dir)


if __name__ == "__main__":
    main()
