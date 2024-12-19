"""Utility functions for extracting data from body pose coordiates
## take body parts locations x,y cords (bp_xcords, bp_ycord)
## returns distances, movement speed, etc"""
from AniML_utils_GeneralFunctions import *
import numpy as np
import pandas as pd
import cv2
import gc
import os
from datetime import datetime
from scipy.ndimage import convolve1d
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns

def getBPcords(data_file, bp_include_list=None, x_scale=1, y_scale=1, Flip=False, Filter=False):
    # print("Getting x, y body part coordinates.")

    # Read the data file into a DataFrame
    data = open_file_as_dataframe(data_file)

    if (x_scale!=1 or y_scale!=1):
        print(f'Scaling x,y by {x_scale},{y_scale}')
        for col in data.columns:
            if '_x' in col:  # Apply to columns with '_x' (indicating x-coordinates)
                data[col] = data[col] * x_scale
            if '_y' in col:  # Apply to columns with '_x' (indicating x-coordinates)
                data[col] = data[col] * y_scale

    # Extract X_test and y coordinates
    bp_xcord = data.iloc[:, ::3].reset_index(drop=True)
    bp_ycord = data.iloc[:, 1::3].reset_index(drop=True)

    # If include_list is provided, filter the columns
    if bp_include_list != None:
        print(f'Bodyparts used: {bp_include_list}')
        included_columns_x = [col for col in bp_xcord.columns if any(substr in col for substr in bp_include_list)]
        included_columns_y = [col for col in bp_ycord.columns if any(substr in col for substr in bp_include_list)]

        bp_xcord = bp_xcord[included_columns_x]
        bp_ycord = bp_ycord[included_columns_y]

    if Flip:
        print('Flipping limb sides')
        columns_to_flip = [('hlpaw_x', 'hrpaw_x'), ('flpaw_x', 'frpaw_x'),
                           ('hlpaw_y', 'hrpaw_y'), ('flpaw_y', 'frpaw_y')]
        for col1, col2 in columns_to_flip:
            bp_xcord[col1], bp_xcord[col2] = bp_xcord[col2].copy(), bp_xcord[col1].copy()
            bp_ycord[col1], bp_ycord[col2] = bp_ycord[col2].copy(), bp_ycord[col1].copy()

    if Filter:
        print('Applying moving window filter')
        bp_xcord = moving_window_filter(bp_xcord, 3, 0.1)
        bp_ycord = moving_window_filter(bp_ycord, 3, 0.1)

    return bp_xcord, bp_ycord

def getBPnames(data_file):
    bp_xcord, _ = getBPcords(data_file)
    bpnames = [col.replace("_x", "") for col in bp_xcord.columns]
    return bpnames

def getBPprob(data_file):
    data = open_file_as_dataframe(data_file)
    bp_prob = data.iloc[0:, 2::3] # find where pose prob is under thresh
    bp_prob = bp_prob.reset_index()
    bp_prob = bp_prob.drop('index', axis=1)
    return bp_prob

"""def bp_distances(bp_xcord, bp_ycord):
    # Take the x,y position of each body part (BP) and check the distance between it and the other BPs
    BP_distances = pd.DataFrame()
    for i in range(len(bp_xcord.columns)):
        for j in range(i + 1, len(bp_xcord.columns)):
            distances = np.sqrt((bp_xcord.iloc[:, i] - bp_xcord.iloc[:, j])**2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, j])**2)
            bp1name = bp_xcord.columns[i].replace("_x", "")  # e.g. get rid of the "_x" in "Nose_x"
            bp2name = bp_xcord.columns[j].replace("_x", "")
            column_name = f"{bp1name}_{bp2name}"
            BP_distances[column_name] = distances
    return BP_distances"""

def bp_distances(bp_xcord, bp_ycord):
    # List to hold all new columns before concatenation
    new_columns = []

    for i in range(len(bp_xcord.columns)):
        for j in range(i + 1, len(bp_xcord.columns)):
            distances = np.sqrt(
                (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, j]) ** 2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, j]) ** 2)
            bp1name = bp_xcord.columns[i].replace("_x", "")  # e.g. get rid of the "_x" in "Nose_x"
            bp2name = bp_xcord.columns[j].replace("_x", "")
            column_name = f"Dis_{bp1name}-{bp2name}"
            # column_name = f"{bp1name}_{bp2name}"
            # Store the new column as a DataFrame with the column name
            new_columns.append(pd.DataFrame({column_name: distances}))

    # Concatenate all new columns with the original DataFrame
    BP_distances = pd.concat(new_columns, axis=1)

    # print("Scaling distances...")
    # from sklearn import preprocessing
    # scaling_method = preprocessing.MinMaxScaler()
    # BP_distances = pd.DataFrame(scaling_method.fit_transform(BP_distances), columns=BP_distances.columns)
    # return BP_distances

    return BP_distances


def angle(bp_xcord, bp_ycord):
    # Take the x, y position of each body part (BP) and check the angles between it and the other BPs
    BP_angles = pd.DataFrame()
    bp_columns = bp_xcord.columns
    import itertools
    permutations = list(itertools.permutations(range(len(bp_columns)), 2))
    unique_permutations = []
    for perm in permutations:
        reverse_perm = perm[::-1]  # Reverse the permutation
        if perm not in unique_permutations and reverse_perm not in unique_permutations:
            unique_permutations.append(perm)
    for i_p in range(len(unique_permutations)):
        for i_bp in range(len(bp_columns)):
            i,j = unique_permutations[i_p]
            k = i_bp
            if (i != k) and (j != k): #dont check an angle from a poin to itslef

                bp1name = bp_columns[i].replace("_x", "")  # e.g. get rid of the "_x" in "snout_x"
                bp2name = bp_columns[j].replace("_x", "")
                bp3name = bp_columns[k].replace("_x", "")
                # print(f'{bp1name}_{bp2name} > {bp3name}')
                AC = (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, k])**2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, k])**2
                BC = (bp_xcord.iloc[:, j] - bp_xcord.iloc[:, k])**2 + (bp_ycord.iloc[:, j] - bp_ycord.iloc[:, k])**2
                AB = (bp_xcord.iloc[:, i] - bp_xcord.iloc[:, j])**2 + (bp_ycord.iloc[:, i] - bp_ycord.iloc[:, j])**2
                AC = np.sqrt(AC)
                BC = np.sqrt(BC)
                AB = np.sqrt(AB)
                AngleC = np.arccos((BC**2 + AC**2 - AB**2) / (2 * AC * BC))  # Law of cosines
                AngleC = np.rad2deg(AngleC)
                angle_column = f'Ang_{bp1name}-{bp3name}-{bp2name}'
                # angle_column = f'deg_{bp1name}_{bp3name}_{bp2name}'
                BP_angles = pd.concat([BP_angles, pd.DataFrame(AngleC, columns=[angle_column])], axis=1)
    return BP_angles


def angle_change(bp_xcord,bp_ycord, t=1):
# Take the x,y position of each body part (BP) and check the distance between it and the other BPs
    BP_angles=pd.DataFrame(angle(bp_xcord,bp_ycord))
    BP_angles_change=BP_angles.diff(periods=t)
    new_columns = ['Chn_' + col_name for col_name in BP_angles_change.columns + str(t)]
    BP_angles_change.columns = new_columns
    return pd.DataFrame(BP_angles_change)

def bp_velocity(bp_xcord, bp_ycord, t=1):
    # Take the x, y position of each body part (BP) and return the movement velocity in a time t period
    BP_velocity = pd.DataFrame()
    for i in range(len(bp_xcord.columns)):
        diff_distances_x = bp_xcord.iloc[:, i].diff(periods=t)
        diff_distances_y = bp_ycord.iloc[:, i].diff(periods=t)
        distance = diff_distances_x ** 2 + diff_distances_y ** 2
        velocity = np.sqrt(distance) / np.abs(t)
        velocity.name = bp_xcord.columns[i].replace("_x", "") + f'_Vel{t}'
        BP_velocity = pd.concat([BP_velocity, velocity], axis=1)
    # Fill missing values and set velocities for first/last time points to zero
    BP_velocity.fillna(0, inplace=True)
    if t > 0:
        BP_velocity.iloc[:t, :] = 0
    elif t < 0:
        BP_velocity.iloc[t:, :] = 0
    return BP_velocity

def distances_velocity(bp_dist, t=1):
    bp_dist_vel=bp_dist.diff(periods=t)
    bp_dist_vel.columns = [col + "_Vel" + str(t) for col in bp_dist_vel.columns]
    return bp_dist_vel

def bp_inFrame(data_file,BPprob_thresh):
    inFrame = getBPprob(data_file)>BPprob_thresh
    inFrame.columns = [col.split('_')[0] + f"_inFrame_p{BPprob_thresh}" for col in inFrame.columns]
    return inFrame

def PoseFeatureExtract(data_file, dt_vel=2, x_scale=1, y_scale=1, Flip=False, Filter=False, BPprob_thresh=0.8, bp_include_list=None):
    #Feature extraction
    bp_xcord, bp_ycord = getBPcords(data_file, bp_include_list=bp_include_list, x_scale=x_scale, y_scale=y_scale, Flip=Flip, Filter=Filter)
    X = pd.concat([bp_distances(bp_xcord, bp_ycord),

                   bp_velocity(bp_xcord, bp_ycord, 1),
                   bp_velocity(bp_xcord, bp_ycord, 1).sum(axis=1).to_frame(name='sum_Vel1'),  #.applymap(lambda x: x if x >= 0.75 else 0),#wasnt in resubmission 8/18/24

                   bp_velocity(bp_xcord, bp_ycord, dt_vel),
                   bp_velocity(bp_xcord, bp_ycord, dt_vel).sum(axis=1).to_frame(name=f'sum_Vel{dt_vel}'),  #.applymap(lambda x: x if x >= 0.75 else 0),

                   bp_velocity(bp_xcord, bp_ycord, 10),
                   bp_velocity(bp_xcord, bp_ycord, 10).sum(axis=1).to_frame(name='sum_Vel10'),  #.applymap(lambda x: x if x >= 0.75 else 0),

                   angle(bp_xcord, bp_ycord),

                   bp_inFrame(data_file,BPprob_thresh)
                   # distances_velocity(bp_distances(bp_xcord, bp_ycord),dt_vel),
                   # MinMovement,MaxMovement, MeanMovement, SumMovement,
                   ],
                   axis=1)
    # reset the row index number
    X = X.reset_index()
    X = X.drop('index', axis=1)
    return X


def remove_correlated_features(X, corr_threshold, plot=False):
    """
    Removes features from the DataFrame X that have a correlation higher than the specified threshold. Out of the two
    highly-correlated features, it will drop the on that has higher mean correlations with other features
    (Gozzi,...,Respopovic, Med 2024).

    Parameters:
    X (pd.DataFrame): The input feature matrix.
    threshold (float): The correlation threshold above which features will be removed. Default is 0.9.

    Returns:
    pd.DataFrame: The feature matrix with highly correlated features removed.
    """
    # # Calculate the correlation matrix
    # corr_matrix = X.corr().abs()
    #
    # # Create a mask to identify the upper triangle of the correlation matrix
    # upper_triangle = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
    #
    # # Find index of feature columns with correlation greater than the threshold
    # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    # print (f'dropp {len(to_drop)} colums with correlation threshold of {corr_threshold}')
    # # Drop the columns
    # X_reduced = X.drop(columns=to_drop)

    if plot:
        plt.figure()
        plt.suptitle('Feature Correlation Matrix')
        # plot corr matrix
        plt.subplot(1,2,1)
        corr_matrix = X.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
        plt.title('Before Reduction')
        plt.tight_layout()
        plt.show()

    # Step 1: Compute the absolute correlation matrix
    corr_matrix = X.corr().abs()  # Calculate the absolute correlation matrix
    # Create a mask for the upper triangle (to avoid duplicate comparisons)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    high_corr_pairs = corr_matrix.mask(mask).stack().reset_index()
          # Find all feature pairs with correlations greater than or equal to 0.9
    high_corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']  # Rename the columns for clarity
    # Filter out feature pairs with absolute correlation >= 0.9
    high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] >= corr_threshold].reset_index()


    # pd.set_option('display.max_rows', None, 'display.max_columns', None)
    print(high_corr_pairs)
    # Step 2: Initialize an empty set to track the features to drop
    to_drop = set()

    # Step 3: Iterate through each pair of highly correlated features and drop the one with the higher mean correlation
    for _, row in high_corr_pairs.iterrows():  # Ensure you use iterrows() correctly here
        feature1 = row['Feature_1']
        feature2 = row['Feature_2']

        # Skip if one of the features is already marked for dropping
        if feature1 in to_drop or feature2 in to_drop:
            continue

        # Calculate the mean correlation of feature1 and feature2 with all other features
        mean_corr1 = corr_matrix[feature1].mean()
        mean_corr2 = corr_matrix[feature2].mean()

        # Drop the feature with the higher mean correlation
        if mean_corr1 > mean_corr2:
            to_drop.add(feature1)
        else:
            to_drop.add(feature2)

    X_reduced = X.drop(columns=to_drop)

    if plot:
        plt.subplot(1,2,2)
        corr_matrix = X_reduced.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
        plt.title('After Reduction')
        plt.tight_layout()
        plt.show()
    print('Dropping:')
    print(to_drop)

    return X_reduced


import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
def AniML_FindThresh(X, y, model, k=5, min_thr=0, max_thr=1, coarse_increment=0.05, fine_increment=0.02,
                     search_radius=0.1, n_jobs=1):
    """
    Finds the best threshold for a given model by evaluating F1-scores using k-fold cross-validation
    with a two-stage search: coarse search and fine search for threshold optimization.

    Parameters:
    X (pd.DataFrame): Features matrix.
    y (np.array or pd.Series): Target values.
    model: Classification model.
    k (int): Number of folds for cross-validation.
    min_thr (float): Minimum threshold value to evaluate.
    max_thr (float): Maximum threshold value to evaluate.
    coarse_increment (float): Step size for initial coarse threshold search.
    fine_increment (float): Step size for refined threshold search around the coarse maximum.
    search_radius (float): Range around the best threshold from coarse search for fine-tuning.
    n_jobs (int): Number of parallel jobs to run.

    Returns:
    list: Best thresholds for each fold.
    """

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_thresholds = []

    print(f'Finding the best threshold across {k}-folds with a two-stage search (n_jobs={n_jobs})...')

    def evaluate_fold(train_idx, test_idx):
        # Split the data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the model
        model.fit(X_train, y_train)

        # Coarse Search
        best_f1, best_thresh = 0, None
        for threshold in np.arange(min_thr, max_thr, coarse_increment):
            y_pred_thr = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_thr)

            if f1 > best_f1:
                best_f1, best_thresh = f1, threshold

        print(f"Coarse search best F1 = {best_f1:.4f} at threshold = {best_thresh:.2f}")

        # Fine Search around best threshold from coarse search
        fine_min_thr = max(min_thr, best_thresh - search_radius)
        fine_max_thr = min(max_thr, best_thresh + search_radius)
        for threshold in np.arange(fine_min_thr, fine_max_thr, fine_increment):
            y_pred_thr = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_thr)

            if f1 > best_f1:
                best_f1, best_thresh = f1, threshold

        print(f"Refined search best F1 = {best_f1:.4f} at threshold = {best_thresh:.2f}")
        return best_thresh, best_f1

    # Use Parallel to run evaluation on multiple folds concurrently
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(train_idx, test_idx) for train_idx, test_idx in kf.split(X)
    )

    # Collect the best thresholds from each fold
    for fold_idx, (best_thresh, best_f1) in enumerate(results):
        best_thresholds.append(best_thresh)

    # Calculate the mean of the best thresholds across all folds
    mean_best_thresh = np.mean(best_thresholds)
    print(f"\nMean best threshold across folds: {mean_best_thresh:.2f}")

    return best_thresholds

#Unused features
    # BPprobs=getBPprob(data_file)
    # MeanMovement = pd.DataFrame(bp_velocity(bp_xcord, bp_ycord, dt_vel).mean(axis=1))
    # MeanMovement.columns = ['MeanMovement']
    # SumMovement = pd.DataFrame(bp_velocity(bp_xcord, bp_ycord, dt_vel).sum(axis=1))
    # SumMovement.columns = ['SumMovement']
    # MinMovement= pd.DataFrame(bp_velocity(bp_xcord, bp_ycord, dt_vel).min(axis=1))
    # MinMovement.columns = ['MinMovement']
    # MaxMovement= pd.DataFrame(bp_velocity(bp_xcord, bp_ycord, dt_vel).max(axis=1))
    # MaxMovement.columns = ['MaxMovement']
