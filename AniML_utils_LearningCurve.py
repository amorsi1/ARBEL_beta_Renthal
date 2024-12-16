import numpy as np
import random
import time
from sklearn.metrics import f1_score, accuracy_score

def AniML_learning_curve(clf_toy, X, y, train_sizes=[0.001, 0.25, 0.5, 0.75, 1], method='f1', best_thresh=0.5, cv=5, X_val=[], y_val=[]):
    """ X_val and y_val are in the case where we want to create a learning curve for a validation set not used during training"""
    test_scores = np.zeros([cv, len(train_sizes)])
    # X_values=X_test.values() #faster not working with dataframes, maybe?
    if (len(X_val) > 0 and len(y_val) > 0):
        if train_sizes[0] < 1: #if it's given integers
            train_sizes = train_sizes * len(y_val)
        print('Testing scores on validation data (not included in training)')
        if (len(y)<np.max(train_sizes)):
            train_sizes[np.where(train_sizes==np.max(train_sizes))]=len(y)
            print(f'Train sizes ({np.max(train_sizes)}) larger than data size ({len(y)}).\n'
                  f'Adjusting the train sizes to fit the number of positive bouts.')
            return None
    else:
        if train_sizes[0] < 1:
            train_sizes = train_sizes * len(y_val)

    print(f'Train sizes: {train_sizes}.')
    print(f'Starting {cv*len(train_sizes)} iterations...')
    startTime_for_tictoc0 = time.time()
    for i_train, train_size in enumerate(train_sizes):
        for i_seed in range(0,cv):
            startTime_for_tictoc = time.time()
            random.seed(i_seed)
            if np.issubdtype(train_sizes.dtype, np.integer):
                idx = random.sample(range(len(X)), int(np.round(train_size)))
                train_sizes_type='int'
            else:
                idx = random.sample(range(len(X)), int(np.round(train_size * len(X))))
                train_sizes_type = 'float'

            X_toy = X.iloc[idx[0:int(((cv - 1) / cv) * len(idx))], :] ## X_toy = X_values[idx[0:int(((cv-1)/cv)*len(idx))]]
            y_toy = y[idx[0:int(((cv - 1) / cv) * len(idx))]]  ## X_toy_test= X_values[idx[int(((cv-1)/cv)*len(idx)):],:]
            if (len(X_val)>0 and len(y_val)>0):
                X_toy_test = X_val
                y_toy_test = y_val
            else:
                X_toy_test = X.iloc[idx[int(((cv-1)/cv)*len(idx)):],:]
                y_toy_test = y[idx[int(((cv-1)/cv)*len(idx)):]]


            clf_toy.fit(X_toy,y_toy)
            y_toy_pred = clf_toy.predict_proba(X_toy_test)[:, 1]>best_thresh
            if method=='f1':
                test_scores[i_seed][i_train] = f1_score(y_toy_pred, y_toy_test)
                print(f'Train size: {train_size} (0-to-1 ratio {np.round(np.sum(y_toy==0) / np.sum(y_toy==1),2)}), iteration: {i_seed+1}/{cv} of {i_train+1}/{len(train_sizes)} '
                      f'trains sizes ------- F1: {np.round(test_scores[i_seed][i_train],5)} | {round(time.time() - startTime_for_tictoc, 2)} sec'),
            if method=='accuracy':
                test_scores[i_seed][i_train] = accuracy_score(y_toy_pred, y_toy_test)
                print(f'Train size: {train_size} (0-to-1 ratio {np.round(np.sum(y_toy==0) / np.sum(y_toy==1),2)}), iteration: {i_seed+1}/{cv} of {i_train+1}/{len(train_sizes)} '
                      f'trains sizes ------- Accuracy: {np.round(test_scores[i_seed][i_train],5)} | {round(time.time() - startTime_for_tictoc, 2)} sec'),

    print(f"Total time: {round(time.time() - startTime_for_tictoc0, 2)/60} minutes.")

    if train_sizes_type == 'float':
        train_sizes_true = len(y)*np.array(train_sizes)
    if train_sizes_type == 'int':
        train_sizes_true = train_sizes

    return train_sizes_true, test_scores.transpose()